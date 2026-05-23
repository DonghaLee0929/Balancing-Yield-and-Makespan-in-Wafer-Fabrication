"""
NSGA-II baseline for the bi-objective HFSP (min makespan, max yield).

test.py 의 학습된 λ-conditioned 모델과 *완전히 동일한 고정 인스턴스* 를 풀어
Hypervolume(HV) 로 비교하기 위한 비교 알고리즘.

인스턴스 (test.py 와 1:1 동일):
  proc_time     : quality_data/P_{p_idx}.csv         (load_problems_from__quality_file)
  wafer_quality : U[wq_min, wq_max], seed 고정 → QualityHelper.sample_wafer_quality 와 동일 RNG
  yield(GT)     : quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv 의
                  (stage,machine) quality factor 곱 (PathPercentileScorer, 평가 전용 ground-truth)

makespan/yield 계산:
  HFSPGraphEnv 를 그대로 *decoder* 로 사용 → 모델 평가(test.py)와 의미가 완전히 동일.

비교 지표:
  Hypervolume. ref(worst) = (hv_m_ref, hv_q_ref), best anchor = (hv_m_best, hv_q_best).
  HV_raw  = 비지배 점들이 ref 까지 지배하는 면적.
  HV_norm = HV_raw / ((hv_m_ref-hv_m_best)·(hv_q_best-hv_q_ref))  ∈ [0,1] (anchor box 기준).

Encoding (FJSP/HFSP 표준 MOEA, e.g. Zhang et al.):
  OS (operation sequence): 길이 J·S 의 job-id 반복열. job j 의 k 번째 등장 = j 의 stage-k op.
                           dispatch 우선순위를 정의하며 precedence 를 자동 보장.
  MA (machine assignment) : 각 op(=j·S+s) 의 stage 내 local machine idx ∈ [0, cnt_s).
                           path(=yield) 와 makespan 에 동시에 영향.
Decode: OS 순서대로 env.step → env 의 slot-fitting/gap-filling 이 timing 을 정함 (insertion decoder).
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from HFSPGraphEnv import HFSPGraphEnv
from FFSProblemDef import load_problems_from__quality_file
from HFSPWrapper import QualityHelper


# =====================================================
# test.py 와 동일한 helper (의미 보존을 위해 그대로 복제)
# =====================================================
def problems_to_edge_proc_time(problems_INT_list, env: HFSPGraphEnv,
                               batch_size: int) -> torch.Tensor:
    """list of (1, J, cnt_s) tensors → (B, num_edges) long. 단일 인스턴스를 B 회 tile.
    edge 순서 규칙은 HFSPGraphEnv.sample_edge_proc_time 과 동일."""
    device = env.device
    machine_offset = torch.tensor(
        np.cumsum([0] + env.machine_cnt_list[:-1]), dtype=torch.long, device=device)
    ept_1 = torch.empty((1, env.num_edges), dtype=torch.long, device=device)
    j_idx = torch.arange(env.num_jobs, device=device)
    for s in range(env.num_stages):
        cnt_s = env.machine_cnt_list[s]
        stage_pt = problems_INT_list[s].to(device).long()
        edge_idx = (j_idx.unsqueeze(1) * env.total_machines
                    + machine_offset[s] + torch.arange(cnt_s, device=device).unsqueeze(0))
        ept_1[:, edge_idx.reshape(-1)] = stage_pt.reshape(1, -1)
    return ept_1.repeat(batch_size, 1)


def schedule_paths_1indexed(env: HFSPGraphEnv) -> np.ndarray:
    """env 의 현재 schedule 에서 (B, J, S) 1-indexed local machine path 추출."""
    BP = env.batch_size
    J = env.num_jobs
    S = env.num_stages
    op_m_3d = env.op_machine.reshape(BP, J, S)
    return (env.machine_local[op_m_3d] + 1).cpu().numpy().astype(np.int64)


def compute_pareto_front(makespan: np.ndarray, yld: np.ndarray) -> np.ndarray:
    """min makespan + max yield 의 비지배 점 indices (makespan 오름차순)."""
    order = np.lexsort((-yld, makespan))
    front, best_y = [], -np.inf
    for i in order:
        if yld[i] > best_y:
            front.append(i)
            best_y = yld[i]
    return np.asarray(front, dtype=np.int64)


class PathPercentileScorer:
    """historical_paths CSV 의 (stage, machine) quality factor 로 path 품질/백분위 계산.
    test.py 의 동명 클래스를 그대로 복제 — yield ground-truth 정의를 1:1 유지."""

    def __init__(self, csv_path: str, machine_cnt_list: list[int]):
        df = pd.read_csv(csv_path)
        S = len(machine_cnt_list)
        max_m = max(machine_cnt_list)
        step_q = np.zeros((S, max_m), dtype=np.float64)
        for s in range(S):
            for m in range(1, machine_cnt_list[s] + 1):
                mask = df[f'Step {s + 1}'] == m
                vals = df.loc[mask, f'Step {s + 1} quality'].unique()
                if len(vals) != 1:
                    raise ValueError(
                        f"(stage {s + 1}, machine {m}) quality 가 {len(vals)} 종류 — "
                        "CSV 가 deterministic 하지 않음")
                step_q[s, m - 1] = vals[0]
        self.step_q = step_q
        self.machine_cnt_list = list(machine_cnt_list)
        self.num_stages = S

        grids = np.meshgrid(*[step_q[s, :machine_cnt_list[s]] for s in range(S)],
                            indexing='ij')
        all_paths = np.stack(grids, axis=-1).reshape(-1, S)
        self.all_products_sorted = np.sort(all_paths.prod(axis=1))
        self.n_paths = int(self.all_products_sorted.size)
        print(f"[percentile] {csv_path}: stages={S}  machine_cnt={machine_cnt_list}  "
              f"total paths={self.n_paths}  product range="
              f"[{self.all_products_sorted[0]:.4f}, {self.all_products_sorted[-1]:.4f}]")

    def path_product(self, paths_1indexed: np.ndarray) -> np.ndarray:
        S = self.num_stages
        if paths_1indexed.shape[-1] != S:
            raise ValueError(
                f"path 마지막 차원 {paths_1indexed.shape[-1]} != num_stages {S}")
        flat = paths_1indexed.reshape(-1, S).astype(np.int64)
        q = np.empty_like(flat, dtype=np.float64)
        for s in range(S):
            q[:, s] = self.step_q[s, flat[:, s] - 1]
        return q.prod(axis=1).reshape(paths_1indexed.shape[:-1])

    def score(self, paths_1indexed: np.ndarray) -> np.ndarray:
        prod = self.path_product(paths_1indexed)
        rank = np.searchsorted(self.all_products_sorted, prod.reshape(-1), side='right')
        pct = rank.astype(np.float64) / self.n_paths * 100.0
        return pct.reshape(prod.shape)


# =====================================================
# Hypervolume (2D, min makespan / max yield)
# =====================================================
def hypervolume(ms: np.ndarray, yld: np.ndarray,
                m_ref: float, q_ref: float) -> tuple[float, np.ndarray]:
    """ref(worst corner) = (m_ref, q_ref). 비지배 점이 ref 까지 지배하는 면적 + 사용된 front idx.

    min makespan / max yield → 비지배 front 는 makespan↑ 일수록 yield↑.
    HV = Σ_i (m_{i+1} − m_i)·(y_i − q_ref),  m_{last+1} = m_ref.
    """
    mask = (ms < m_ref) & (yld > q_ref)
    if not mask.any():
        return 0.0, np.empty(0, dtype=np.int64)
    idx_in = np.where(mask)[0]
    m_in, y_in = ms[idx_in], yld[idx_in]
    order = np.lexsort((-y_in, m_in))            # makespan asc, tie → yield desc
    fm, fy, fidx, best = [], [], [], -np.inf
    for k in order:
        if y_in[k] > best:                       # strict → 동일 yield(더 큰 makespan) 은 지배됨
            fm.append(m_in[k]); fy.append(y_in[k]); fidx.append(idx_in[k]); best = y_in[k]
    hv = 0.0
    for i in range(len(fm)):
        next_m = fm[i + 1] if i + 1 < len(fm) else m_ref
        hv += (next_m - fm[i]) * (fy[i] - q_ref)
    return float(hv), np.asarray(fidx, dtype=np.int64)


# =====================================================
# Decoder — population (OS, MA) → (makespan, yield) via env
# =====================================================
class HFSPDecoder:
    def __init__(self, env, problems_INT_list, env_edge_lookup_np,
                 machine_offset_np, scorer, wq_np, yield_mode='raw',
                 yield_objective='ground_truth', quality_helper=None):
        self.env = env
        self.problems_INT_list = problems_INT_list
        self.lookup = env_edge_lookup_np                  # (num_ops, total_machines)
        self.machine_offset = machine_offset_np           # (S,)
        self.scorer = scorer
        self.wq_np = wq_np                                # (J,)
        self.yield_mode = yield_mode
        self.yield_objective = yield_objective            # 'ground_truth' | 'predicted'
        self.qh = quality_helper                          # 'predicted' 일 때만 사용
        self.J = env.num_jobs
        self.S = env.num_stages
        self.L = env.num_ops
        if yield_objective == 'predicted' and (quality_helper is None or not quality_helper.is_active):
            raise ValueError("yield_objective='predicted' 는 active QualityHelper 가 필요")
        self._wq_t = torch.as_tensor(wq_np, device=env.device, dtype=torch.float32)  # (J,)

    def evaluate(self, OS: np.ndarray, MA: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """OS (B,L), MA (B,num_ops) → (makespan, gt_yield, obj_yield) 각 (B,).

        gt_yield  : ground-truth (scorer; raw/percentile) — HV/plot/리포트 전용.
        obj_yield : NSGA-II 선택용 yield. yield_objective='ground_truth' → gt 와 동일,
                    'predicted' → QualityHelper.compute_yield (모델이 학습 때 쓰는 예측 신호).
        """
        env, J, S, L = self.env, self.J, self.S, self.L
        B = OS.shape[0]
        bidx = np.arange(B)

        # 각 position 의 stage = 그 job 이 지금까지 등장한 횟수 (precedence-safe dispatch).
        counter = np.zeros((B, J), dtype=np.int64)
        stage_pos = np.empty((B, L), dtype=np.int64)
        for t in range(L):
            jt = OS[:, t]
            stage_pos[:, t] = counter[bidx, jt]
            counter[bidx, jt] += 1

        op = OS * S + stage_pos                                   # (B, L) op index
        ma_local = np.take_along_axis(MA, op, axis=1)            # (B, L)
        machine_global = self.machine_offset[stage_pos] + ma_local
        edge = self.lookup[op, machine_global]                   # (B, L) — 항상 valid(>=0)
        edge_t = torch.as_tensor(edge, device=env.device, dtype=torch.long)

        ept = problems_to_edge_proc_time(self.problems_INT_list, env, B)
        env.reset(batch_size=B, edge_proc_time=ept, identical_job=True)
        for t in range(L):
            env.step(edge_t[:, t], last=(t == L - 1))

        ms = env.makespan().cpu().numpy().astype(np.float64)     # (B,)
        paths = schedule_paths_1indexed(env)                     # (B, J, S)
        if self.yield_mode == 'percentile':
            gt_yld = self.scorer.score(paths).mean(axis=1).astype(np.float64)
        else:
            prod_bj = self.scorer.path_product(paths)            # (B, J)
            gt_yld = (prod_bj * self.wq_np).mean(axis=1).astype(np.float64)

        if self.yield_objective == 'predicted':
            wq_bp = self._wq_t.unsqueeze(0).expand(B, -1)        # (B, J)
            obj_yld = self.qh.compute_yield(
                env, wq_bp, aggregate='mean').cpu().numpy().astype(np.float64)
        else:
            obj_yld = gt_yld
        return ms, gt_yld, obj_yld


# =====================================================
# NSGA-II core
# =====================================================
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """minimize 둘 다: a 가 b 를 지배?"""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_nondominated_sort(objs: np.ndarray) -> list[np.ndarray]:
    """objs (N,2) minimize → front index 리스트 (front[0] = 최상위 비지배)."""
    N = objs.shape[0]
    # 벡터화 지배행렬: dom[p,q] = p dominates q
    le = objs[:, None, :] <= objs[None, :, :]
    lt = objs[:, None, :] <  objs[None, :, :]
    dom = le.all(axis=2) & lt.any(axis=2)
    n_dom = dom.sum(axis=0)                  # 자신을 지배하는 개체 수
    fronts, cur = [], np.where(n_dom == 0)[0]
    n_dom = n_dom.copy()
    while cur.size:
        fronts.append(cur)
        nxt_counts = dom[cur].sum(axis=0)    # cur 가 지배하는 개체들의 카운트 감소
        n_dom = n_dom - nxt_counts
        n_dom[cur] = -1                      # 이미 배정된 개체 제외
        cur = np.where(n_dom == 0)[0]
    return fronts


def crowding_distance(objs_front: np.ndarray) -> np.ndarray:
    """front 내 점들의 crowding distance (경계점 = inf)."""
    n = objs_front.shape[0]
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for m in range(objs_front.shape[1]):
        order = np.argsort(objs_front[:, m])
        vmin, vmax = objs_front[order[0], m], objs_front[order[-1], m]
        dist[order[0]] = dist[order[-1]] = np.inf
        if vmax > vmin:
            span = vmax - vmin
            dist[order[1:-1]] += (objs_front[order[2:], m]
                                  - objs_front[order[:-2], m]) / span
    return dist


def assign_rank_crowd(objs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """전체 population 에 rank(작을수록 좋음) 와 crowding distance 부여."""
    N = objs.shape[0]
    rank = np.empty(N, dtype=np.int64)
    crowd = np.empty(N, dtype=np.float64)
    for r, fr in enumerate(fast_nondominated_sort(objs)):
        rank[fr] = r
        crowd[fr] = crowding_distance(objs[fr])
    return rank, crowd


def tournament(rank, crowd, rng) -> int:
    i, j = rng.integers(0, len(rank), size=2)
    if rank[i] != rank[j]:
        return i if rank[i] < rank[j] else j
    return i if crowd[i] >= crowd[j] else j


# ── genetic operators ───────────────────────────────
def pox_crossover(p1: np.ndarray, p2: np.ndarray, J: int, rng) -> np.ndarray:
    """Precedence Operation Crossover (OS). job 분할로 반복수(=S) 보존."""
    set1 = rng.random(J) < 0.5
    child = np.empty_like(p1)
    keep = set1[p1]                          # set1 job 은 p1 위치 그대로 유지
    child[keep] = p1[keep]
    child[~keep] = p2[~set1[p2]]             # 나머지는 p2 순서로 채움
    return child


def ma_crossover(m1: np.ndarray, m2: np.ndarray, rng) -> np.ndarray:
    mask = rng.random(m1.shape) < 0.5
    return np.where(mask, m1, m2)


def mutate_os(os_ind: np.ndarray, rng, p: float) -> np.ndarray:
    if rng.random() < p:
        i, j = rng.integers(0, len(os_ind), size=2)
        os_ind = os_ind.copy()
        os_ind[i], os_ind[j] = os_ind[j], os_ind[i]
    return os_ind


def mutate_ma(ma_ind: np.ndarray, cnt_per_op: np.ndarray, rng, p_gene: float) -> np.ndarray:
    mut = rng.random(ma_ind.shape) < p_gene
    if not mut.any():
        return ma_ind
    rand_m = (rng.random(ma_ind.shape) * cnt_per_op).astype(np.int64)
    rand_m = np.minimum(rand_m, cnt_per_op - 1)
    return np.where(mut, rand_m, ma_ind)


def init_population(N, J, S, machine_cnt_list, rng):
    num_ops = J * S
    base_os = np.repeat(np.arange(J), S)                      # job 당 S 회
    OS = np.stack([rng.permutation(base_os) for _ in range(N)])
    cnt_per_op = np.array([machine_cnt_list[op % S] for op in range(num_ops)], dtype=np.int64)
    MA = (rng.random((N, num_ops)) * cnt_per_op).astype(np.int64)
    MA = np.minimum(MA, cnt_per_op - 1)
    return OS, MA, cnt_per_op


# =====================================================
# Main NSGA-II loop
# =====================================================
def run_nsga2(decoder: HFSPDecoder, J, S, machine_cnt_list, *,
              pop_size=100, n_gen=200, p_cross_os=1.0, p_cross_ma=1.0,
              p_mut_os=0.2, p_mut_ma_gene=0.05, seed=0,
              hv_anchors=(100.0, 500.0, 0.85, 0.50)):
    rng = np.random.default_rng(seed)
    m_best, m_ref, q_best, q_ref = hv_anchors

    OS, MA, cnt_per_op = init_population(pop_size, J, S, machine_cnt_list, rng)
    ms, gt, obj = decoder.evaluate(OS, MA)
    objs = np.stack([ms, -obj], axis=1)               # 선택은 obj_yield 기준 (minimize 둘 다)
    rank, crowd = assign_rank_crowd(objs)

    hv_hist = []
    t0 = t_prev = time.time()
    for gen in range(n_gen):
        # ── offspring 생성 ──
        cOS = np.empty_like(OS)
        cMA = np.empty_like(MA)
        for k in range(pop_size):
            pa = tournament(rank, crowd, rng)
            pb = tournament(rank, crowd, rng)
            os_c = (pox_crossover(OS[pa], OS[pb], J, rng)
                    if rng.random() < p_cross_os else OS[pa].copy())
            ma_c = (ma_crossover(MA[pa], MA[pb], rng)
                    if rng.random() < p_cross_ma else MA[pa].copy())
            cOS[k] = mutate_os(os_c, rng, p_mut_os)
            cMA[k] = mutate_ma(ma_c, cnt_per_op, rng, p_mut_ma_gene)

        c_ms, c_gt, c_obj = decoder.evaluate(cOS, cMA)
        c_objs = np.stack([c_ms, -c_obj], axis=1)

        # ── 부모+자식 결합 후 환경선택 (μ+λ) ──
        all_OS = np.concatenate([OS, cOS], axis=0)
        all_MA = np.concatenate([MA, cMA], axis=0)
        all_ms = np.concatenate([ms, c_ms])
        all_gt = np.concatenate([gt, c_gt])
        all_obj = np.concatenate([obj, c_obj])
        all_objs = np.concatenate([objs, c_objs], axis=0)

        fronts = fast_nondominated_sort(all_objs)
        sel = []
        for fr in fronts:
            if len(sel) + len(fr) <= pop_size:
                sel.extend(fr.tolist())
            else:
                cd = crowding_distance(all_objs[fr])
                take = pop_size - len(sel)
                sel.extend(fr[np.argsort(-cd)[:take]].tolist())
                break
        sel = np.asarray(sel, dtype=np.int64)

        OS, MA = all_OS[sel], all_MA[sel]
        ms, gt, obj, objs = all_ms[sel], all_gt[sel], all_obj[sel], all_objs[sel]
        rank, crowd = assign_rank_crowd(objs)

        hv, _ = hypervolume(ms, gt, m_ref, q_ref)        # HV 는 항상 ground-truth 기준
        hv_hist.append(hv)
        now = time.time()
        gen_dt = now - t_prev                             # 이번 gen 1회 소요 시간
        t_prev = now
        if gen == 0 or (gen + 1) % 10 == 0 or gen == n_gen - 1:
            nd = (rank == 0).sum()
            print(f"[gen {gen+1:4d}/{n_gen}] HV_raw={hv:10.4f}  "
                  f"|front0|={nd:3d}  ms[min={ms.min():.1f}]  GTyld[max={gt.max():.4f}]  "
                  f"({gen_dt:.2f}s/gen)")

    return OS, MA, ms, gt, obj, np.asarray(hv_hist)


# =====================================================
# Plot
# =====================================================
def plot_results(ms, yld, save_path, title, hv_norm,
                 xlim=None, ylim=None, xticks=None, yticks=None):
    """Pareto front 만 그린다 (population/ref/best 마커 없음). 축 범위·tick 은 인자로 지정."""
    front_idx = compute_pareto_front(ms, yld)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(ms[front_idx], yld[front_idx], '-o', color='crimson',
            markersize=5, linewidth=1.3, label=f'Pareto Front (n={len(front_idx)})')
    ax.set_xlabel('Average Makespan ↓')
    ax.set_ylabel('Yield ↑')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_title(f'{title}')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return front_idx


# =====================================================
# Entry
# =====================================================
def main():
    p = argparse.ArgumentParser(description="NSGA-II baseline for HFSP (test.py 비교군)")
    p.add_argument('--num_jobs',  type=int, default=25)
    p.add_argument('--machines',  type=str, default='5,3,7,3,5,7')
    p.add_argument('--p_idx',     type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--q_idx',     type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--paths_idx', type=int, default=1)
    p.add_argument('--yield_mode', type=str, default='raw', choices=['raw', 'percentile'],
                   help="HV/리포트용 ground-truth yield 계산 방식 (scorer 기반).")
    p.add_argument('--yield_objective', type=str, default='predicted',
                   choices=['ground_truth', 'predicted'],
                   help="NSGA-II 선택(domination)에 쓰는 yield. "
                        "'ground_truth' = scorer GT (oracle 상한), "
                        "'predicted' = QualityHelper.compute_yield (모델 학습 신호와 동일 → 공정 비교). "
                        "어느 쪽이든 최종 HV/plot 은 ground-truth 로 평가.")
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00',
                   help="초기 웨이퍼 품질 U[lo,hi]. test.py 와 동일하게 0.99,1.00.")
    p.add_argument('--seed',      type=int, default=0,
                   help="wafer_quality / proc_time 인스턴스용 seed (test.py 와 동일하게 0).")
    p.add_argument('--ga_seed',   type=int, default=0, help="NSGA-II RNG seed.")
    # NSGA-II hyperparams
    p.add_argument('--pop',       type=int, default=100)
    p.add_argument('--gen',       type=int, default=100)
    p.add_argument('--p_mut_os',  type=float, default=0.2)
    p.add_argument('--p_mut_ma',  type=float, default=0.05, help="MA per-gene mutation prob.")
    # 축 범위 + HV anchor (test.py 와 동일 규약: 'lo,hi,step', tick 은 hi 제외).
    #   HV anchor 를 축에서 유도 → m=[best=xlo, ref=xhi],  q=[ref=ylo, best=yhi].
    p.add_argument('--xlim', type=str, default='100,500,100',
                   help="x축 (makespan) 'lo,hi' 또는 'lo,hi,step'. HV anchor m=[best=lo, ref=hi].")
    p.add_argument('--ylim', type=str, default='0.50,0.85,0.05',
                   help="y축 (yield) 'lo,hi' 또는 'lo,hi,step'. HV anchor q=[ref=lo, best=hi].")
    args = p.parse_args()

    machine_cnt_list = [int(x) for x in args.machines.split(',')]
    J, S = args.num_jobs, len(machine_cnt_list)
    wq_lo, wq_hi = [float(x) for x in args.wafer_quality.split(',')]

    def _parse_axis(s):
        parts = [float(x) for x in s.split(',')]
        if len(parts) == 2:
            return (parts[0], parts[1]), None
        if len(parts) == 3:
            lo, hi, step = parts
            ticks = np.arange(lo, hi, step)
            if ticks.size == 0 or abs(ticks[-1] - hi) > 1e-9:
                ticks = np.append(ticks, hi)              # 끝점(hi=anchor) 도 tick 으로 표시
            return (lo, hi), ticks
        raise ValueError(f"axis must be 'lo,hi' or 'lo,hi,step', got: {s!r}")

    xlim, xticks = _parse_axis(args.xlim)
    ylim, yticks = _parse_axis(args.ylim)
    hv_m_best, hv_m_ref = xlim
    hv_q_ref,  hv_q_best = ylim
    if args.yield_mode == 'percentile':
        hv_q_best, hv_q_ref = 100.0, 0.0                  # percentile 0~100 강제 (test.py 규약)
        ylim, yticks = (0.0, 100.0), None
    hv_anchors = (hv_m_best, hv_m_ref, hv_q_best, hv_q_ref)
    # 이후 코드가 참조하는 args.hv_* 에 그대로 반영 (단일 출처 유지).
    args.hv_m_best, args.hv_m_ref, args.hv_q_best, args.hv_q_ref = hv_anchors

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    # ── env / 인스턴스 ──
    env = HFSPGraphEnv(num_jobs=J, machine_cnt_list=machine_cnt_list, device=device)
    problems_INT_list = load_problems_from__quality_file(
        f'quality_data/P_{args.p_idx}.csv', J, machine_cnt_list)
    env_edge_lookup_np = env.env_edge_lookup.cpu().numpy()
    machine_offset_np = np.cumsum([0] + machine_cnt_list[:-1]).astype(np.int64)

    # ── wafer_quality: test.py(QualityHelper.sample_wafer_quality) 와 동일 RNG ──
    wq_csv = f'quality_data/Q_{args.q_idx}/historical_paths_{args.paths_idx}.csv'
    model_path = f'quality_results/Q_{args.q_idx}/paths_{args.paths_idx}/best_hb_model.zip'
    wafer_path = f'quality_results/Q_{args.q_idx}/paths_{args.paths_idx}/wafer_quality.json'
    qh = QualityHelper(num_stages=S, device=device,
                       model_path=model_path, wafer_path=wafer_path)
    if not qh.is_active:
        raise RuntimeError(f"quality pipeline not available — {model_path}, {wafer_path}")
    qh.wafer_quality_min, qh.wafer_quality_max = wq_lo, wq_hi
    wq_1 = qh.sample_wafer_quality(B=1, num_jobs=J, seed=args.seed, device=device)
    wq_np = wq_1[0].cpu().numpy().astype(np.float64)         # (J,)

    scorer = PathPercentileScorer(wq_csv, machine_cnt_list)

    print(f"[nsga2] device={device}  J={J}  machines={machine_cnt_list}  NE={env.num_edges}  "
          f"yield_mode={args.yield_mode}  yield_objective={args.yield_objective}")
    print(f"[nsga2] instance: P_{args.p_idx}, Q_{args.q_idx}, paths_{args.paths_idx}  "
          f"wafer_q=U[{wq_lo:.4f},{wq_hi:.4f}] mean={wq_np.mean():.4f}")
    print(f"[nsga2] HV anchors: m=[best={args.hv_m_best}, ref={args.hv_m_ref}]  "
          f"q=[ref={args.hv_q_ref}, best={args.hv_q_best}]")
    print(f"[nsga2] pop={args.pop}  gen={args.gen}  p_mut_os={args.p_mut_os}  "
          f"p_mut_ma={args.p_mut_ma}  ga_seed={args.ga_seed}")

    decoder = HFSPDecoder(env, problems_INT_list, env_edge_lookup_np,
                          machine_offset_np, scorer, wq_np, yield_mode=args.yield_mode,
                          yield_objective=args.yield_objective, quality_helper=qh)

    OS, MA, ms, gt, obj, hv_hist = run_nsga2(
        decoder, J, S, machine_cnt_list,
        pop_size=args.pop, n_gen=args.gen, p_mut_os=args.p_mut_os,
        p_mut_ma_gene=args.p_mut_ma, seed=args.ga_seed, hv_anchors=hv_anchors)

    # ── 최종 HV / front : 항상 ground-truth yield(gt) 로 평가 (모델 평가와 동일 기준) ──
    yld = gt
    hv_raw, front_used = hypervolume(ms, yld, args.hv_m_ref, args.hv_q_ref)
    box = (args.hv_m_ref - args.hv_m_best) * (args.hv_q_best - args.hv_q_ref)
    hv_norm = hv_raw / box if box > 0 else float('nan')
    front_idx = compute_pareto_front(ms, yld)

    print("\n========== NSGA-II result ==========")
    print(f"yield_objective = {args.yield_objective}"
          + ("  (NSGA-II 가 예측 yield 로 선택 → HV 는 GT 로 평가)"
             if args.yield_objective == 'predicted' else "  (GT oracle)"))
    if args.yield_objective == 'predicted':
        print(f"objective(pred) yield: min={obj.min():.4f}  max={obj.max():.4f}  "
              f"(GT 와의 평균 괴리={np.abs(obj - gt).mean():.4f})")
    print(f"Pareto front size = {len(front_idx)}")
    print(f"makespan: min={ms.min():.2f}  max={ms.max():.2f}")
    print(f"yield(GT): min={yld.min():.4f}  max={yld.max():.4f}")
    print(f"HV_raw  = {hv_raw:.4f}   (ref=({args.hv_m_ref:.0f}, {args.hv_q_ref:.2f}))")
    print(f"HV_norm = {hv_norm:.4f}  (anchor box [{args.hv_m_best:.0f},{args.hv_m_ref:.0f}]"
          f"×[{args.hv_q_ref:.2f},{args.hv_q_best:.2f}])")

    # ── 저장 ──
    obj_tag = '' if args.yield_objective == 'ground_truth' else f'_{args.yield_objective}'
    tag = f'nsga2_J{J}_Q{args.q_idx}_P{args.p_idx}_p{args.paths_idx}'
    png_path = f'test_results/{tag}.png'
    # csv_path = f'test_results/{tag}_front.csv'
    title = (f'NSGA-II  W={J}, (Q{args.q_idx},P{args.p_idx}), paths_{args.paths_idx}')
    plot_results(ms, yld, png_path, title, hv_norm,
                 xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
    # pd.DataFrame({'makespan': ms[front_idx], 'yield': yld[front_idx]}).to_csv(
    #     csv_path, index=False)
    print(f"\nsaved -> {png_path}")
    # print(f"saved -> {csv_path}")


if __name__ == "__main__":
    main()
