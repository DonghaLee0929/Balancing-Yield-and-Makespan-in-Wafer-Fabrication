"""
experiment_table.py — 방법별 비교 표 (bi-objective HFSP: min makespan / max yield).

고정 인스턴스 (기본 Q3, P3, paths_1, W=25) 에 대해 아래 표의 5개 지표를 방법별로 계산한다.

    Method          | HV ↑ | IGD+ ↓ | Makespan ↓ | Quality ↑ | Time (s)
    EST             |   est_jm 우선 휴리스틱 (λ 무관, 단일 점)
    Quality greedy  |   quality z-score 가 가장 큰 (job,machine) 선택 (λ 무관, 단일 점)
    NSGA2           |   nsga2.py 의 NSGA-II (population → Pareto front)
    Ours (g)        |   학습된 λ-conditioned 모델, greedy decoding (λ-sweep)
    Ours (s)        |   학습된 λ-conditioned 모델, stochastic sampling (λ-sweep × samples)

모든 방법은 *완전히 동일한* 인스턴스(같은 proc_time, 같은 wafer_quality, 같은 ground-truth
yield scorer, 같은 HV anchor)로 평가 — test.py / nsga2.py 의 rollout/decoder/scorer 코드를
그대로 import 해서 쓰므로 표 수치가 그 스크립트들의 결과와 1:1 로 일치한다.

10-path 평균으로 확장: --paths_idx '1~10' (또는 '1,3,5') 로 주면 각 path 별로 위 표를 만든 뒤
지표를 path 축으로 평균(mean±std)한다. 기본은 paths_1 단일.

지표 정의:
  HV       : 비지배 점들이 ref(worst corner)까지 지배하는 면적을 anchor box 로 정규화 (0~1, ↑).
             test.py / nsga2.py 의 hypervolume() 동일 정의·동일 anchor.
  IGD+     : Inverted Generational Distance plus (↓, 작을수록 좋음). 5개 방법 점을 모두 합쳐
             만든 best-known front(=reference set Z) 의 각 점에서, 그 방법 front 까지의
             'dominated-side' 거리 d+ 의 평균. anchor 로 [0,1] 정규화한 (둘 다 최소화) 목적
             공간에서 계산. front 가 Z 의 영역을 못 덮을수록 커진다. crowding distance 와
             달리 점 개수에 편향되지 않아 cardinality 가 다른 front 를 공정 비교 가능
             (Pareto-compliant). 단일 점(EST/Quality greedy) 도 정의됨.
  Makespan : front 의 최소(=best) makespan (↓).
  Quality  : front 의 최대(=best) yield (↑).
  Time (s) : 그 방법의 탐색/추론 wall-clock (모델 로드 등 공통 셋업 제외).
"""

from __future__ import annotations
import os
import sys
import time
import argparse

# 로컬 test.py 가 stdlib `test` 패키지에 가려지지 않도록 스크립트 디렉터리를 최우선에.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 콘솔(cp949)에서 ↑/↓/—/λ 등 유니코드 출력 시 UnicodeEncodeError 방지.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import torch

from HFSPGraphEnv import HFSPGraphEnv
from FFSPModel import FFSPModel
from HFSPWrapper import make_env_edge_lookup, get_feat_dims
# 예측 모델 없음 — ground-truth quality 한 객체가 피처·리워드·채점을 모두 제공.
# 머신 수 증강(m%base 복제) 지원, proc_time 도 같은 규칙으로 복제 로드.
from quality_augment import GroundTruthQuality, load_proc_time_augmented

# test.py 의 평가 머신을 그대로 재사용 (모델/EST rollout, HV, pareto).
from test import (
    problems_to_edge_proc_time,
    schedule_paths_1indexed,
    compute_pareto_front,
    hypervolume,
    _run_single_experiment,
)
# nsga2.py 의 NSGA-II decoder/loop 재사용.
from nsga2 import HFSPDecoder, run_nsga2


METHODS = ['EST', 'Quality greedy', 'NSGA2', 'Ours (g)', 'Ours (s)']


# =====================================================
# Quality-greedy baseline — quality z-score 가 가장 큰 (job, machine) 선택
# =====================================================
def sample_quality_greedy_action(env: HFSPGraphEnv, quality_helper: GroundTruthQuality):
    """매 step feasible (job, machine) 중 quality z-score 최대 엣지 선택.

    GroundTruthQuality.compute_machine_quality 가 만드는 (B,J,M) z-score prior
    (stage-내, 클수록 고품질) 를 그대로 점수로 사용. 동점은 EST 로 미세 tie-break
    (-1e-4·est) 해 makespan 폭주를 방지. sample_est_action 과 대칭 구조.
    """
    B, J, M = env.batch_size, env.num_jobs, env.total_machines
    qz = quality_helper.compute_machine_quality(env)            # (B, J, M) — 클수록 좋음
    est_jm = env.edge_pool['est']                               # (B, J, M) — 동점 tie-break
    score = qz - 1e-4 * est_jm

    edge_idx_3d = env.env_edge_lookup[env.active_op]            # (B, J, M)
    valid = edge_idx_3d >= 0
    safe = torch.where(valid, edge_idx_3d, torch.zeros_like(edge_idx_3d))
    feas = env.feasible_mask.gather(1, safe.view(B, -1)).view(B, J, M) & valid

    neg_inf = torch.tensor(float('-inf'), device=env.device)
    masked = torch.where(feas, score, neg_inf)
    best_flat = masked.view(B, -1).argmax(dim=1)
    return safe.view(B, -1).gather(1, best_flat.unsqueeze(1)).squeeze(1)


def run_quality_greedy(env, quality_helper, scorer, problems_INT_list, wq_1,
                       seed, yield_mode, device):
    """quality-greedy 1회 rollout → (makespans[1], yields[1]). _run_single_experiment 의
    yield 계산(raw/percentile)을 그대로 복제해 다른 방법과 동일 ground-truth 로 채점."""
    ept = problems_to_edge_proc_time(problems_INT_list, env, 1)
    env.reset(seed=seed, batch_size=1, edge_proc_time=ept, identical_job=True)
    T = env.num_ops
    for t in range(T):
        a = sample_quality_greedy_action(env, quality_helper)
        env.step(a, last=(t == T - 1))

    ms = env.makespan().cpu().numpy().astype(np.float32)        # (1,)
    paths = schedule_paths_1indexed(env)                        # (1, J, S)
    if yield_mode == 'percentile':
        yld = scorer.score(paths).mean(axis=1).astype(np.float32)
    else:
        prod_bj = scorer.path_product(paths)                    # (1, J)
        wq_np = wq_1.cpu().numpy()                              # (1, J)
        yld = (prod_bj * wq_np).mean(axis=1).astype(np.float32)
    return ms, yld


# =====================================================
# Metrics
# =====================================================
def _to_min_normalized(ms, yld, anchors):
    """두 목적을 anchor 로 [0,1] 정규화하고 *둘 다 '작을수록 좋음'(minimization)* 으로 변환.
        makespan: (ms - m_best) / (m_ref - m_best)   — 0 이 best
        yield   : (q_best - yld) / (q_best - q_ref)   — 0 이 best (yield 는 클수록 좋으므로 반전)
    IGD+ 의 d+ 가 '둘 다 최소화' 좌표를 요구하므로 yield 를 반전한다. 반환: (n, 2), 둘 다 ↓.
    """
    m_best, m_ref, q_best, q_ref = anchors
    ms = np.asarray(ms, dtype=np.float64)
    yld = np.asarray(yld, dtype=np.float64)
    m_n = (ms - m_best) / (m_ref - m_best)
    q_n = (q_best - yld) / (q_best - q_ref)
    return np.stack([m_n, q_n], axis=1)                         # (n, 2), 둘 다 ↓


def build_reference_front(per_method_pts, anchors):
    """모든 방법의 점을 합쳐 비지배 집합(best-known front) 을 만들고, 정규화·최소화 좌표로 반환.

    IGD+ 의 기준집합 Z. 이렇게 만든 union front 는 어떤 단일 방법보다도 우월(또는 동등)하므로,
    각 방법은 'Z 를 얼마나 덮느냐' 로 평가된다 → 점 개수 편향 없이 cardinality 가 다른 front 들을
    공정 비교. 반환: (k, 2) 배열(둘 다 ↓) 또는 점이 없으면 None.
    """
    all_ms, all_yld = [], []
    for ms, yld in per_method_pts.values():
        all_ms.append(np.asarray(ms, dtype=np.float64).ravel())
        all_yld.append(np.asarray(yld, dtype=np.float64).ravel())
    ms = np.concatenate(all_ms)
    yld = np.concatenate(all_yld)
    front = compute_pareto_front(ms, yld)
    if front.size == 0:
        return None
    return _to_min_normalized(ms[front], yld[front], anchors)


def igd_plus_metric(ms, yld, ref_front_min, anchors):
    """IGD+ (↓) — 기준 front Z(ref_front_min) 의 각 점 z 에서 이 방법 점집합 A 까지의
    'dominated-side' 거리 d+ 의 평균.

        d+(z, a) = sqrt( max(a_m - z_m, 0)^2 + max(a_q - z_q, 0)^2 )    (둘 다 최소화 좌표)
        IGD+(A)  = mean_{z in Z} min_{a in A} d+(z, a)

    a 가 z 를 (약)지배하면 그 축 기여 0 → A 가 Z 를 잘 덮을수록 0 에 가깝다. 점 개수에
    편향되지 않아 greedy(성김) vs sampling(빽빽) 을 공정 비교(=CD 의 대체). Z 가 비면 nan.
    """
    if ref_front_min is None or ref_front_min.shape[0] == 0:
        return float('nan')
    A = _to_min_normalized(ms, yld, anchors)                    # (|A|, 2), 둘 다 ↓
    if A.shape[0] == 0:
        return float('nan')
    diff = A[None, :, :] - ref_front_min[:, None, :]            # (|Z|, |A|, 2): a - z
    dplus = np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2, axis=2))  # (|Z|, |A|): d+ (a 가 worse 인 축만)
    return float(dplus.min(axis=1).mean())                      # 각 z 의 최근접 a → 평균


def compute_metrics(ms, yld, anchors):
    """(ms, yld) 점집합 → {HV, Makespan, Quality}. HV 는 anchors 기준.

    IGD+ 는 5개 방법 공통 reference front 가 필요해 여기서 계산하지 않고 path 단위에서
    build_reference_front + igd_plus_metric 으로 따로 계산한다.
    """
    m_best, m_ref, q_best, q_ref = anchors
    ms64 = np.asarray(ms, dtype=np.float64)
    yld64 = np.asarray(yld, dtype=np.float64)
    hv_raw, _ = hypervolume(ms64, yld64, m_ref, q_ref)
    box = (m_ref - m_best) * (q_best - q_ref)
    hv = hv_raw / box if box > 0 else float('nan')
    return {
        'HV': hv,
        'Makespan': float(ms64.min()),
        'Quality': float(yld64.max()),
    }


def timed(fn, device):
    """fn 을 실행하며 wall-clock 측정 (cuda 면 sync 포함). 반환: (결과, 초)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return out, time.time() - t0


# =====================================================
# Per-path evaluation — 한 인스턴스(path)에 대해 5개 방법 모두 실행
# =====================================================
def evaluate_one_path(*, env, model, env_edge_lookup_t, device,
                      problems_INT_list, machines, J, S,
                      q_idx, paths_idx, yield_mode, anchors, seed, ga_seed,
                      wq_min, wq_max, num_lambdas, samples,
                      nsga_pop, nsga_gen):
    """단일 path 인스턴스에 대해 method -> (ms_array, yld_array, time_s) dict 반환."""
    m_best, m_ref, q_best, q_ref = anchors

    wq_csv = f'quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv'

    # 예측 모델 없음 — ground-truth quality 한 객체가 피처(quality_helper)·채점(scorer) 겸용.
    # machines 가 CSV base(5,3,7,3,5,7) 초과면 (m%base) 복제 증강.
    gq = GroundTruthQuality(num_stages=S, device=device, csv_path=wq_csv,
                            machine_cnt_list=machines,
                            wafer_quality_min=wq_min, wafer_quality_max=wq_max)

    # 모든 방법이 동일 wafer_quality 를 보도록 path 마다 한 번만 샘플.
    wq_1 = gq.sample_wafer_quality(B=1, num_jobs=J, seed=seed, device=device)   # (1, J)
    qh = scorer = gq                                                            # 동일 ground-truth 객체

    lam_g = torch.linspace(0.0, 1.0, num_lambdas, device=device)               # (N,)
    lam_s = lam_g.repeat_interleave(samples)                                   # (N·S,)

    outputs = {}

    # ── EST (단일 점, λ 무관) ──
    (ms, yld), t = timed(lambda: _run_single_experiment(
        None, env, env_edge_lookup_t, device, qh, scorer,
        problems_INT_list, wq_1, torch.zeros(1, device=device),
        1, 1, seed, True, method='est', yield_mode=yield_mode), device)
    outputs['EST'] = (ms, yld, t)

    # ── Quality greedy (단일 점, λ 무관) ──
    (ms, yld), t = timed(lambda: run_quality_greedy(
        env, qh, scorer, problems_INT_list, wq_1, seed, yield_mode, device), device)
    outputs['Quality greedy'] = (ms, yld, t)

    # ── NSGA-II (population → front) ──
    env_edge_lookup_np = env.env_edge_lookup.cpu().numpy()
    machine_offset_np = np.cumsum([0] + machines[:-1]).astype(np.int64)
    wq_np_j = wq_1[0].cpu().numpy().astype(np.float64)                         # (J,)

    def _run_nsga():
        decoder = HFSPDecoder(env, problems_INT_list, env_edge_lookup_np,
                              machine_offset_np, scorer, wq_np_j,
                              yield_mode=yield_mode, yield_objective='ground_truth',
                              quality_helper=None)   # 예측 모델 없음 → GT oracle 만
        _, _, ms_n, gt_n, _, _ = run_nsga2(
            decoder, J, S, machines,
            pop_size=nsga_pop, n_gen=nsga_gen, seed=ga_seed,
            hv_anchors=(m_best, m_ref, q_best, q_ref))
        return ms_n.astype(np.float32), gt_n.astype(np.float32)

    (ms, yld), t = timed(_run_nsga, device)
    outputs['NSGA2'] = (ms, yld, t)

    # ── Ours (greedy) — λ-sweep, samples=1 ──
    (ms, yld), t = timed(lambda: _run_single_experiment(
        model, env, env_edge_lookup_t, device, qh, scorer,
        problems_INT_list, wq_1, lam_g, num_lambdas, 1, seed, True,
        method='model', yield_mode=yield_mode), device)
    outputs['Ours (g)'] = (ms, yld, t)

    # ── Ours (sampling) — λ-sweep × samples, stochastic ──
    def _run_sample():
        torch.manual_seed(seed)                # 샘플링 재현성 (방법 실행 순서 무관)
        return _run_single_experiment(
            model, env, env_edge_lookup_t, device, qh, scorer,
            problems_INT_list, wq_1, lam_s, num_lambdas * samples, 1, seed, False,
            method='model', yield_mode=yield_mode)

    (ms, yld), t = timed(_run_sample, device)
    outputs['Ours (s)'] = (ms, yld, t)

    return outputs, anchors


# =====================================================
# Table rendering
# =====================================================
def _fmt(vals, nd):
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return '—'
    mean = np.nanmean(arr)
    if arr.size > 1:
        return f"{mean:.{nd}f}±{np.nanstd(arr):.{nd}f}"
    return f"{mean:.{nd}f}"


def build_table(agg, header, n_paths):
    """agg[method][metric] = list(per-path values) → 출력 문자열 + (가능시) DataFrame."""
    cols = ['Method', 'HV ↑', 'IGD+ ↓', 'Makespan ↓', 'Quality ↑', 'Time (s)']
    decimals = {'HV ↑': 4, 'IGD+ ↓': 4, 'Makespan ↓': 1, 'Quality ↑': 4, 'Time (s)': 2}
    key_map = {'HV ↑': 'HV', 'IGD+ ↓': 'IGD+', 'Makespan ↓': 'Makespan',
               'Quality ↑': 'Quality', 'Time (s)': 'Time'}

    rows = []
    for m in METHODS:
        row = {'Method': m}
        for c in cols[1:]:
            row[c] = _fmt(agg[m][key_map[c]], decimals[c])
        rows.append(row)

    # 콘솔 출력 (pandas 있으면 사용, 없으면 수동 정렬)
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=cols).set_index('Method')
        body = df.to_string()
    except Exception:
        df = None
        widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
        line = '  '.join(c.ljust(widths[c]) for c in cols)
        sep = '  '.join('-' * widths[c] for c in cols)
        body = '\n'.join([line, sep] + [
            '  '.join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows])

    n_tag = f"(avg over {n_paths} paths, mean±std)" if n_paths > 1 else "(single path)"
    out = f"\n{header}\n{n_tag}\n{body}\n"
    return out, rows


# =====================================================
# Optional overlay plot (단일 path 일 때만)
# =====================================================
def plot_fronts(per_method_pts, anchors, save_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m_best, m_ref, q_best, q_ref = anchors

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = {'EST': 'tab:gray', 'Quality greedy': 'tab:green', 'NSGA2': 'tab:orange',
              'Ours (g)': 'tab:blue', 'Ours (s)': 'tab:red'}
    for name in METHODS:
        ms, yld = per_method_pts[name]
        ms = np.asarray(ms, np.float64); yld = np.asarray(yld, np.float64)
        front = compute_pareto_front(ms, yld)
        ax.scatter(ms, yld, s=12, alpha=0.25, color=colors[name])
        if front.size >= 2:
            ax.plot(ms[front], yld[front], '-o', ms=4, lw=1.2,
                    color=colors[name], label=name)
        else:
            ax.scatter(ms[front], yld[front], s=90, marker='*',
                       color=colors[name], edgecolor='black', zorder=5, label=name)
    ax.set_xlabel('Makespan ↓'); ax.set_ylabel('Yield ↑')
    ax.set_xlim(m_best, m_ref)
    if q_best > 1.5:   # percentile 모드
        ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =====================================================
# Entry
# =====================================================
def _parse_idx_spec(s: str) -> list[int]:
    s = s.strip()
    if '~' in s:
        lo, hi = s.split('~')
        return list(range(int(lo), int(hi) + 1))
    if ',' in s:
        return [int(x) for x in s.split(',')]
    return [int(s)]

# 5,3,7,3,5,7
# 10,6,14,6,10,14
# 20,12,28,12,20,28
def main():
    p = argparse.ArgumentParser(description="HFSP 방법 비교 표 (EST/Quality greedy/NSGA2/Ours).")
    p.add_argument('--ckpt', type=str, default='checkpoints/saved_new_new_baseline.pt')
    p.add_argument('--num_jobs', type=int, default=25)
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7',
                   help="stage별 머신 수. stage 수는 CSV의 6 고정. base(5,3,7,3,5,7) 초과 시 "
                        "(m%%base) 복제 증강(2배=동일 인자 2벌), 미만 시 앞쪽 subset. 예: '10,6,14,6,10,14'.")
    p.add_argument('--p_idx', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--q_idx', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--paths_idx', type=str, default='1',
                   help="historical_paths 인덱스. '1' / '1~10' / '1,3,5'. 여러 개면 path 평균.")
    p.add_argument('--yield_mode', type=str, default='raw', choices=['raw', 'percentile'])
    p.add_argument('--num_lambdas', type=int, default=32, help="Ours 의 λ grid 크기.")
    p.add_argument('--samples', type=int, default=8, help="Ours (s) 의 λ 당 sample 수.")
    p.add_argument('--nsga_pop', type=int, default=100)
    p.add_argument('--nsga_gen', type=int, default=100)
    p.add_argument('--xlim', type=str, default='100,600',
                   help="makespan anchor 'm_best,m_ref'.")
    p.add_argument('--ylim', type=str, default='0,1',
                   help="yield anchor 'q_ref,q_best'. percentile 모드는 0,100 으로 강제.")
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ga_seed', type=int, default=0)
    p.add_argument('--out', type=str, default='test_results/comparison_table')
    p.add_argument('--no_plot', action='store_false')
    args = p.parse_args()

    machines = [int(x) for x in args.machines.split(',')]
    J, S = args.num_jobs, len(machines)
    paths_idx_list = _parse_idx_spec(args.paths_idx)
    wq_min, wq_max = [float(x) for x in args.wafer_quality.split(',')]

    m_best, m_ref = [float(x) for x in args.xlim.split(',')]
    q_ref, q_best = [float(x) for x in args.ylim.split(',')]
    if args.yield_mode == 'percentile':
        q_ref, q_best = 0.0, 100.0
    anchors = (m_best, m_ref, q_best, q_ref)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f"[table] device={device}  W={J}  machines={machines}  "
          f"(Q{args.q_idx},P{args.p_idx})  paths={paths_idx_list}  "
          f"yield_mode={args.yield_mode}")
    print(f"[table] anchors: m=[best={m_best}, ref={m_ref}]  q=[ref={q_ref}, best={q_best}]")
    print(f"[table] Ours: λ×{args.num_lambdas}, samples/λ={args.samples}  |  "
          f"NSGA-II: pop={args.nsga_pop}, gen={args.nsga_gen}  (yield=ground-truth only)")

    # ── env / proc_time / 모델 (path 무관, 한 번만) ──
    # proc_time 도 machines 가 base 초과면 (m%base) 복제 증강해서 로드.
    env = HFSPGraphEnv(num_jobs=J, machine_cnt_list=machines, device=device)
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)
    problems_INT_list = load_proc_time_augmented(
        f'quality_data/P_{args.p_idx}.csv', J, machines)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_params = ckpt['model_params']
    edge_feat_dim = (ckpt.get('edge_feat_dim')
                     or model_params.pop('edge_feature_dim', None)
                     or get_feat_dims(env)[2])
    model = FFSPModel(ckpt['row_feat_dim'], ckpt['col_feat_dim'],
                      edge_feat_dim, **model_params).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # ── path 별 평가 ──
    agg = {m: {'HV': [], 'IGD+': [], 'Makespan': [], 'Quality': [], 'Time': []}
           for m in METHODS}
    last_pts = None
    for paths_idx in paths_idx_list:
        print(f"\n========== paths_{paths_idx} ==========")
        outputs, _ = evaluate_one_path(
            env=env, model=model, env_edge_lookup_t=env_edge_lookup_t, device=device,
            problems_INT_list=problems_INT_list, machines=machines, J=J, S=S,
            q_idx=args.q_idx, paths_idx=paths_idx, yield_mode=args.yield_mode,
            anchors=anchors, seed=args.seed, ga_seed=args.ga_seed,
            wq_min=wq_min, wq_max=wq_max,
            num_lambdas=args.num_lambdas, samples=args.samples,
            nsga_pop=args.nsga_pop, nsga_gen=args.nsga_gen)

        # 1st pass: 방법별 독립 지표(HV/Makespan/Quality) + 점집합 수집.
        last_pts = {}
        md_all = {}
        for name in METHODS:
            ms, yld, t = outputs[name]
            md_all[name] = (compute_metrics(ms, yld, anchors), t)
            last_pts[name] = (ms, yld)

        # 2nd pass: 5개 방법 점을 합쳐 best-known front → 방법별 IGD+ (공통 기준 필요).
        ref_front = build_reference_front(last_pts, anchors)
        for name in METHODS:
            md, t = md_all[name]
            ms, yld = last_pts[name]
            igd = igd_plus_metric(ms, yld, ref_front, anchors)
            agg[name]['HV'].append(md['HV'])
            agg[name]['IGD+'].append(igd)
            agg[name]['Makespan'].append(md['Makespan'])
            agg[name]['Quality'].append(md['Quality'])
            agg[name]['Time'].append(t)
            igd_str = '—' if np.isnan(igd) else f"{igd:.4f}"
            print(f"  {name:14s}  HV={md['HV']:.4f}  IGD+={igd_str}  "
                  f"ms={md['Makespan']:.1f}  q={md['Quality']:.4f}  t={t:.2f}s  "
                  f"|pts|={np.asarray(ms).size}")

    # ── 표 출력 + 저장 ──
    header = (f"N={J}, M={machines}, (Q{args.q_idx},P{args.p_idx}), "
              f"yield={args.yield_mode}")
    table_str, rows = build_table(agg, header, len(paths_idx_list))
    print(table_str)

    args.out = f'test_results/J{J}_M{sum(machines)}_table'
    csv_path = f"{args.out}.csv"
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"saved -> {csv_path}")
    except Exception as e:
        print(f"[warn] CSV 저장 실패: {e}")

    if not args.no_plot and len(paths_idx_list) == 1 and last_pts is not None:
        png_path = f"{args.out}.png"
        plot_fronts(last_pts, anchors, png_path,
                    title=f"{header}  paths_{paths_idx_list[0]}")
        print(f"saved -> {png_path}")


if __name__ == "__main__":
    main()
