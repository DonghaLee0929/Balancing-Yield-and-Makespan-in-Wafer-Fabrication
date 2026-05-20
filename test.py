"""
HFSP Pareto Front evaluation — 학습된 λ-conditioned 모델을 고정 인스턴스 1개에
대해 λ ∈ linspace(0, 1, N) 로 sweep 하며 (makespan, yield) 점을 모아 Pareto
front 를 시각화.

- proc_time     : quality_data/P_{p_idx}.csv  (load_problems_from__quality_file)
- wafer_quality : quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv  앞 num_jobs 행 (랜덤 X)
- quality model : quality_results/Q_{q_idx}/paths_{paths_idx}/best_hb_model.zip  (사전 학습, Hummingbird PyTorch)
- samples_per_lambda == 1 → greedy decoding (PMOCO 류 표준 패턴)
- samples_per_lambda  > 1 → 각 λ 에서 stochastic sampling

Pareto: makespan ↓ , yield ↑ 의 비지배 점들.
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from HFSPGraphEnv import HFSPGraphEnv, sample_est_action
from FFSProblemDef import load_problems_from__quality_file
from FFSPModel import FFSPModel
from HFSPWrapper import (
    QualityHelper, make_env_edge_lookup, get_feat_dims, _rollout_loop,
)


# =====================================================
# helpers
# =====================================================
def problems_to_edge_proc_time(problems_INT_list, env: HFSPGraphEnv,
                               batch_size: int) -> torch.Tensor:
    """list of (1, J, cnt_s) tensors → (B, num_edges) long on env.device. 단일 인스턴스를 B 회 tile.

    edge 순서 규칙은 HFSPGraphEnv.sample_edge_proc_time 과 동일:
        idx = j * total_machines + machine_offset[s] + k_local.
    """
    device = env.device
    machine_offset = torch.tensor(
        np.cumsum([0] + env.machine_cnt_list[:-1]), dtype=torch.long, device=device)
    ept_1 = torch.empty((1, env.num_edges), dtype=torch.long, device=device)
    j_idx = torch.arange(env.num_jobs, device=device)
    for s in range(env.num_stages):
        cnt_s = env.machine_cnt_list[s]
        stage_pt = problems_INT_list[s].to(device).long()             # (1, J, cnt_s)
        edge_idx = (j_idx.unsqueeze(1) * env.total_machines
                    + machine_offset[s] + torch.arange(cnt_s, device=device).unsqueeze(0))
        ept_1[:, edge_idx.reshape(-1)] = stage_pt.reshape(1, -1)
    return ept_1.repeat(batch_size, 1)


def compute_pareto_front(makespan: np.ndarray, yld: np.ndarray) -> np.ndarray:
    """min makespan + max yield 의 비지배 점 indices. makespan 오름차순으로 정렬됨.

    같은 makespan 끼리는 yield 가 가장 큰 점만 비지배 — 따라서 makespan ↑,
    yield ↓ (즉 -yield ↑) 로 정렬한 뒤 strict yield 증가만 채택.
    """
    order = np.lexsort((-yld, makespan))   # primary: makespan asc, tie-break: yield desc
    front, best_y = [], -np.inf
    for i in order:
        if yld[i] > best_y:
            front.append(i)
            best_y = yld[i]
    return np.asarray(front, dtype=np.int64)


# =====================================================
# Path-percentile scorer
# =====================================================
class PathPercentileScorer:
    """historical_paths CSV 의 (stage, machine) quality factor 로 path 의 percentile 계산.

    ⚠ 평가(test) 전용 ground-truth. CSV 의 실제 quality factor 를 직접 읽으므로
       이 클래스/메서드를 train_ASIL.py 등 학습 코드에서 절대 import·사용하지 말 것.
       학습은 QualityHelper.compute_yield (예측 모델) 만 reward 로 써야 한다.

    1) CSV 의 "Step s quality" 컬럼을 보고 (stage s, machine m) 쌍별 quality factor 를 추출.
       각 (stage, machine) 의 값이 CSV 내에서 일정해야 함 (deterministic per-(stage,machine)).
    2) machine_cnt_list 로부터 가능한 모든 path 의 product quality 분포를 사전 계산.
    3) path_product(paths) → path 따라 quality factor 곱 (raw ground-truth yield 의 핵심).
    4) score(paths)        → 각 path 의 product 가 전체 분포에서 차지하는 백분위 (0–100).
    """

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
                            indexing='ij')                          # list of (cnt_0,…,cnt_{S-1})
        all_paths = np.stack(grids, axis=-1).reshape(-1, S)         # (P_all, S)
        self.all_products_sorted = np.sort(all_paths.prod(axis=1))  # (P_all,)
        self.n_paths = int(self.all_products_sorted.size)
        print(f"[percentile] {csv_path}: stages={S}  "
              f"machine_cnt={machine_cnt_list}  total paths={self.n_paths}  "
              f"product range=[{self.all_products_sorted[0]:.4f}, "
              f"{self.all_products_sorted[-1]:.4f}]")

    def path_product(self, paths_1indexed: np.ndarray) -> np.ndarray:
        """paths: (..., S) 1-indexed → (stage,machine) quality factor 의 path 곱 (...,).

        초기 wafer_quality 를 곱하기 전의 path 품질 — raw ground-truth yield 의 핵심.
        """
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
        """paths: (..., S) 1-indexed machine indices → percentile (..., ) in [0, 100]."""
        prod = self.path_product(paths_1indexed)                    # (...,)
        rank = np.searchsorted(self.all_products_sorted, prod.reshape(-1), side='right')
        pct = rank.astype(np.float64) / self.n_paths * 100.0
        return pct.reshape(prod.shape)


def schedule_paths_1indexed(env: HFSPGraphEnv) -> np.ndarray:
    """env 의 현재 schedule 에서 (B, J, S) 1-indexed local machine path 추출."""
    BP = env.batch_size
    J = env.num_jobs
    S = env.num_stages
    op_m_3d = env.op_machine.reshape(BP, J, S)
    return (env.machine_local[op_m_3d] + 1).cpu().numpy().astype(np.int64)


# =====================================================
# Evaluation
# =====================================================
def _rollout_loop_est(env: HFSPGraphEnv, obs, device) -> torch.Tensor:
    """ESTv2 휴리스틱 rollout — λ/quality 무관, 결정론적. (BP,) makespan reward G 반환.

    모델 대신 sample_est_v2_action (EST 우선, tie-break 처리시간) 으로 매 step action 선택.
    λ 를 보지 않으므로 BP 안의 모든 trajectory 가 (동일 proc_time 이면) 같은 결과 → Pareto 1점.
    """
    BP = env.batch_size
    total_reward = torch.zeros(BP, dtype=torch.float32, device=device)
    T = env.num_ops
    for t in range(T):
        actions = sample_est_action(env, obs)
        obs, reward, _ = env.step(actions, last=(t == T - 1))
        total_reward += reward
    return total_reward


def _run_single_experiment(
    model, env, env_edge_lookup_t, device,
    quality_helper, scorer,
    problems_INT_list, wq_1,
    lam_all_t, total, B, seed, greedy,
    method: str = 'model',
    yield_mode: str = 'raw',
) -> tuple[np.ndarray, np.ndarray]:
    """단일 (Q, paths) 인스턴스에 대해 λ-sweep 1회 → (makespans[total], yields[total]).

    method='model' : 학습된 λ-conditioned 모델 rollout.
    method='est'    : EST 휴리스틱 rollout (λ 무관, 모델 불필요). yield 계산은 동일.

    yield 점수는 항상 ground-truth (scorer 기반) — 예측 모델(compute_yield) 미사용.
    """
    BP = B * total
    ept_bp = problems_to_edge_proc_time(problems_INT_list, env, BP)         # (BP, NE) torch
    wq_bp  = wq_1.to(device).float().repeat(total, 1) if wq_1 is not None else None  # (BP, J)
    obs = env.reset(seed=seed, batch_size=BP,
                    edge_proc_time=ept_bp, identical_job=True)
    with torch.no_grad():
        if method == 'est':
            G = _rollout_loop_est(env, obs, device)
        else:
            _, G = _rollout_loop(env, model, env_edge_lookup_t, device,
                                 B=B, P=total, greedy=greedy, with_grad=False,
                                 wafer_quality_t=wq_bp,
                                 lambdas_t=lam_all_t,
                                 quality_helper=quality_helper)
    makespans = (-G).cpu().numpy().astype(np.float32)                       # (BP,)
    # yield 점수는 항상 ground-truth — historical_paths CSV 의 (stage,machine)
    # quality factor 를 path 따라 곱한 값 기반. (예측 모델 compute_yield 미사용)
    paths_3d = schedule_paths_1indexed(env)                                 # (BP, J, S) np
    if yield_mode == 'percentile':
        # 초기 품질은 모든 path 에 균일하게 곱해지는 상수 → percentile rank 불변.
        yields = scorer.score(paths_3d).mean(axis=1).astype(np.float32)     # (BP,)
    else:  # 'raw' : path 따라 quality factor 곱 × 초기 wafer_quality, job 평균
        prod_bj = scorer.path_product(paths_3d)                             # (BP, J)
        wq_np   = wq_bp.cpu().numpy()                                       # (BP, J) 초기 품질
        yields  = (prod_bj * wq_np).mean(axis=1).astype(np.float32)         # (BP,)
    return makespans, yields


def plot_pareto(
    makespans: np.ndarray,
    yields: np.ndarray,
    lam_all_np: np.ndarray,
    save_path: str,
    title: str,
    yield_mode: str = 'raw',
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: np.ndarray | None = None,
    yticks: np.ndarray | None = None,
    scatter_label: str | None = None,
) -> np.ndarray:
    """(makespan, yield) 산점 + Pareto front 를 저장. front index 배열 반환."""
    front_idx = compute_pareto_front(makespans, yields)
    y_axis_label = ('Yield ↑' if yield_mode == 'raw'
                    else 'Average Yield Precentile ↑')

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    sc = ax.scatter(makespans, yields,
                    c=lam_all_np, cmap='viridis',
                    s=18, alpha=0.85, edgecolors='none',
                    label=scatter_label or f'samples (n={len(makespans)})')
    plt.colorbar(sc, ax=ax, label='λ (0 = makespan-only,  1 = yield-only)')
    ax.plot(makespans[front_idx], yields[front_idx],
            '-o', color='crimson', markersize=5, linewidth=1.3,
            label=f'Pareto Front (n={len(front_idx)})')
    ax.set_xlabel('Average Makespan ↓')
    ax.set_ylabel(y_axis_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return front_idx


def evaluate_pareto(
    ckpt_path='checkpoints/saved_ASIL_0512.pt',
    p_idx: int = 1,
    q_idx: int = 1,
    paths_idx_list=(1,),
    num_jobs=15,
    machine_cnt_list=(5, 3, 7, 3, 5, 7),
    num_lambdas=51,
    samples=1,
    seed=0,
    wq_min: float = 0.99,
    wq_max: float = 1.00,
    save_path='pareto_front.png',
    method: str = 'model',
    yield_mode: str = 'raw',
    single_view: str = 'best',
    hv_m_best: float = 30.0,
    hv_m_ref: float = 110.0,
    hv_q_best: float = 0.97,
    hv_q_ref: float = 0.90,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: np.ndarray | None = None,
    yticks: np.ndarray | None = None,
):
    """λ-conditioned 모델의 Pareto front 평가 — PMOCO 류 표준 패턴.

    각 paths_idx 별 1회 실험 → (num_lambdas, samples_per_lambda) (ms, yld) 매트릭스.
    λ 별 stochastic sample 중 λ-가중 정규화 점수가 가장 높은 1점 선택
      ms_score  = (hv_m_ref − ms)  / (hv_m_ref − hv_m_best)        # ↓ 목적
      yld_score = (yld − hv_q_ref) / (hv_q_best − hv_q_ref)         # ↑ 목적
      score     = (1−λ)·ms_score + λ·yld_score   (argmax over S)
    → (N_runs, num_lambdas) 의 (ms, yld) 를 paths_idx 축으로 평균
    → 최종 num_lambdas 점들로 Pareto front.

    yield_mode (둘 다 ground-truth — historical_paths CSV 의 quality factor 기반,
                예측 모델 미사용. ⚠ 평가 전용, 학습엔 절대 사용 금지):
        'raw'        — (stage,machine) quality factor 의 path 곱 × 초기 wafer_quality, job 평균
        'percentile' — 위 path 곱의 전체 분포 대비 백분위 (job 평균; 초기 품질은 rank 불변)
    """
    p_csv = f'quality_data/P_{p_idx}.csv'
    if method not in ('model', 'est'):
        raise ValueError(f"method must be 'model' or 'est', got {method!r}")
    if yield_mode not in ('raw', 'percentile'):
        raise ValueError(f"yield_mode must be 'raw' or 'percentile', got {yield_mode!r}")
    if single_view not in ('best', 'all'):
        raise ValueError(f"single_view must be 'best' or 'all', got {single_view!r}")
    if yield_mode == 'percentile':
        # percentile 보상은 0~100 스케일 → 사용자 입력 무시하고 강제 고정.
        hv_q_best, hv_q_ref = 100.0, 0.0
    paths_idx_list = list(paths_idx_list)
    if not paths_idx_list:
        raise ValueError("paths_idx_list must be non-empty")
    num_lambdas = int(num_lambdas)
    samples = int(samples)
    if num_lambdas < 2:
        raise ValueError(f"num_lambdas must be >= 2, got {num_lambdas}")
    if samples < 1:
        raise ValueError(f"samples_per_lambda must be >= 1, got {samples}")
    wq_min = float(wq_min)
    wq_max = float(wq_max)
    if not (0.0 <= wq_min <= wq_max):
        raise ValueError(
            f"require 0 <= wq_min <= wq_max, got wq_min={wq_min}, wq_max={wq_max}")
    total = num_lambdas * samples
    greedy = (samples == 1)
    N_runs = len(paths_idx_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    # ── env (method 무관) ──
    env = HFSPGraphEnv(num_jobs=num_jobs, machine_cnt_list=list(machine_cnt_list), device=device)
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)

    # ── checkpoint / model (Q-independent, 한 번만 로드) — EST baseline 은 모델 불필요 ──
    model = None
    if method == 'model':
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        row_feat_dim = ckpt['row_feat_dim']
        col_feat_dim = ckpt['col_feat_dim']
        model_params = ckpt['model_params']
        edge_feat_dim = (ckpt.get('edge_feat_dim')
                         or model_params.pop('edge_feature_dim', None)
                         or get_feat_dims(env)[2])
        model = FFSPModel(row_feat_dim, col_feat_dim, edge_feat_dim, **model_params).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()

    # ── 고정 proc time ──
    problems_INT_list = load_problems_from__quality_file(
        p_csv, num_jobs, list(machine_cnt_list))

    # ── λ grid ──
    B = 1
    lam_grid_t = torch.linspace(0.0, 1.0, num_lambdas, device=device)        # (N,)
    lam_all_t  = lam_grid_t.repeat_interleave(samples)            # (total,)
    lam_grid_np = lam_grid_t.cpu().numpy()                                   # (N,) — plot 좌표

    print(f"[eval] method={method}  "
          f"ckpt={ckpt_path if method == 'model' else '(none, EST)'}  "
          f"device={device}  yield_mode={yield_mode}")
    if method == 'est' and (num_lambdas > 1 or samples > 1):
        print(f"[eval] NOTE: EST 는 lambda/quality 무관 결정론적 휴리스틱 - "
              f"lambda-sweep/samples 가 모두 동일 점으로 수렴 (Pareto front = 1점)")
    print(f"[eval] jobs={num_jobs}  machines={list(machine_cnt_list)}  "
          f"NE={env.num_edges}  num_lambdas={num_lambdas}  "
          f"samples/λ={samples}  total={total}  greedy={greedy}")
    print(f"[eval] proc_time <- {p_csv}")
    print(f"[eval] wafer_quality ~ U[{wq_min:.4f}, {wq_max:.4f}]"
          + ("  (raw yield 에 직접 반영)" if yield_mode == 'raw'
             else "  (percentile: yield 값엔 무관, 모델 입력에만 반영)"))
    print(f"[eval] Q_{q_idx}  paths_idx_list={paths_idx_list}  N_runs={N_runs}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    runs_ms  = np.zeros((N_runs, total), dtype=np.float32)
    runs_yld = np.zeros((N_runs, total), dtype=np.float32)

    for r, paths_idx in enumerate(paths_idx_list):
        wq_csv     = f'quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv'
        model_path = f'quality_results/Q_{q_idx}/paths_{paths_idx}/best_hb_model.zip'
        wafer_path = f'quality_results/Q_{q_idx}/paths_{paths_idx}/wafer_quality.json'

        quality_helper = QualityHelper(num_stages=env.num_stages,
                                       device=device,
                                       model_path=model_path,
                                       wafer_path=wafer_path)
        if not quality_helper.is_active:
            raise RuntimeError(
                f"quality pipeline not available — {model_path}, {wafer_path} 확인")
        # CSV/JSON 기본 범위(하드코딩) 를 CLI 입력으로 덮어씀 → U[wq_min, wq_max] 샘플.
        quality_helper.wafer_quality_min = wq_min
        quality_helper.wafer_quality_max = wq_max
        wq_1 = quality_helper.sample_wafer_quality(
            B=1, num_jobs=num_jobs, seed=seed + r, device=device)
        if wq_1 is None:
            raise RuntimeError("wafer_quality 샘플링 실패 (quality_helper inactive?)")
        # raw/percentile 둘 다 CSV quality factor 기반 ground-truth → 항상 생성.
        scorer = PathPercentileScorer(wq_csv, list(machine_cnt_list))

        print(f"[run {r+1}/{N_runs}] paths_idx={paths_idx}  "
              f"wq=[{wq_1.min().item():.4f},{wq_1.max().item():.4f}] "
              f"mean={wq_1.mean().item():.4f}")
        ms, yld = _run_single_experiment(
            model, env, env_edge_lookup_t, device,
            quality_helper, scorer,
            problems_INT_list, wq_1,
            lam_all_t, total, B, seed, greedy,
            method=method, yield_mode=yield_mode)
        runs_ms[r]  = ms
        runs_yld[r] = yld

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # ── 집계 ──
    # single_view='all' (& N_runs==1): 모든 (λ, sample) 점을 그대로 표시.
    # 그 외:
    #   1) 각 (run, λ) 의 samples_per_lambda 개 중 λ-가중 정규화 점수 argmax 1점 선택.
    #   2) 선택된 점들을 paths_idx 축으로 평균 → λ 당 1점.
    show_all = (single_view == 'all') and (N_runs == 1)
    if single_view == 'all' and N_runs > 1:
        print(f"[eval] single_view='all' 은 N_runs=1 일 때만 적용 — "
              f"N_runs={N_runs} 이므로 'best' 로 대체")
    if show_all:
        plot_ms  = runs_ms[0]                                                # (total,)
        plot_yld = runs_yld[0]                                               # (total,)
        plot_lam = lam_all_t.cpu().numpy()                                   # (total,)
    else:
        runs_ms_3d  = runs_ms.reshape(N_runs, num_lambdas, samples)
        runs_yld_3d = runs_yld.reshape(N_runs, num_lambdas, samples)
        if samples > 1:
            ms_score  = (hv_m_ref - runs_ms_3d) / (hv_m_ref - hv_m_best)     # ↓ → 1=best
            yld_score = (runs_yld_3d - hv_q_ref) / (hv_q_best - hv_q_ref)    # ↑ → 1=best
            lam_col   = lam_grid_np[None, :, None]                           # (1, N, 1)
            score     = (1.0 - lam_col) * ms_score + lam_col * yld_score     # (R, N, S)
            best_j    = np.argmax(score, axis=2)[..., None]                  # (R, N, 1)
            sel_ms    = np.take_along_axis(runs_ms_3d,  best_j, axis=2).squeeze(2)
            sel_yld   = np.take_along_axis(runs_yld_3d, best_j, axis=2).squeeze(2)
        else:
            sel_ms  = runs_ms_3d.squeeze(2)
            sel_yld = runs_yld_3d.squeeze(2)
        plot_ms  = sel_ms.mean(axis=0)                                       # (N,)
        plot_yld = sel_yld.mean(axis=0)                                      # (N,)
        plot_lam = lam_grid_np

    y_label_short = 'yield' if yield_mode == 'raw' else 'average path %tile'
    fmt = '.4f' if yield_mode == 'raw' else '.2f'
    stat_tag = 'all samples' if show_all else 'per-λ'

    print(f"[eval] hv anchors: m=[best={hv_m_best}, ref={hv_m_ref}]  "
          f"q=[ref={hv_q_ref}, best={hv_q_best}]")
    print(f"[eval] elapsed = {elapsed:.3f}s  "
          f"({N_runs * total / elapsed:.1f} samples/s)  N_runs={N_runs}  "
          f"single_view={single_view}")
    print(f"[eval] {stat_tag} makespan: mean={plot_ms.mean():.2f}  "
          f"min={plot_ms.min():.1f}  max={plot_ms.max():.1f}")
    print(f"[eval] {stat_tag} {y_label_short}: "
          f"mean={plot_yld.mean():{fmt}}  "
          f"min={plot_yld.min():{fmt}}  max={plot_yld.max():{fmt}}")

    title_mode = 'x1' if greedy else f'x{samples}'
    if N_runs == 1:
        run_tag = f'paths_{paths_idx_list[0]}'
    else:
        run_tag = f'paths {paths_idx_list[0]}~{paths_idx_list[-1]}'
    if method == 'EST':
        scatter_label = 'EST (λ-independent)'
    elif show_all:
        scatter_label = f'per-λ all samples'
    else:
        scatter_label = f'per-λ best average'
    method_tag = 'EST' if method == 'est' else 'model'
    title = (f'W={num_jobs}, (Q{q_idx},P{p_idx}), {run_tag}, '
             f'λ=x{num_lambdas}, s={title_mode}')

    front_idx = plot_pareto(
        plot_ms, plot_yld, plot_lam, save_path, title,
        yield_mode=yield_mode,
        xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks,
        scatter_label=scatter_label,
    )
    print(f"[eval] Pareto front size = {len(front_idx)}")
    print(f"saved -> {save_path}")

    return plot_ms, plot_yld, front_idx


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        type=str, default='checkpoints/saved_J25_baseline_0519.pt')
    p.add_argument('--method',      type=str, default='model',
                   choices=['model', 'est'],
                   help="스케줄링 정책. 'model' = 학습된 λ-conditioned 모델, "
                        "'est' = EST 휴리스틱 baseline (λ 무관, ckpt 불필요). "
                        "yield 계산(raw/percentile)은 두 경우 동일.")
    p.add_argument('--num_jobs',    type=int, default=15)
    p.add_argument('--machines',    type=str, default='5,3,7,3,5,7')
    p.add_argument('--yield_mode',  type=str, default='raw',
                   choices=['raw', 'percentile'],
                   help="둘 다 ground-truth (CSV quality factor 기반, 예측 모델 미사용). "
                        "'raw' = quality factor 의 path 곱 × 초기 품질 (job 평균), "
                        "'percentile' = 그 path 곱의 전체 분포 대비 백분위 (job 평균)")
    p.add_argument('--num_lambdas',        type=int, default=32,
                   help='λ grid 크기. linspace(0,1,N) 으로 sweep.')
    p.add_argument('--samples', type=int, default=1,
                   help='λ 당 trajectory 수. 1 = greedy, >1 = stochastic sampling.')
    p.add_argument('--paths_idx',   type=str, default='1~10',
                   help="historical_paths 인덱스. 단일('5') / range('1~10') / list('1,3,5'). "
                        "여러 개면 각 idx 별 1회 실험 후 (makespan, yield) 평균으로 Pareto 그림. "
                        "모델/풀 : quality_results/Q_{q_idx}/paths_{paths_idx}/{best_hb_model.zip, wafer_quality.json}. "
                        "입력 CSV : quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv")
    p.add_argument('--single_view', type=str, default='all',
                   choices=['best', 'all'],
                   help="paths_idx 가 1개일 때만 적용: "
                        "'best' = 람다별 λ-가중 정규화 점수 argmax 1점, "
                        "'all'  = 모든 (λ, sample) 점 표시. "
                        "paths_idx 가 여러 개면 항상 'best' 동작.")
    p.add_argument('--p_idx',       type=int, choices=[1, 2, 3], default=1,
                   help="proc_time 인스턴스 인덱스 → quality_data/P_{p_idx}.csv")
    p.add_argument('--q_idx',       type=int, choices=[1, 2, 3], default=1,
                   help="quality 시나리오 인덱스 (Q 폴더)")
    p.add_argument('--xlim', type=str, default='30,111,20',
                   help="x축 (makespan) 범위. 'lo,hi' 또는 'lo,hi,step' "
                        "(예: '40,170,30' → tick 40,70,100,130,160; xlim=170).")
    p.add_argument('--ylim', type=str, default='40,101,10',
                   help="y축 (yield/percentile) 범위. 'lo,hi' 또는 'lo,hi,step'.")
    p.add_argument('--wafer_quality', type=str, default='0.90,0.90',
                   help="초기 웨이퍼 품질 U[lo,hi] 샘플 범위. 'lo,hi' (예: '0.99,1.00'). "
                        "lo==hi 면 모든 job 이 상수 품질 (예: '1.00,1.00'). "
                        "raw yield 엔 직접 반영, percentile 모드에선 모델 입력에만 반영.")
    args = p.parse_args()

    machines = [int(x) for x in args.machines.split(',')]

    def _parse_idx_spec(s: str) -> list[int]:
        """'5' → [5]; '1~10' → [1..10]; '1,3,5' → [1,3,5]."""
        s = s.strip()
        if '~' in s:
            lo, hi = s.split('~')
            return list(range(int(lo), int(hi) + 1))
        if ',' in s:
            return [int(x) for x in s.split(',')]
        return [int(s)]

    paths_idx_list = _parse_idx_spec(args.paths_idx)

    def _parse_axis(s):
        if not s:
            return None, None
        parts = [float(x) for x in s.split(',')]
        if len(parts) == 2:
            return (parts[0], parts[1]), None
        if len(parts) == 3:
            lo, hi, step = parts
            ticks = np.arange(lo, hi, step)   # hi 는 tick 으로 안 찍음 (xlim 만 hi)
            return (lo, hi), ticks
        raise ValueError(f"axis must be 'lo,hi' or 'lo,hi,step', got: {s!r}")

    xlim, xticks = _parse_axis(args.xlim)
    ylim, yticks = _parse_axis(args.ylim)

    wq_parts = [float(x) for x in args.wafer_quality.split(',')]
    if len(wq_parts) != 2:
        raise ValueError(f"--wafer_quality must be 'lo,hi', got: {args.wafer_quality!r}")
    wq_min, wq_max = wq_parts

    paths_tag = args.paths_idx.replace('~', '-').replace(',', '+')
    save_path = f'test_results/{args.method}_J{args.num_jobs}_Q{args.q_idx}_P{args.p_idx}_p{len(paths_idx_list)}_s{args.samples}.png'
    evaluate_pareto(
        ckpt_path=args.ckpt,
        p_idx=args.p_idx,
        q_idx=args.q_idx,
        paths_idx_list=paths_idx_list,
        num_jobs=args.num_jobs,
        machine_cnt_list=machines,
        num_lambdas=args.num_lambdas,
        samples=args.samples,
        wq_min=wq_min,
        wq_max=wq_max,
        save_path=save_path,
        method=args.method,
        yield_mode=args.yield_mode,
        single_view=args.single_view,
        hv_m_best=xlim[0],
        hv_m_ref=xlim[1],
        hv_q_best=ylim[1],
        hv_q_ref=ylim[0],
        xlim=xlim,
        ylim=ylim,
        xticks=xticks,
        yticks=yticks,
    )
