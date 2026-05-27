"""
experiment_pareto.py — test.py 의 λ-conditioned 모델 Pareto front 평가를 *그대로* 하되,
같은 그림에 비교 알고리즘 2개(Comparison 1, Comparison 2) 와 EST 의 점/Pareto front 를
함께 overlay 한다.

- Ours(모델) / EST 점은 test.py 의 `_run_single_experiment` rollout 을 재사용하되, quality
  파이프라인은 예측 모델(QualityHelper) 이 아니라 GroundTruthQuality 를 쓴다 → ① 엣지 피처
  (quality_zscore) ② 리워드 ③ yield 채점 이 모두 historical_paths CSV 의 ground-truth quality
  factor 기반(oracle). 예측 모델 미개입. (experiment_table.py 의 quality 설정과 동일)
- Comparison 1 / 2 점은 quality_data/J_{W}_Q_{q}_P_{p}.txt 에서 로드한다 (각 행
  'makespan, percentile' 의 10-path 평균; makespan 올림·percentile 내림, 소수점 1자리
  → Ours 에 유리하게). 파일 마지막 두 줄이 Comparison 1(=7)·Comparison 2 행 수.
  올림으로 makespan 이 겹친 행은 '아래 행일수록 작은 makespan' 규칙으로 0.1 씩 분리해
  각 점을 distinct Pareto 점으로 유지. 그림엔 *선 없이 점만* 찍는다 (각 점이 개별 해).
  파일이 없으면 모델 점 범위 안 랜덤 placeholder 로 대체.

Pareto: makespan ↓ , yield ↑ 의 비지배 점들 (test.py 정의 그대로).
"""

from __future__ import annotations
import os
import sys
import time
import argparse

# 로컬 test.py 가 stdlib `test` 패키지에 가려지지 않도록 스크립트 디렉터리를 최우선에.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 콘솔(cp949)에서 ↑/↓/λ 등 유니코드 출력 시 UnicodeEncodeError 방지.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 그림 폰트를 논문 표(LaTeX serif)와 통일 — STIX(Times 류 serif, ↑↓λ 글리프 풀 커버 + bold 지원,
# matplotlib 기본 번들이라 머신 무관 재현). 없으면 Times New Roman → DejaVu Serif 로 폴백.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

from HFSPGraphEnv import HFSPGraphEnv
from FFSProblemDef import load_problems_from__quality_file
from FFSPModel import FFSPModel
from HFSPWrapper import make_env_edge_lookup, get_feat_dims
# 예측 모델 없음 — GroundTruthQuality 한 객체가 엣지 피처·리워드·채점을 모두 ground truth 로 제공.
from quality_augment import GroundTruthQuality

# test.py 의 rollout/HV/pareto 정의를 그대로 재사용 (quality 소스만 ground truth 로 교체).
from test import (
    compute_pareto_front,
    hypervolume,
    _run_single_experiment,
)


# =====================================================
# 비교 알고리즘 Pareto front — quality_data/J_{W}_Q_{q}_P_{p}.txt 에서 로드
# =====================================================
# 파일 구조:
#   - 데이터 행:    'makespan, percentile' (10-path 평균; 소수점 많음)
#   - 마지막 두 줄: Comparison 1 행 수(n1, 항상 7), Comparison 2 행 수(n2)
#   - 첫 n1 행 = Comparison 1,  다음 n2 행 = Comparison 2
# 반올림(Ours 에 유리하게, 소수점 1자리): makespan 올림(ceil), percentile 내림(floor).
COMPARISON_NAMES = ['Kim et al. (2025)', 'Lee et al. (2019)']

# overlay 그림에서 각 비교군의 색/마커.
_COMP_STYLE = {
    'Comparison 1': dict(color='tab:orange',  marker='o', size=50),
    'Comparison 2': dict(color='tab:blue', marker='^', size=58),   # 세모는 시각상 작아 약간 크게
}

# 범례·라벨 표기용 이름 (내부 키 Comparison 1/2 → 논문 저자명 COMPARISON_NAMES).
_COMP_DISPLAY = {
    'Comparison 1': COMPARISON_NAMES[0],
    'Comparison 2': COMPARISON_NAMES[1],
}


def _favorable_round(ms_raw: np.ndarray, yld_raw: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """makespan 올림·percentile 내림(소수점 1자리, Ours 에 유리하게).

    행 순서 = Pareto 순서 (위→아래로 makespan·percentile 감소). 올림 때문에 인접 행의
    makespan 이 같은 값으로 겹치면, 그 연속 run 안에서 *맨 아래 행을 가장 작게* 두고 위로
    갈수록 0.1 씩 올려 각 점을 distinct 한 Pareto 점으로 분리한다 (위 행을 올리므로 favorable
    유지). 겹치지 않는 점은 그대로 (Comparison 2 처럼 비단조 배열도 건드리지 않음).
    """
    ms = np.ceil(np.asarray(ms_raw, dtype=np.float64) * 10.0) / 10.0
    yld = np.floor(np.asarray(yld_raw, dtype=np.float64) * 10.0) / 10.0
    n = len(ms)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ms[j + 1] == ms[i]:     # 동일 makespan 연속 run [i..j]
            j += 1
        if j > i:                                   # 겹침 → 아래(j)=기준, 위로 +0.1
            base = ms[j]
            for k in range(i, j + 1):
                ms[k] = round(base + (j - k) * 0.1, 1)
        i = j + 1
    return ms, yld


def load_comparison_fronts(num_jobs: int, q_idx: int, p_idx: int,
                           data_dir: str = 'quality_data'
                           ) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """J_{W}_Q_{q}_P_{p}.txt → {'Comparison 1': (ms, yld), 'Comparison 2': (ms, yld)}.

    Comparison 1/2 각각에 _favorable_round 적용 (makespan 올림·percentile 내림 + 겹침 분리).
    파일이 없으면 None (호출부에서 랜덤 placeholder 로 대체).
    """
    path = os.path.join(data_dir, f'J_{num_jobs}_Q_{q_idx}_P_{p_idx}.txt')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"{path}: 줄이 너무 적음 ({len(lines)})")
    n1, n2 = int(lines[-2]), int(lines[-1])
    data_lines = lines[:-2]
    if len(data_lines) != n1 + n2:
        raise ValueError(
            f"{path}: 데이터 행 {len(data_lines)} != n1({n1})+n2({n2})={n1 + n2}")

    ms_raw, yld_raw = [], []
    for ln in data_lines:
        a, b = ln.split(',')
        ms_raw.append(float(a))
        yld_raw.append(float(b))
    ms_raw = np.asarray(ms_raw, dtype=np.float64)
    yld_raw = np.asarray(yld_raw, dtype=np.float64)
    # Comparison 1/2 는 별개 front → 겹침 분리도 그룹별로 (경계 넘어 섞지 않음).
    ms1, yld1 = _favorable_round(ms_raw[:n1], yld_raw[:n1])
    ms2, yld2 = _favorable_round(ms_raw[n1:n1 + n2], yld_raw[n1:n1 + n2])
    return {
        'Comparison 1': (ms1, yld1),
        'Comparison 2': (ms2, yld2),
    }


def random_pareto_front(rng: np.random.Generator,
                        ms_range: tuple[float, float],
                        yld_range: tuple[float, float],
                        n_points: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """min makespan / max yield 의 *그럴듯한* 랜덤 Pareto front (파일 없을 때만 쓰는 placeholder).

    front 는 makespan ↑ 일수록 yield ↑ (concave) 인 단조 증가 곡선 + 작은 노이즈.
    """
    m_lo, m_hi = ms_range
    y_lo, y_hi = yld_range
    ms = np.sort(rng.uniform(m_lo, m_hi, size=n_points))
    t = (ms - ms[0]) / (ms[-1] - ms[0] + 1e-9)              # 0..1
    base = y_lo + (y_hi - y_lo) * np.sqrt(t)                # concave 증가
    noise = rng.normal(0.0, (y_hi - y_lo) * 0.02, size=n_points)
    yld = np.maximum.accumulate(base + noise)              # 단조 비감소 보장
    yld = np.clip(yld, y_lo, y_hi)
    return ms.astype(np.float64), yld.astype(np.float64)


def random_comparison_fronts(model_ms: np.ndarray, model_yld: np.ndarray,
                             seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """파일이 없을 때만: 모델 점 범위 안에 비교군마다 다른 랜덤 front (임시 시각화)."""
    m_lo, m_hi = float(model_ms.min()), float(model_ms.max())
    y_lo, y_hi = float(model_yld.min()), float(model_yld.max())
    span = max(y_hi - y_lo, 1e-9)
    fronts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k, name in enumerate(COMPARISON_NAMES):
        rng = np.random.default_rng(seed + 1000 * (k + 1))
        top = y_hi - span * (0.10 + 0.12 * k)               # 비교군마다 상한을 조금씩 낮춤
        fronts[name] = random_pareto_front(
            rng, (m_lo, m_hi), (y_lo, top), n_points=7 + k)
    return fronts


# =====================================================
# 모델 λ-sweep 결과 집계 (test.py evaluate_pareto 의 집계 로직과 동일)
# =====================================================
def aggregate_model_points(runs_ms, runs_yld, lam_grid_np, lam_all_np,
                           num_lambdas, samples, single_view, N_runs,
                           hv_m_best, hv_m_ref, hv_q_best, hv_q_ref):
    """(N_runs, total) → (plot_ms, plot_yld, plot_lam). test.py 와 동일 규약.

    single_view='all' (& N_runs==1): 모든 (λ, sample) 점.
    그 외: (run, λ) 별 λ-가중 정규화 점수 argmax 1점 선택 후 paths 축 평균 → λ 당 1점.
    """
    show_all = (single_view == 'all') and (N_runs == 1)
    if single_view == 'all' and N_runs > 1:
        print(f"[eval] single_view='all' 은 N_runs=1 일 때만 적용 — "
              f"N_runs={N_runs} 이므로 'best' 로 대체")
    if show_all:
        return runs_ms[0], runs_yld[0], lam_all_np

    runs_ms_3d = runs_ms.reshape(N_runs, num_lambdas, samples)
    runs_yld_3d = runs_yld.reshape(N_runs, num_lambdas, samples)
    if samples > 1:
        ms_score = (hv_m_ref - runs_ms_3d) / (hv_m_ref - hv_m_best)      # ↓ → 1=best
        yld_score = (runs_yld_3d - hv_q_ref) / (hv_q_best - hv_q_ref)    # ↑ → 1=best
        lam_col = lam_grid_np[None, :, None]                            # (1, N, 1)
        score = (1.0 - lam_col) * ms_score + lam_col * yld_score        # (R, N, S)
        best_j = np.argmax(score, axis=2)[..., None]                    # (R, N, 1)
        sel_ms = np.take_along_axis(runs_ms_3d, best_j, axis=2).squeeze(2)
        sel_yld = np.take_along_axis(runs_yld_3d, best_j, axis=2).squeeze(2)
    else:
        sel_ms = runs_ms_3d.squeeze(2)
        sel_yld = runs_yld_3d.squeeze(2)
    return sel_ms.mean(axis=0), sel_yld.mean(axis=0), lam_grid_np


# =====================================================
# Overlay plot
# =====================================================
FIGSIZE_SINGLE = (6.2, 4.3)        # 단일 overlay 그림 (작게)
FIGSIZE_GRID = (13.0, 7.6)         # 2x3 합본 (윗줄 J15, 아랫줄 J25)


def _thin_close_points(ms, yld, xlim, ylim, strength):
    """strength(0..1) 로 서로 가까운 점을 솎아낸다 (Ours 프런티어 밀집 완화).

    0 = 미적용(전부 유지), 1 = 적당히 많이 제거. makespan 오름차순으로 훑으며 직전 *유지점*과의
    정규화 거리(축 범위로 나눈 Euclidean)가 임계 미만이면 건너뛴다. 양 끝점(최소·최대 makespan)
    은 항상 유지해 프런티어 범위를 보존. 반환: 입력 순서 기준 keep 불리언 마스크.
    """
    ms = np.asarray(ms, dtype=np.float64)
    yld = np.asarray(yld, dtype=np.float64)
    n = len(ms)
    if not strength or strength <= 0 or n <= 2:
        return np.ones(n, dtype=bool)
    min_gap = min(float(strength), 1.0) * 0.12        # strength=1 → 정규화 거리 0.12 이상만 유지
    x_lo, x_hi = (xlim if xlim is not None else (float(ms.min()), float(ms.max())))
    y_lo, y_hi = (ylim if ylim is not None else (float(yld.min()), float(yld.max())))
    x_rng = max(x_hi - x_lo, 1e-9)
    y_rng = max(y_hi - y_lo, 1e-9)
    order = np.argsort(ms)
    keep = np.zeros(n, dtype=bool)
    last = order[0]
    keep[last] = True                                 # 첫 점(최소 makespan)
    for idx in order[1:]:
        d = np.hypot((ms[idx] - ms[last]) / x_rng, (yld[idx] - yld[last]) / y_rng)
        if d >= min_gap:
            keep[idx] = True
            last = idx
    keep[order[-1]] = True                            # 마지막 점(최대 makespan)도 유지
    return keep


def _draw_overlay(ax, model_ms, model_yld, model_lam, comparison_fronts, *,
                  yield_mode='raw', xlim=None, ylim=None, xticks=None, yticks=None,
                  experiment=False, colorbar=True, legend=True, thin=0.0):
    """주어진 ax 에 overlay(점만) 를 그린다. figure 생성/저장은 호출부 담당.

    experiment=False: 전체 점, Ours 는 λ-color(viridis)(+colorbar). True: 비지배만, Ours 단일색.
    어느 쪽이든 *모든 알고리즘 점을 합친 전역 Pareto front* 점에 얇은 검정 ring.
    겹침 우선순위 Ours > Comparison 1 > Comparison 2 (ring 은 자기 점 +0.5 위).

    축 범위(xlim/ylim) 밖 점은 *그리지만* 않는다 — 시각화에서만 제외하고 Pareto front·HV·
    전역 ring 판정엔 그대로 포함한다 (makespan 이 xlim 상한을 넘는 해도 계산엔 반영).
    """
    model_ms = np.asarray(model_ms, dtype=np.float64)
    model_yld = np.asarray(model_yld, dtype=np.float64)

    # 각 시리즈의 *플롯될* 점 (experiment 면 자기 시리즈 비지배만)
    if experiment:
        k = compute_pareto_front(model_ms, model_yld)
        ours_ms, ours_yld = model_ms[k], model_yld[k]
        ours_lam = None                              # experiment 는 단일색 → λ 미사용
    else:
        ours_ms, ours_yld = model_ms, model_yld
        ours_lam = np.asarray(model_lam)
    # 밀집한 Ours 프런티어 솎기 (thin 0..1; 0=미적용). λ 색 배열도 같은 마스크로 동기화.
    if thin and thin > 0:
        keep = _thin_close_points(ours_ms, ours_yld, xlim, ylim, thin)
        ours_ms, ours_yld = ours_ms[keep], ours_yld[keep]
        if ours_lam is not None:
            ours_lam = ours_lam[keep]
    comp_plot = {}
    for name, (cms, cyld) in comparison_fronts.items():
        cms = np.asarray(cms, dtype=np.float64)
        cyld = np.asarray(cyld, dtype=np.float64)
        if experiment:
            k = compute_pareto_front(cms, cyld)
            cms, cyld = cms[k], cyld[k]
        comp_plot[name] = (cms, cyld)

    # 전역 비지배: 모든 알고리즘(Ours + 비교군) 점을 합쳐 한 번에 Pareto 판정 (EST 제외)
    parts = ([('Ours', ours_ms, ours_yld)]
             + [(n, comp_plot[n][0], comp_plot[n][1]) for n in comp_plot])
    all_ms = np.concatenate([pt[1] for pt in parts])
    all_yld = np.concatenate([pt[2] for pt in parts])
    gmask = np.zeros(all_ms.size, dtype=bool)
    gmask[compute_pareto_front(all_ms, all_yld)] = True
    off = np.cumsum([0] + [pt[1].size for pt in parts])
    gm = {pt[0]: gmask[off[i]:off[i + 1]] for i, pt in enumerate(parts)}

    # 축 범위 밖(makespan>xlim 상한 등) 점을 *그리기 직전에만* 솎는 마스크. 위의 전역 Pareto/
    # HV 판정은 전체 점으로 이미 끝났으므로 여기서 빼도 계산엔 영향 없음. xlim/ylim=None → 전부.
    def _in_range(ms_a, yld_a):
        m = np.ones(np.shape(ms_a), dtype=bool)
        if xlim is not None:
            m &= (ms_a >= xlim[0]) & (ms_a <= xlim[1])
        if ylim is not None:
            m &= (yld_a >= ylim[0]) & (yld_a <= ylim[1])
        return m

    def _ring(ms_a, yld_a, mask, marker, size, z):
        # 전역 비지배 & in-range 점에만 face 없는 얇은 검정 ring. clip_on=False → 경계 점도 안 잘림.
        if ms_a.size == 0 or not mask.any():
            return
        ax.scatter(ms_a[mask], yld_a[mask], s=size, marker=marker, facecolors='none',
                   edgecolors='black', linewidths=0.5, zorder=z, clip_on=False)

    n_comp = len(comp_plot)
    ours_z = n_comp + 2                                  # 비교군 전부보다 앞

    # Ours (범위 밖 점은 시각화에서만 제외; 전역 ring 판정엔 이미 포함됨)
    vis_o = _in_range(ours_ms, ours_yld)
    if experiment:
        ax.scatter(ours_ms[vis_o], ours_yld[vis_o], s=40, marker='s', color='crimson',
                   alpha=0.9, edgecolors='none', zorder=ours_z, clip_on=False,
                   label=f'Ours  (n={int(vis_o.sum())})')
    else:
        # colorbar 는 λ 의미(0~1)대로 vmin/vmax 고정 — 범위 밖 점이 빠져도 색 스케일 불변.
        sc = ax.scatter(ours_ms[vis_o], ours_yld[vis_o], c=ours_lam[vis_o], cmap='viridis',
                        s=34, marker='s', vmin=0.0, vmax=1.0, alpha=0.9, edgecolors='none',
                        zorder=ours_z, clip_on=False, label=f'Ours  (n={int(vis_o.sum())})')
        if colorbar:
            cbar = ax.figure.colorbar(sc, ax=ax)
            cbar.set_label('λ (0 = makespan-only,  1 = yield-only)')
    _ring(ours_ms, ours_yld, gm['Ours'] & vis_o, 's', 40 if experiment else 34, ours_z + 0.5)

    # 비교 알고리즘 1, 2 (먼저 나온 비교군이 더 앞; 범위 밖 점은 시각화에서만 제외)
    for idx, (name, (cms, cyld)) in enumerate(comp_plot.items()):
        st = _COMP_STYLE.get(name, dict(color=None, marker='o', size=50))
        csize = st.get('size', 50)
        z = 2 + (n_comp - 1 - idx)
        vis_c = _in_range(cms, cyld)
        ax.scatter(cms[vis_c], cyld[vis_c], s=csize, marker=st['marker'], color=st['color'],
                   alpha=0.9, edgecolors='white', linewidths=0.5, zorder=z, clip_on=False,
                   label=f'{_COMP_DISPLAY.get(name, name)}  (n={int(vis_c.sum())})')
        _ring(cms, cyld, gm[name] & vis_c, st['marker'], csize, z + 0.5)

    y_axis_label = ('Yield ↑' if yield_mode == 'raw'
                    else 'Average Yield Percentile ↑')
    ax.set_xlabel('Average Makespan ↓')
    ax.set_ylabel(y_axis_label)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)              # 범위 밖 점은 위에서 이미 제외; 경계의 in-range 점만 안 잘림
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, linestyle='--', alpha=0.4)
    if legend:
        ax.legend(loc='best', fontsize=8)


def plot_overlay(model_ms, model_yld, model_lam,
                 est_pt, comparison_fronts,
                 save_path, title, yield_mode='raw',
                 xlim=None, ylim=None, xticks=None, yticks=None,
                 experiment=False, thin=0.0):
    """단일 overlay 그림 1장 생성 + 저장 (작은 figsize). 그리기는 _draw_overlay 재사용."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    _draw_overlay(ax, model_ms, model_yld, model_lam, comparison_fronts,
                  yield_mode=yield_mode, xlim=xlim, ylim=ylim, xticks=xticks,
                  yticks=yticks, experiment=experiment, colorbar=True, legend=True,
                  thin=thin)
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =====================================================
# Evaluation
# =====================================================
def load_overlay_model(ckpt_path: str, env: HFSPGraphEnv, device):
    """ckpt 차원으로 FFSPModel 빌드 + 가중치 로드. num_jobs 와 무관 (ckpt 가 dim 결정)
    → batch 에서 한 번만 로드해 여러 (num_jobs, Q, P) config 에 재사용 가능."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_params = ckpt['model_params']
    edge_feat_dim = (ckpt.get('edge_feat_dim')
                     or model_params.pop('edge_feature_dim', None)
                     or get_feat_dims(env)[2])
    model = FFSPModel(ckpt['row_feat_dim'], ckpt['col_feat_dim'],
                      edge_feat_dim, **model_params).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def evaluate_pareto_overlay(
    ckpt_path='checkpoints/saved_J25_new_baseline.pt',
    p_idx: int = 3,
    q_idx: int = 3,
    paths_idx_list=(1,),
    num_jobs=25,
    machine_cnt_list=(5, 3, 7, 3, 5, 7),
    num_lambdas=32,
    samples=8,
    seed=0,
    comp_seed=0,
    wq_min: float = 0.99,
    wq_max: float = 1.00,
    save_path='test_results/pareto_overlay.png',
    yield_mode: str = 'percentile',
    single_view: str = 'best',
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: np.ndarray | None = None,
    yticks: np.ndarray | None = None,
    experiment: bool = False,
    thin: float = 0.0,
    model=None,
    device=None,
    ax=None,
):
    """test.py 와 동일하게 모델 Pareto front 를 평가하고, EST + 비교군 2개를 overlay.

    HV anchor 는 test.py 규약대로 축에서 유도:
        m=[best=xlim[0], ref=xlim[1]],  q=[ref=ylim[0], best=ylim[1]].
    """
    if yield_mode not in ('raw', 'percentile'):
        raise ValueError(f"yield_mode must be 'raw' or 'percentile', got {yield_mode!r}")
    if single_view not in ('best', 'all'):
        raise ValueError(f"single_view must be 'best' or 'all', got {single_view!r}")
    paths_idx_list = list(paths_idx_list)
    if not paths_idx_list:
        raise ValueError("paths_idx_list must be non-empty")
    num_lambdas = int(num_lambdas)
    samples = int(samples)
    if num_lambdas < 2:
        raise ValueError(f"num_lambdas must be >= 2, got {num_lambdas}")
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")

    # ── HV anchor (축에서 유도; percentile 은 q 를 0~100 강제) ──
    hv_m_best, hv_m_ref = (xlim if xlim is not None else (0.0, 1.0))
    hv_q_ref, hv_q_best = (ylim if ylim is not None else (0.0, 1.0))
    if yield_mode == 'percentile':
        hv_q_best, hv_q_ref = 100.0, 0.0

    p_csv = f'quality_data/P_{p_idx}.csv'
    total = num_lambdas * samples
    greedy = (samples == 1)
    B = 1
    N_runs = len(paths_idx_list)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    # ── env ──
    env = HFSPGraphEnv(num_jobs=num_jobs, machine_cnt_list=list(machine_cnt_list), device=device)
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)

    # ── model: 외부서 주입되면 그대로 재사용(batch), 없으면 ckpt 에서 로드 ──
    if model is None:
        model = load_overlay_model(ckpt_path, env, device)

    # ── 고정 proc time ──
    problems_INT_list = load_problems_from__quality_file(
        p_csv, num_jobs, list(machine_cnt_list))

    # ── λ grid ──
    lam_grid_t = torch.linspace(0.0, 1.0, num_lambdas, device=device)        # (N,)
    lam_all_t = lam_grid_t.repeat_interleave(samples)                        # (total,)
    lam_grid_np = lam_grid_t.cpu().numpy()
    lam_all_np = lam_all_t.cpu().numpy()
    lam0_t = torch.zeros(1, device=device)                                   # EST 용 더미 λ

    print(f"[eval] device={device}  ckpt={ckpt_path}  yield_mode={yield_mode}")
    print(f"[eval] jobs={num_jobs}  machines={list(machine_cnt_list)}  "
          f"NE={env.num_edges}  num_lambdas={num_lambdas}  samples/λ={samples}  "
          f"total={total}  greedy={greedy}")
    print(f"[eval] proc_time <- {p_csv}   wafer_quality ~ U[{wq_min:.4f}, {wq_max:.4f}]")
    print(f"[eval] Q_{q_idx}  paths_idx_list={paths_idx_list}  N_runs={N_runs}")
    print(f"[eval] HV anchors: m=[best={hv_m_best}, ref={hv_m_ref}]  "
          f"q=[ref={hv_q_ref}, best={hv_q_best}]")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()

    runs_ms = np.zeros((N_runs, total), dtype=np.float32)
    runs_yld = np.zeros((N_runs, total), dtype=np.float32)
    est_ms_runs = np.zeros(N_runs, dtype=np.float32)
    est_yld_runs = np.zeros(N_runs, dtype=np.float32)

    for r, paths_idx in enumerate(paths_idx_list):
        wq_csv = f'quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv'

        # 예측 모델 없음 — ground-truth quality 한 객체가 엣지 피처(compute_machine_quality)·
        # 리워드(compute_yield)·평가 채점(score/path_product)을 모두 CSV factor 로 제공 (oracle).
        gq = GroundTruthQuality(num_stages=env.num_stages, device=device, csv_path=wq_csv,
                                machine_cnt_list=list(machine_cnt_list),
                                wafer_quality_min=wq_min, wafer_quality_max=wq_max)
        wq_1 = gq.sample_wafer_quality(B=1, num_jobs=num_jobs, seed=seed + r, device=device)
        quality_helper = scorer = gq                    # 동일 ground-truth 객체

        print(f"[run {r+1}/{N_runs}] paths_idx={paths_idx}  "
              f"wq=[{wq_1.min().item():.4f},{wq_1.max().item():.4f}] "
              f"mean={wq_1.mean().item():.4f}")

        # ── Ours (모델 λ-sweep) ──
        ms, yld = _run_single_experiment(
            model, env, env_edge_lookup_t, device,
            quality_helper, scorer, problems_INT_list, wq_1,
            lam_all_t, total, B, seed, greedy,
            method='model', yield_mode=yield_mode)
        runs_ms[r] = ms
        runs_yld[r] = yld

        # ── EST (λ 무관 단일 점; yield 계산은 모델과 동일 ground-truth) ──
        est_ms, est_yld = _run_single_experiment(
            None, env, env_edge_lookup_t, device,
            quality_helper, scorer, problems_INT_list, wq_1,
            lam0_t, 1, B, seed, True,
            method='est', yield_mode=yield_mode)
        est_ms_runs[r] = est_ms[0]
        est_yld_runs[r] = est_yld[0]

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    # ── 집계 ──
    plot_ms, plot_yld, plot_lam = aggregate_model_points(
        runs_ms, runs_yld, lam_grid_np, lam_all_np,
        num_lambdas, samples, single_view, N_runs,
        hv_m_best, hv_m_ref, hv_q_best, hv_q_ref)
    est_pt = (float(est_ms_runs.mean()), float(est_yld_runs.mean()))

    fmt = '.4f' if yield_mode == 'raw' else '.2f'
    print(f"[eval] elapsed = {elapsed:.3f}s  "
          f"({N_runs * (total + 1) / elapsed:.1f} samples/s)")
    print(f"[eval] Ours  makespan: mean={plot_ms.mean():.2f}  "
          f"min={plot_ms.min():.1f}  max={plot_ms.max():.1f}")
    print(f"[eval] Ours  yield: mean={plot_yld.mean():{fmt}}  "
          f"min={plot_yld.min():{fmt}}  max={plot_yld.max():{fmt}}")
    print(f"[eval] EST   point: makespan={est_pt[0]:.2f}  yield={est_pt[1]:{fmt}}")

    # ── 비교 알고리즘 front: quality_data/J_{W}_Q_{q}_P_{p}.txt 우선, 없으면 랜덤 ──
    comparison_fronts = load_comparison_fronts(num_jobs, q_idx, p_idx)
    comp_src = 'file'
    if comparison_fronts is None:
        raise ValueError(f"[eval] 비교 데이터 파일 quality_data/J_{num_jobs}_Q_{q_idx}_P_{p_idx}.txt "
              f"없음")
    for name, (cms, cyld) in comparison_fronts.items():
        print(f"[eval] {name} ({comp_src}): n={len(cms)}  "
              f"ms=[{cms.min():.1f},{cms.max():.1f}]  "
              f"yld=[{cyld.min():{fmt}},{cyld.max():{fmt}}]")

    # ── HV (test.py / nsga2.py 와 동일 정의·anchor → 직접 비교 가능) ──
    hv_box = (hv_m_ref - hv_m_best) * (hv_q_best - hv_q_ref)

    def _hv_norm(ms_a, yld_a):
        hv_raw, _ = hypervolume(np.asarray(ms_a, np.float64),
                                np.asarray(yld_a, np.float64), hv_m_ref, hv_q_ref)
        return hv_raw / hv_box if hv_box > 0 else float('nan')

    print(f"[eval] HV_norm  Ours={_hv_norm(plot_ms, plot_yld):.4f}  "
          f"EST={_hv_norm([est_pt[0]], [est_pt[1]]):.4f}  "
          + "  ".join(f"{n}={_hv_norm(c[0], c[1]):.4f}"
                      for n, c in comparison_fronts.items()))

    # ── 그림 ──
    title_mode = 'x1' if greedy else f'x{samples}'
    run_tag = (f'paths_{paths_idx_list[0]}' if N_runs == 1
               else f'paths {paths_idx_list[0]}~{paths_idx_list[-1]}')
    title = (f'J={num_jobs}, (Q{q_idx},P{p_idx}), {run_tag}, '
             f'λ=x{num_lambdas}, s={title_mode}')

    if ax is None:
        plot_overlay(plot_ms, plot_yld, plot_lam, est_pt, comparison_fronts,
                     save_path, title, yield_mode=yield_mode,
                     xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks,
                     experiment=experiment, thin=thin)
        print(f"saved -> {save_path}")
    else:
        # grid subplot 에 직접 그림 (figure/colorbar/legend/저장은 호출부가 관리)
        _draw_overlay(ax, plot_ms, plot_yld, plot_lam, comparison_fronts,
                      yield_mode=yield_mode, xlim=xlim, ylim=ylim,
                      xticks=xticks, yticks=yticks, experiment=experiment,
                      colorbar=False, legend=False, thin=thin)
        ax.set_title(f'J={num_jobs}, (Q{q_idx}, P{p_idx})', fontweight='bold')
    return plot_ms, plot_yld, est_pt, comparison_fronts


# =====================================================
# Batch configs — run_all.bat 의 6개 (num_jobs, q_idx, p_idx, xlim, ylim) 조합.
#   percentile 전용 (yield 0~100 스케일). ylim 은 W=15 '40,100,10', W=25 '30,100,10'.
#   run_all.bat SAMPLES=8 → batch 기본 --samples 8 (stochastic sampling).
# =====================================================
BATCH_CONFIGS = [
    dict(num_jobs=15, q_idx=1, p_idx=1, xlim='30,110,20',  ylim='40,100,10'),
    dict(num_jobs=15, q_idx=2, p_idx=2, xlim='50,230,20',  ylim='40,100,10'),
    dict(num_jobs=15, q_idx=3, p_idx=3, xlim='70,330,30',  ylim='40,100,10'),
    dict(num_jobs=25, q_idx=1, p_idx=1, xlim='40,180,30',  ylim='30,100,10'),
    dict(num_jobs=25, q_idx=2, p_idx=2, xlim='40,360,80',  ylim='30,100,10'),
    dict(num_jobs=25, q_idx=3, p_idx=3, xlim='100,500,100', ylim='30,100,10'),
]


def _parse_idx_spec(s: str) -> list[int]:
    s = s.strip()
    if '~' in s:
        lo, hi = s.split('~')
        return list(range(int(lo), int(hi) + 1))
    if ',' in s:
        return [int(x) for x in s.split(',')]
    return [int(s)]


def _parse_axis(s):
    if not s:
        return None, None
    parts = [float(x) for x in s.split(',')]
    if len(parts) == 2:
        return (parts[0], parts[1]), None
    if len(parts) == 3:
        lo, hi, step = parts
        # lo + k·step 격자 틱만 (hi 까지). hi 가 격자에 안 맞으면 끝점 틱을 따로 붙이지 않는다
        # → 축은 hi 까지 그려지되 마지막 라벨은 격자에 맞는 값까지만 (예: 40~180,30 → …160).
        ticks = np.arange(lo, hi + step * 1e-6, step)
        return (lo, hi), ticks
    raise ValueError(f"axis must be 'lo,hi' or 'lo,hi,step', got: {s!r}")


# =====================================================
# Entry
# =====================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="HFSP Pareto overlay (Ours/model + Comparison 1/2 + EST). "
                    "기본 mode=batch: run_all.bat 의 6개 config 를 percentile 로 순차 실행.")
    p.add_argument('--mode',        type=str, default='single', choices=['batch', 'single'],
                   help="'batch'(기본)=6개를 한 그림(2x3, 윗줄 J15·아랫줄 J25)으로 합본 저장, "
                        "'single'=1개만.")
    p.add_argument('--ckpt',        type=str, default='checkpoints/0527_pbi.pt')
    p.add_argument('--num_jobs',    type=int, default=25, help="single mode 전용.")
    p.add_argument('--machines',    type=str, default='5,3,7,3,5,7')
    p.add_argument('--num_lambdas', type=int, default=32, help='λ grid 크기. linspace(0,1,N).')
    p.add_argument('--samples',     type=int, default=64,
                   help='λ 당 trajectory 수. >1 = stochastic sampling (run_all.bat SAMPLES=8), 1 = greedy.')
    p.add_argument('--p_idx',       type=int, choices=[1, 2, 3], default=3, help="single mode 전용.")
    p.add_argument('--q_idx',       type=int, choices=[1, 2, 3], default=3, help="single mode 전용.")
    p.add_argument('--xlim',        type=str, default='100,500,100',
                   help="single mode 전용 x축 (makespan) 'lo,hi' 또는 'lo,hi,step'. "
                        "batch 는 config 별 xlim 사용. HV anchor m=[best=lo, ref=hi].")
    p.add_argument('--ylim',        type=str, default='30,100,10',
                   help="single mode 전용 y축 (yield) 'lo,hi' 또는 'lo,hi,step'. "
                        "batch 는 config 별 ylim 사용. percentile 은 0~100 스케일. "
                        "HV anchor q=[ref=lo, best=hi] (percentile 은 0~100 강제).")
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00',
                   help="초기 웨이퍼 품질 U[lo,hi]. (예: '0.99,1.00')") # 유리하게 하려면 1 고정이 나을수도
    p.add_argument('--comp_seed',   type=int, default=0,
                   help="비교군 랜덤 front 생성 seed (실데이터 주입 전까지만 의미).")
    p.add_argument('--thin',        type=float, default=0,
                   help="Ours 프런티어에서 서로 가까운 점 솎기 강도 0~1. "
                        "0=미적용(전부), 1=적당히 많이 제거.")
    p.add_argument('--out',         type=str, default='',
                   help="single mode 저장 경로. 비우면 test_results/ 아래 자동 명명.")
    p.add_argument('--experiment',  action='store_false',
                   help="실험 버전 ON: 각 시리즈 지배당하는 해 제거(비지배 점만) + "
                        "Ours 를 λ 색 없이 단일 색(crimson). 파일명에 _exp 접미사. "
                        "OFF: 전체 점 + Ours λ-color.")
    args = p.parse_args()

    machines = [int(x) for x in args.machines.split(',')]
    args.paths_idx = '1~10'
    paths_idx_list = _parse_idx_spec(args.paths_idx)

    wq_parts = [float(x) for x in args.wafer_quality.split(',')]
    if len(wq_parts) != 2:
        raise ValueError(f"--wafer_quality must be 'lo,hi', got: {args.wafer_quality!r}")
    wq_min, wq_max = wq_parts

    def _auto_save(num_jobs, q_idx, p_idx):
        exp_tag = '_exp' if args.experiment else ''
        return (f'test_results/pareto_J{num_jobs}_'
                f'Q{q_idx}_P{p_idx}_p{len(paths_idx_list)}_s{args.samples}.png')

    if args.mode == 'batch':
        # batch = 6개 config 를 한 그림(2x3)으로 합본. BATCH_CONFIGS 순서가 곧 윗줄 J15·아랫줄 J25.
        # 루프에서 각 config 를 subplot(ax)에 그리고, 맨 마지막에 공통 범례+제목 붙여 저장.
        # 모델은 ckpt 차원으로 빌드 → num_jobs 무관. 한 번만 로드해 6개 config 에 재사용.
        from matplotlib.lines import Line2D
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tmp_env = HFSPGraphEnv(num_jobs=BATCH_CONFIGS[0]['num_jobs'],
                               machine_cnt_list=machines, device=device)
        model = load_overlay_model(args.ckpt, tmp_env, device)
        fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_GRID)
        n = len(BATCH_CONFIGS)
        for i, (ax, cfg) in enumerate(zip(axes.flat, BATCH_CONFIGS), 1):
            xlim, xticks = _parse_axis(cfg['xlim'])
            ylim, yticks = _parse_axis(cfg['ylim'])
            print(f"\n=== [{i}/{n}] J={cfg['num_jobs']}, (Q{cfg['q_idx']},P{cfg['p_idx']}) ===")
            evaluate_pareto_overlay(
                ckpt_path=args.ckpt, model=model, device=device, ax=ax,
                p_idx=cfg['p_idx'], q_idx=cfg['q_idx'],
                paths_idx_list=paths_idx_list,
                num_jobs=cfg['num_jobs'], machine_cnt_list=machines,
                num_lambdas=args.num_lambdas, samples=args.samples,
                comp_seed=args.comp_seed, wq_min=wq_min, wq_max=wq_max,
                save_path='', experiment=args.experiment, thin=args.thin,
                xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
        # 맨 마지막: 공통 범례(아래 중앙, n 표기 없이) + 제목 + 저장
        ours_fc = 'crimson' if args.experiment else plt.cm.viridis(0.6)
        handles = [
            Line2D([0], [0], linestyle='', marker='s', markersize=9,
                   markerfacecolor=ours_fc, markeredgecolor='white',
                   label='Ours'),
            Line2D([0], [0], linestyle='', marker=_COMP_STYLE['Comparison 1']['marker'],
                   markersize=9, markerfacecolor=_COMP_STYLE['Comparison 1']['color'],
                   markeredgecolor='white', label=_COMP_DISPLAY['Comparison 1']),
            Line2D([0], [0], linestyle='', marker=_COMP_STYLE['Comparison 2']['marker'],
                   markersize=9, markerfacecolor=_COMP_STYLE['Comparison 2']['color'],
                   markeredgecolor='white', label=_COMP_DISPLAY['Comparison 2']),
        ]
        fig.tight_layout(rect=[0, 0.13, 1, 0.94])     # 아래 범례 + 위 제목 공간
        # 범례를 subplot 바로 아래에 바짝 (upper-center 기준점을 subplot 하단 근처로).
        # 글자는 subplot 제목(12pt) + 볼드.
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.13),
                   ncol=3, frameon=True, prop=dict(size=12, weight='bold'),
                   columnspacing=0.6, handletextpad=0.4)
        # 제목을 상단 subplot 바로 위로 내림 (기본 y=0.98 이라 너무 떠 보임).
        fig.suptitle('Pareto Frontier Graph', fontsize=16, fontweight='bold', y=0.965)
        out = args.out or f'test_results/pareto_grid_p{len(paths_idx_list)}_s{args.samples}.png'
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"\nsaved -> {out}\n=== done ===")
    else:  # single
        xlim, xticks = _parse_axis(args.xlim)
        ylim, yticks = _parse_axis(args.ylim)
        save_path = args.out or _auto_save(args.num_jobs, args.q_idx, args.p_idx)
        evaluate_pareto_overlay(
            ckpt_path=args.ckpt,
            p_idx=args.p_idx,
            q_idx=args.q_idx,
            paths_idx_list=paths_idx_list,
            num_jobs=args.num_jobs,
            machine_cnt_list=machines,
            num_lambdas=args.num_lambdas,
            samples=args.samples,
            comp_seed=args.comp_seed,
            wq_min=wq_min,
            wq_max=wq_max,
            save_path=save_path,
            experiment=args.experiment,
            thin=args.thin,
            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            yticks=yticks,
        )
