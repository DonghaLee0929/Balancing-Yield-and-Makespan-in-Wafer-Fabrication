"""
experiment_continual.py — 웨이퍼 *지속학습(continual learning)* 비교 표/그림.

스토리: (P3, Q3) 에서 학습한 λ-conditioned 정책을 분포가 크게 바뀐 (P1, Q1) 환경에서
'이어 학습' 했을 때의 **새 환경 적응 효율(전이/파인튜닝)** 을 본다. 모든 방법을 *동일한
타깃 인스턴스(P1, Q1)* 에서, *동일한 ground-truth Q1 scorer* 로 채점해 공정 비교한다.

핵심 설계 — `_run_single_experiment` 에서 `quality_helper`(정책이 *보는* 엣지 피처
quality_zscore) 와 `scorer`(*실제* yield 채점) 가 분리돼 있다는 사실을 그대로 활용한다:
    · makespan 은 보상 G 에서, yield 는 scorer 에서 따로 계산됨 (test.py).
    · quality_helper 는 eval 롤아웃에서 오직 compute_machine_quality(엣지 피처)에만 쓰임
      (HFSPWrapper.build_model_state) — compute_yield 는 호출되지 않음.
따라서 '피처/보상 예측기를 따로 갈아끼우는' 4개 조건은 *eval 시 quality_helper 만 바꾸면*
정확히 재현된다. scorer 는 언제나 타깃 Q1 ground truth 로 고정한다.

비교 행 (각 정책 ckpt 는 사용자가 따로 학습해 둔다 — 이 스크립트는 *평가/비교*만):
  scratch     : (P1,Q1) 으로 처음부터 학습한 정책      | 피처=Q1
  full-adapt  : (P3,Q3)→(P1,Q1) 이어학습, 피처·보상 모두 Q1 (= Ours/제안) | 피처=Q1
  stale-feat  : (P3,Q3)→(P1,Q1) 이어학습, 보상만 Q1·피처는 옛 Q3 유지       | 피처=Q3(stale)
  masked-feat : (P3,Q3)→(P1,Q1) 이어학습, 보상만 Q1·피처는 -1 마스킹        | 피처=마스킹
  zero-shot   : (P3,Q3) 정책을 *추가 학습 없이* (P1,Q1) 에서 평가 (= 학습곡선의 t=0) | 피처=Q1
  NSGA        : (P1,Q1) 인스턴스를 처음부터 푸는 NSGA-II (비학습 참조 천장)

⚠ eval 시 각 행의 피처 소스는 *그 정책이 학습된 피처 조건과 반드시 일치해야* 의미가 산다
   (stale-feat 정책은 Q3 피처로 학습됐어야 하고, masked-feat 정책은 -1 마스킹으로 학습됐어야
   한다). 스크립트는 ckpt 의 학습 조건을 강제하지 못하므로, 행별 ckpt 를 올바르게 지정할 것.

지표(HV/IGD+/Makespan/Quality/Time)·HV anchor·정규화·IGD+ 기준집합은 experiment_table.py /
test.py / nsga2.py 의 코드를 그대로 import 해 쓰므로 수치가 그 스크립트들과 1:1 일치한다.
"""

from __future__ import annotations
import os
import sys
import json
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
from matplotlib.lines import Line2D

# 그림 폰트를 논문 표(LaTeX serif)와 통일 — STIX(Times 류 serif, ↑↓λ 글리프 풀 커버 + bold 지원,
# matplotlib 기본 번들이라 머신 무관 재현). 없으면 Times New Roman → DejaVu Serif 로 폴백.
# (experiment_pareto.py 와 동일 설정)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

from HFSPGraphEnv import HFSPGraphEnv
from FFSPModel import FFSPModel
from HFSPWrapper import make_env_edge_lookup, get_feat_dims
from quality_augment import GroundTruthQuality, load_proc_time_augmented

# test.py 의 모델/EST rollout 평가 머신 재사용.
from test import _run_single_experiment, compute_pareto_front
# nsga2.py 의 NSGA-II decoder/loop 재사용.
from nsga2 import HFSPDecoder, run_nsga2
# experiment_table.py 의 지표(HV/IGD+)·타이밍·포맷 + 캐시 정합성 검증 헬퍼 재사용.
from experiment_table import (
    compute_metrics,
    build_reference_front,
    igd_plus_metric,
    timed,
    _fmt,
    cache_meta_mismatch,
    COMPARISON_DIR,
)


# 표·그림의 행 순서 (canonical). 실제 출력은 ckpt 가 주어진 행 + NSGA 만.
ALL_ROWS = ['scratch', 'full-adapt', 'stale-feat', 'masked-feat', 'zero-shot', 'NSGA']
HIGHLIGHT_ROW = 'full-adapt'        # 제안 방법 (표에 * 강조)

# 행별 피처 소스 키: 'target'=Q1 GT, 'stale'=Q3 GT, 'masked'=-1 상수.
_ROW_FEAT = {
    'scratch':     'target',
    'full-adapt':  'target',
    'stale-feat':  'stale',
    'masked-feat': 'masked',
    'zero-shot':   'target',
}

# 0523_baseline 에서 fine-tune 으로 갈라진 adapt 행들. 학습 곡선 plot 에서 이들의 *t=0*
# = zero-shot 값으로 prepend → 모두 같은 출발점에서 시작하는 그림. (epoch_1.pt 는 1 epoch
# *훈련된 후* 의 ckpt 라 prepend 안 하면 정책 trajectory 의 진짜 시작점이 안 보임.)
_ADAPT_ROWS = {'full-adapt', 'stale-feat', 'masked-feat'}


# 그림용 색/마커. continual 4행 + zero-shot 은 *red 계열* 로 통일 — 모두 같은
# 0523_baseline 에서 갈라진 family 임을 시각적으로 전달. scratch 만 회색(이 계열
# 밖, '처음부터' 학습이라 다른 origin).
_ROW_STYLE = {
    'scratch':        dict(color='tab:gray',  marker='o'),
    # ↓ red family — full-adapt(주역) 가장 진한 crimson, ablation 행들은 변주.
    'full-adapt':     dict(color='crimson',   marker='s'),     # main proposed
    'stale-feat':     dict(color='indianred', marker='^'),     # 중간 톤 red
    'masked-feat':    dict(color='lightcoral', marker='v'),    # 옅은 pink-red
    'zero-shot':      dict(color='firebrick', marker='D'),     # 진하지만 dashed → 참조선
    'NSGA':           dict(color='tab:blue',  marker='P'),
}


# =====================================================
# 피처 소스 — masked wrapper
# =====================================================
class MaskedQuality:
    """compute_machine_quality 만 상수(mask_value)로 덮는 quality_helper.

    _run_single_experiment 의 모델 롤아웃은 quality_helper 를 *오직* 엣지 피처
    (compute_machine_quality) 에만 쓴다(보상/채점은 scorer) → 이 한 메서드면 충분.
    유효 엣지에 mask_value(-1)를 채우면 HFSPWrapper.build_model_state 의 무효 엣지 -1
    패딩과 일관 → 학습 시 '피처 -1 마스킹' 조건과 정확히 일치.
    """

    def __init__(self, mask_value: float = -1.0):
        self.mask_value = float(mask_value)

    def compute_machine_quality(self, env) -> torch.Tensor:
        return torch.full((env.batch_size, env.num_jobs, env.total_machines),
                          self.mask_value, dtype=torch.float32, device=env.device)


# =====================================================
# 모델 로딩
# =====================================================
def load_policy(ckpt_path, env, device):
    """ckpt 차원으로 FFSPModel 빌드 + 가중치 로드. 그래프 모델이라 머신/잡 수 무관."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mp = ckpt['model_params']
    edge_feat_dim = (ckpt.get('edge_feat_dim')
                     or mp.pop('edge_feature_dim', None)
                     or get_feat_dims(env)[2])
    model = FFSPModel(ckpt['row_feat_dim'], ckpt['col_feat_dim'],
                      edge_feat_dim, **mp).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


# =====================================================
# 방법별 front 산출 (모델 λ-sweep / NSGA) — experiment_shift.py 와 동일 호출
# =====================================================
def model_front(model, env, env_edge_lookup_t, device, quality_helper, scorer,
                proc, wq_1, num_lambdas, samples, seed, yield_mode):
    """λ-sweep 으로 모델 Pareto front 산출. samples>1 = stochastic sampling, 1 = greedy."""
    lam_g = torch.linspace(0.0, 1.0, num_lambdas, device=device)
    if samples <= 1:
        lam, total, greedy = lam_g, num_lambdas, True
    else:
        lam = lam_g.repeat_interleave(samples)
        total, greedy = num_lambdas * samples, False
        torch.manual_seed(seed)                       # 샘플링 재현성 (방법 순서 무관)
    return _run_single_experiment(
        model, env, env_edge_lookup_t, device, quality_helper, scorer,
        proc, wq_1, lam, total, 1, seed, greedy,
        method='model', yield_mode=yield_mode)


def nsga_front(env, scorer, proc, wq_1, machines, J, S, anchors,
               nsga_pop, nsga_gen, ga_seed, yield_mode):
    """타깃 Q1 GT scorer 기반 NSGA-II → (ms, gt_yield). experiment_shift.py 와 동일 호출."""
    env_edge_lookup_np = env.env_edge_lookup.cpu().numpy()
    machine_offset_np = np.cumsum([0] + machines[:-1]).astype(np.int64)
    wq_np_j = wq_1[0].cpu().numpy().astype(np.float64)
    decoder = HFSPDecoder(env, proc, env_edge_lookup_np, machine_offset_np,
                          scorer, wq_np_j, yield_mode=yield_mode,
                          yield_objective='ground_truth', quality_helper=None)
    _, _, ms_n, gt_n, _, _ = run_nsga2(
        decoder, J, S, machines, pop_size=nsga_pop, n_gen=nsga_gen,
        seed=ga_seed, hv_anchors=anchors)
    return ms_n.astype(np.float32), gt_n.astype(np.float32)


# =====================================================
# 한 path 인스턴스에 대해 활성 행 모두 실행
# =====================================================
def evaluate_one_path(*, policies, feat_src, device, J, S, machines,
                      q_idx, src_q_idx, paths_idx, src_paths_idx, p_idx,
                      anchors, seed, ga_seed, wq_min, wq_max, mask_value,
                      num_lambdas, samples, nsga_pop, nsga_gen, yield_mode,
                      run_nsga):
    """한 path → {row: (ms, yld, time_s)} dict.

    policies : {row: model} (ckpt 가 주어진 정책 행만). feat_src 인자는 사용 안 함(여기서 구성).
    """
    # ── 타깃 (P1,Q1) ground truth: scorer 겸 'target' 피처 소스 ──
    tgt_csv = f'quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv'
    gq_tgt = GroundTruthQuality(num_stages=S, device=device, csv_path=tgt_csv,
                                machine_cnt_list=list(machines),
                                wafer_quality_min=wq_min, wafer_quality_max=wq_max)

    # ── stale 피처 소스: 옛 (Q3) 품질 landscape (stale-feat 행이 있을 때만 로드) ──
    feat_helpers = {'target': gq_tgt, 'masked': MaskedQuality(mask_value)}
    need_stale = any(_ROW_FEAT.get(r) == 'stale' for r in policies)
    if need_stale:
        src_csv = f'quality_data/Q_{src_q_idx}/historical_paths_{src_paths_idx}.csv'
        feat_helpers['stale'] = GroundTruthQuality(
            num_stages=S, device=device, csv_path=src_csv,
            machine_cnt_list=list(machines),
            wafer_quality_min=wq_min, wafer_quality_max=wq_max)

    # ── env / proc_time(P1) ──
    env = HFSPGraphEnv(num_jobs=J, machine_cnt_list=list(machines), device=device)
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)
    proc = load_proc_time_augmented(f'quality_data/P_{p_idx}.csv', J, list(machines))

    # 모든 방법이 동일 wafer_quality 를 보도록 한 번만 샘플 (타깃 GT 에서).
    wq_1 = gq_tgt.sample_wafer_quality(B=1, num_jobs=J, seed=seed, device=device)   # (1, J)

    out = {}
    # ── 정책 행들: 피처 소스만 행마다 다르게, scorer 는 모두 타깃 Q1 GT ──
    for row, model in policies.items():
        qh = feat_helpers[_ROW_FEAT[row]]
        (ms, yld), t = timed(lambda m=model, q=qh: model_front(
            m, env, env_edge_lookup_t, device, q, gq_tgt, proc, wq_1,
            num_lambdas, samples, seed, yield_mode), device)
        out[row] = (ms, yld, t)

    # ── NSGA (P1,Q1) 참조 ──
    if run_nsga:
        (ms, yld), t = timed(lambda: nsga_front(
            env, gq_tgt, proc, wq_1, list(machines), J, S, anchors,
            nsga_pop, nsga_gen, ga_seed, yield_mode), device)
        out['NSGA'] = (ms, yld, t)

    return out


# =====================================================
# 표 렌더링
# =====================================================
def build_table(agg, header, n_paths, rows):
    """agg[row][metric] = list(per-path values) → 출력 문자열 + rows(list of dict)."""
    cols = ['Method', 'HV ↑', 'IGD+ ↓', 'Makespan ↓', 'Quality ↑', 'Time (s)']
    decimals = {'HV ↑': 4, 'IGD+ ↓': 4, 'Makespan ↓': 1, 'Quality ↑': 4, 'Time (s)': 2}
    key_map = {'HV ↑': 'HV', 'IGD+ ↓': 'IGD+', 'Makespan ↓': 'Makespan',
               'Quality ↑': 'Quality', 'Time (s)': 'Time'}

    table_rows = []
    for r in rows:
        label = r + (' *' if r == HIGHLIGHT_ROW else '')
        row = {'Method': label}
        for c in cols[1:]:
            row[c] = _fmt(agg[r][key_map[c]], decimals[c])
        table_rows.append(row)

    try:
        import pandas as pd
        body = pd.DataFrame(table_rows, columns=cols).set_index('Method').to_string()
    except Exception:
        widths = {c: max(len(c), *(len(r[c]) for r in table_rows)) for c in cols}
        line = '  '.join(c.ljust(widths[c]) for c in cols)
        sep = '  '.join('-' * widths[c] for c in cols)
        body = '\n'.join([line, sep] + [
            '  '.join(str(r[c]).ljust(widths[c]) for c in cols) for r in table_rows])

    n_tag = (f"(avg over {n_paths} paths, mean±std)" if n_paths > 1
             else "(single path)")
    out = f"\n{header}\n{n_tag}\n{body}\n(* = full-adapt = 제안 방법)\n"
    return out, table_rows


# =====================================================
# Overlay plot — 행별 Pareto front (점 + 비지배 선)
# =====================================================
def plot_overlay(per_row_pts, anchors, save_path, title, rows):
    m_best, m_ref, q_best, q_ref = anchors
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name in rows:
        if name not in per_row_pts:
            continue
        ms, yld = per_row_pts[name]
        ms = np.asarray(ms, np.float64)
        yld = np.asarray(yld, np.float64)
        st = _ROW_STYLE.get(name, dict(color=None, marker='o'))
        front = compute_pareto_front(ms, yld)
        ax.scatter(ms, yld, s=14, alpha=0.20, color=st['color'])
        if front.size >= 2:
            ax.plot(ms[front], yld[front], '-', marker=st['marker'], ms=5, lw=1.3,
                    color=st['color'], label=name)
        else:                                   # 단일 점(zero-shot 등) → 큰 마커
            ax.scatter(ms[front], yld[front], s=90, marker=st['marker'],
                       color=st['color'], edgecolor='black', zorder=5, label=name)
    ax.set_xlabel('Makespan ↓')
    ax.set_ylabel('Yield ↑')
    ax.set_xlim(m_best, m_ref)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# =====================================================
# 학습 곡선(sweep) 모드 — 행별 ckpt 디렉터리에서 epoch_{e}.pt 를 sweep
# =====================================================
def _setup_path(*, device, J, S, machines, q_idx, src_q_idx, paths_idx,
                src_paths_idx, p_idx, wq_min, wq_max, mask_value, seed,
                need_stale):
    """env / proc / 행별 feat_helpers / 공통 wq_1 한 번에 셋업 (sweep / snapshot 공통)."""
    tgt_csv = f'quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv'
    gq_tgt = GroundTruthQuality(num_stages=S, device=device, csv_path=tgt_csv,
                                machine_cnt_list=list(machines),
                                wafer_quality_min=wq_min, wafer_quality_max=wq_max)
    feat_helpers = {'target': gq_tgt, 'masked': MaskedQuality(mask_value)}
    if need_stale:
        src_csv = f'quality_data/Q_{src_q_idx}/historical_paths_{src_paths_idx}.csv'
        feat_helpers['stale'] = GroundTruthQuality(
            num_stages=S, device=device, csv_path=src_csv,
            machine_cnt_list=list(machines),
            wafer_quality_min=wq_min, wafer_quality_max=wq_max)
    env = HFSPGraphEnv(num_jobs=J, machine_cnt_list=list(machines), device=device)
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)
    proc = load_proc_time_augmented(f'quality_data/P_{p_idx}.csv', J, list(machines))
    wq_1 = gq_tgt.sample_wafer_quality(B=1, num_jobs=J, seed=seed, device=device)
    return gq_tgt, feat_helpers, env, env_edge_lookup_t, proc, wq_1


def _eval_policy(model, qh, env, env_edge_lookup_t, device, gq_tgt, proc, wq_1,
                 num_lambdas, samples, seed, yield_mode):
    """단일 정책 → (ms, yld, time). model_front 를 timed 로 감쌌을 뿐."""
    (ms, yld), t = timed(lambda m=model, q=qh: model_front(
        m, env, env_edge_lookup_t, device, q, gq_tgt, proc, wq_1,
        num_lambdas, samples, seed, yield_mode), device)
    return ms, yld, t


def _smooth(a, window):
    """중앙형 이동평균. window<=1 이면 원본 반환. 가장자리는 가능한 범위만 평균."""
    a = np.asarray(a, dtype=np.float64)
    if window <= 1 or a.size <= 1:
        return a.copy()
    half = window // 2
    out = np.full_like(a, np.nan)
    for i in range(a.size):
        lo, hi = max(0, i - half), min(a.size, i + half + 1)
        out[i] = np.nanmean(a[lo:hi])
    return out


def _rolling_std(a, window):
    """중앙형 rolling std — single-path 의 noise envelope 음영 용."""
    a = np.asarray(a, dtype=np.float64)
    if window <= 1 or a.size <= 1:
        return np.zeros_like(a)
    half = window // 2
    out = np.zeros_like(a)
    for i in range(a.size):
        lo, hi = max(0, i - half), min(a.size, i + half + 1)
        out[i] = np.nanstd(a[lo:hi]) if hi - lo > 1 else 0.0
    return out


# 플롯 스타일 상수 — 논문 그림 디자인 반복 시 여기만 만지면 됨.
# experiment_pareto.py 의 grid 합본 그림과 동일 컨벤션 (하단 공통 범례 + 상단 suptitle).
_PLOT_SMOOTH_WINDOW = 7         # 이동평균 window (epoch 단위)
_PLOT_LINE_WIDTH = 2.0
_PLOT_REF_LINE_WIDTH = 1.4
_PLOT_BAND_ALPHA = 0.18
_PLOT_LEGEND_FONTSIZE = 12      # 하단 공통 범례 — pareto 와 동일
_PLOT_SUPTITLE_FONTSIZE = 16    # 상단 큰 제목 — pareto 와 동일
_PLOT_FIGSIZE = (15.0, 5.2)     # suptitle + 하단 legend 공간 확보 위해 살짝 키움


def plot_curves(curve_agg, static_agg, epoch_list, sweep_rows, save_path, title):
    """3-panel (HV ↑ / Makespan ↓ / Quality ↑) × x=epoch — 학술 그림 스타일.

    곡선: 마커 없이 깨끗한 line (이동평균 + rolling std 음영 band).
    참조점: dashed 수평선.
    범례: 패널 별 ax.legend 대신 **fig.legend 하단 공통 1개** (experiment_pareto.py 와 동일 패턴).
    """
    epochs = np.array(epoch_list, dtype=np.float64)
    metrics = [('HV', 'HV ↑'), ('Makespan', 'Makespan ↓'), ('Quality', 'Quality ↑')]
    # zero-shot 평가 = 0523_baseline 그대로 → adapt 행들의 *t=0 정책 상태*. 곡선의 epoch=0
    # 점으로 prepend 해 모두 같은 출발점에서 갈라지는 모양을 만든다 (시각·의미 정합).
    zs_agg = static_agg.get('zero-shot', {})
    has_zs = bool(zs_agg.get('HV'))
    fig, axes = plt.subplots(1, 3, figsize=_PLOT_FIGSIZE)
    for ax, (key, label) in zip(axes, metrics):
        zs_v = (float(np.nanmean(zs_agg[key]))
                if has_zs and zs_agg.get(key) else None)
        for r in sweep_rows:
            vals_per_e = [curve_agg[r][e][key] for e in epoch_list]
            raw_mean = np.array([float(np.nanmean(v)) if v else np.nan
                                 for v in vals_per_e])
            n_paths = max((len(v) for v in vals_per_e), default=1)
            mean = _smooth(raw_mean, _PLOT_SMOOTH_WINDOW)
            if n_paths > 1:
                path_std = np.array([float(np.nanstd(v)) if len(v) > 1 else 0.0
                                     for v in vals_per_e])
                band = _smooth(path_std, _PLOT_SMOOTH_WINDOW)
            else:                                       # 1 path → rolling std envelope
                band = _rolling_std(raw_mean, _PLOT_SMOOTH_WINDOW)
            # Adapt 행: epoch=0 점에 zero-shot 값 prepend (정책 trajectory 의 실제 시작점).
            # *smoothing 이후* 에 prepend 해 t=0 이 정확히 zero-shot 값에 anchor 되도록 한다.
            if r in _ADAPT_ROWS and zs_v is not None:
                x = np.concatenate([[0.0], epochs])
                mean = np.concatenate([[zs_v], mean])
                band = np.concatenate([[0.0], band])    # 단일 deterministic 점 → 분산 0
            else:
                x = epochs
            st = _ROW_STYLE.get(r, dict(color=None, marker='o'))
            ax.plot(x, mean, '-', color=st['color'], lw=_PLOT_LINE_WIDTH)
            if np.any(band > 0):
                ax.fill_between(x, mean - band, mean + band,
                                color=st['color'], alpha=_PLOT_BAND_ALPHA,
                                linewidth=0)
        for name, d in static_agg.items():
            vals = d[key]
            if not vals:
                continue
            v = float(np.nanmean(vals))
            st = _ROW_STYLE.get(name, dict(color='black', marker='o'))
            ax.axhline(v, linestyle='--', color=st['color'],
                       lw=_PLOT_REF_LINE_WIDTH, alpha=0.85)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(label)
        ax.grid(True, linestyle='--', alpha=0.4)

    # 하단 공통 범례 proxy handles. 순서: scratch family(처음부터-학습 origin) 먼저,
    # 그다음 adapt family(0523_baseline 에서 갈라진 행들) + zero-shot/NSGA 참조선.
    def _handle(name, ls, lw):
        st = _ROW_STYLE.get(name, dict(color='black', marker='o'))
        return Line2D([0], [0], linestyle=ls, linewidth=lw,
                      color=st['color'], label=name)

    handles = []
    # 1) scratch (실선) — 회색 origin
    if 'scratch' in sweep_rows:
        handles.append(_handle('scratch', '-', _PLOT_LINE_WIDTH))
    # 2) adapt family — 나머지 sweep 행 (실선): full-adapt, [stale-feat], masked-feat
    for r in sweep_rows:
        if r == 'scratch':
            continue
        handles.append(_handle(r, '-', _PLOT_LINE_WIDTH))
    # 3) 정적 참조 (dashed) — zero-shot, NSGA
    for name, d in static_agg.items():
        if not d.get('HV'):
            continue
        handles.append(_handle(name, '--', _PLOT_REF_LINE_WIDTH))

    fig.tight_layout(rect=[0, 0.13, 1, 0.94])           # 아래 범례 + 위 suptitle 공간
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.13),
               ncol=len(handles), frameon=True,
               prop=dict(size=_PLOT_LEGEND_FONTSIZE, weight='bold'),
               columnspacing=0.8, handletextpad=0.4)
    fig.suptitle(title, fontsize=_PLOT_SUPTITLE_FONTSIZE, fontweight='bold', y=0.965)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def write_curve_csv(curve_agg, static_agg, epoch_list, sweep_rows, csv_path):
    """long-form: 한 행 = (method, epoch) 의 평균/표준편차 + n_paths. 참조점은 epoch=-1."""
    keys = ('HV', 'IGD+', 'Makespan', 'Quality', 'Time')
    rows = []

    def _summarize(d, method, epoch):
        if not d['HV']:
            return None
        rec = {'method': method, 'epoch': epoch, 'n_paths': len(d['HV'])}
        for k in keys:
            v = np.asarray(d[k], dtype=np.float64)
            rec[f'{k}_mean'] = float(np.nanmean(v)) if v.size else float('nan')
            rec[f'{k}_std'] = float(np.nanstd(v)) if v.size > 1 else 0.0
        return rec

    for r in sweep_rows:
        for e in epoch_list:
            rec = _summarize(curve_agg[r][e], r, e)
            if rec is not None:
                rows.append(rec)
    for name, d in static_agg.items():
        rec = _summarize(d, name, -1)
        if rec is not None:
            rows.append(rec)

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    except Exception:
        cols = ['method', 'epoch', 'n_paths'] + [f'{k}_{s}' for k in keys for s in ('mean', 'std')]
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write(','.join(cols) + '\n')
            for r in rows:
                f.write(','.join(str(r.get(c, '')) for c in cols) + '\n')


# =====================================================
# 점 캐시 — (W,Q,P) 당 *2 파일*:
#   continual_points_W{J}_Q{q}P{p}.jsonl       (매 epoch 한 줄 append, 점 데이터)
#   continual_points_W{J}_Q{q}P{p}.meta.json   (그룹키 + artifacts, 가끔 rewrite)
#
# jsonl 한 줄 = {"path","slug","epoch","ms","yld","t"} 단일 점.
#   - Append-only — 매 epoch I/O 가 그 한 줄(~30 KB) 뿐, 전체 rewrite 0
#   - 자연 dedup — 같은 (path,slug,epoch) 가 다시 들어오면 load 시 *마지막 라인* 이 이김
#   - 다른 row 안 만지면 그 라인 그대로 → 머지 위험 0 (실수로 wipe 불가능)
#   - 사람이 즉시 inspect 가능 (`head -1 *.jsonl | jq` 또는 텍스트 에디터)
#
# 그룹키(model/nsga) 검증은 experiment_shift.py 패턴 그대로 — 한 그룹 설정이 바뀌면 그 그룹의
# 캐시 hit 가 거부되고 새 점들이 append 됨. artifact (slug_e{epoch}: sig) 는 meta.json 에서
# 일관 관리 — ckpt mtime/size 가 바뀌면 그 (행, epoch) 만 무효화.
# =====================================================
_SWEEP_SLUG = {'scratch': 'scratch', 'full-adapt': 'fulladapt',
               'stale-feat': 'stalefeat', 'masked-feat': 'maskedfeat'}
_STATIC_SLUG = {'zero-shot': 'zeroshot', 'NSGA': 'nsga'}


def _artifact_sig(path):
    """ckpt 파일 정체성 — 경로+mtime+size. 재학습돼 파일 바뀌면 키가 달라져 자동 무효화."""
    if not path:
        return None
    try:
        st = os.stat(path)
        return {'path': str(path), 'mtime': float(st.st_mtime), 'size': int(st.st_size)}
    except OSError:
        return {'path': str(path), 'mtime': None, 'size': None}


def _continual_cache_path(J, q_idx, p_idx):
    return os.path.join(COMPARISON_DIR, f"continual_points_W{J}_Q{q_idx}P{p_idx}.jsonl")


def _continual_meta_path(J, q_idx, p_idx):
    return os.path.join(COMPARISON_DIR, f"continual_points_W{J}_Q{q_idx}P{p_idx}.meta.json")


def _build_cur_metas(*, args, machines, wq_min, wq_max, anchors):
    """검증키 두 그룹 — model 행(λ-sweep)·nsga 행 각각이 점에 영향받는 설정만 담는다.
    artifact 는 여기 안 들어가고 meta.json 의 별도 dict 로 관리(=세밀 부분 무효화)."""
    base = {'J': int(args.num_jobs), 'base_machines': list(machines),
            'q_idx': int(args.q_idx), 'p_idx': int(args.p_idx),
            'wq_min': float(wq_min), 'wq_max': float(wq_max),
            'yield_mode': str(args.yield_mode)}
    model = {**base, 'src_q_idx': int(args.src_q_idx),
             'src_paths_idx': int(args.src_paths_idx),
             'mask_value': float(args.mask_value),
             'num_lambdas': int(args.num_lambdas),
             'samples': int(args.samples), 'seed': int(args.seed)}
    nsga = {**base, 'nsga_pop': int(args.nsga_pop), 'nsga_gen': int(args.nsga_gen),
            'ga_seed': int(args.ga_seed),
            'anchors': [float(a) for a in anchors]}
    return {'model': model, 'nsga': nsga}


def _load_continual_cache(J, q_idx, p_idx):
    """jsonl + meta.json → (cache_data, file_meta). 파일 없으면 빈 값.

    cache_data 형식은 dict {'p{path}_{slug}_e{epoch}_{ms|yld|t}': np.array} — 옛 npz 인터페이스
    그대로 유지해 downstream 코드 그대로 사용. 같은 (path,slug,epoch) 가 jsonl 에 여러 번
    나오면 *나중 라인* 이 이김 (자연 dedup).
    """
    cache_data = {}
    jsonl_path = _continual_cache_path(J, q_idx, p_idx)
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    path = int(rec['path'])
                    slug = str(rec['slug'])
                    epoch = int(rec['epoch'])
                    pk = f'p{path}_{slug}_e{epoch}'
                    cache_data[f'{pk}_ms'] = np.asarray(rec['ms'], np.float32)
                    cache_data[f'{pk}_yld'] = np.asarray(rec['yld'], np.float32)
                    cache_data[f'{pk}_t'] = np.asarray(float(rec.get('t', 0.0)), np.float32)
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue                                  # 잘못된 라인은 무시

    file_meta = {'groups': {}, 'artifacts': {}}
    meta_path = _continual_meta_path(J, q_idx, p_idx)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            file_meta['groups'] = {k: m[k] for k in ('model', 'nsga') if k in m}
            file_meta['artifacts'] = m.get('artifacts', {}) or {}
        except Exception:
            pass

    return cache_data, file_meta


def _append_eval_record(J, q_idx, p_idx, paths_idx, slug, epoch, ms, yld, t):
    """jsonl 에 *한 줄* append — 매 epoch I/O 가 ~수십 KB 한 줄 뿐, 전체 rewrite 0.

    파일이 없으면 새로 만든다. 동시 쓰기 안 보호함 (한 머신에서 순차 실행 가정).
    """
    jsonl_path = _continual_cache_path(J, q_idx, p_idx)
    os.makedirs(os.path.dirname(jsonl_path) or '.', exist_ok=True)
    rec = {
        'path': int(paths_idx),
        'slug': str(slug),
        'epoch': int(epoch),
        'ms': np.asarray(ms, np.float32).ravel().tolist(),
        'yld': np.asarray(yld, np.float32).ravel().tolist(),
        't': float(t),
    }
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def _save_continual_meta(J, q_idx, p_idx, cur_metas, artifacts):
    """meta.json rewrite (~수 KB) — group keys 와 artifacts. jsonl 과 분리해 lightweight."""
    meta_path = _continual_meta_path(J, q_idx, p_idx)
    os.makedirs(os.path.dirname(meta_path) or '.', exist_ok=True)
    meta = {
        'model': cur_metas['model'],
        'nsga': cur_metas['nsga'],
        'artifacts': artifacts,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _cached_or_eval(*, slug, epoch, paths_idx, ckpt_path, group_ok, refresh,
                    cache_data, file_artifacts, new_artifacts, compute_fn,
                    J, q_idx, p_idx):
    """캐시 hit → 즉시 반환, miss → compute_fn() 호출 후 jsonl 에 *한 줄 append*.

    ckpt_path=None 이면 artifact 검증 생략 (NSGA: 그룹 키만으로 판별).
    반환: (ms, yld, t, hit:bool). cache_data 도 in-memory 로 갱신 → 같은 run 안 후속 코드 일관.
    """
    pk = f'p{paths_idx}_{slug}_e{epoch}'
    art_key = f'{slug}_e{epoch}' if ckpt_path else None
    cur_sig = _artifact_sig(ckpt_path) if ckpt_path else None
    if art_key is not None:
        # 파일이 사라진 경우 None sig 로 덮어쓰지 말고 이전 저장본 보존
        # → 다음 재실행에서도 stored 가 살아 있어 캐시 hit 가능.
        if cur_sig and cur_sig.get('mtime') is None and art_key in file_artifacts:
            new_artifacts[art_key] = file_artifacts[art_key]
        else:
            new_artifacts[art_key] = cur_sig

    # Cache hit
    if not refresh and group_ok and f'{pk}_ms' in cache_data:
        artifact_ok = True
        if art_key is not None:
            stored = file_artifacts.get(art_key)
            if stored is None:
                artifact_ok = False
            elif cur_sig and cur_sig.get('mtime') is None:
                artifact_ok = True                          # ckpt 삭제됨 → 캐시 신뢰
            else:
                artifact_ok = (json.dumps(stored, sort_keys=True) ==
                               json.dumps(cur_sig, sort_keys=True))
        if artifact_ok:
            ms = cache_data[f'{pk}_ms']
            yld = cache_data[f'{pk}_yld']
            t = float(cache_data[f'{pk}_t']) if f'{pk}_t' in cache_data else 0.0
            return ms, yld, t, True

    # Cache miss — compute + append 한 줄
    ms, yld, t = compute_fn()
    ms_arr = np.asarray(ms, np.float32).ravel()
    yld_arr = np.asarray(yld, np.float32).ravel()
    cache_data[f'{pk}_ms'] = ms_arr                          # in-memory 갱신
    cache_data[f'{pk}_yld'] = yld_arr
    cache_data[f'{pk}_t'] = np.asarray(float(t), np.float32)
    _append_eval_record(J, q_idx, p_idx, paths_idx, slug, epoch, ms_arr, yld_arr, t)
    return ms_arr, yld_arr, t, False


def run_sweep(*, args, machines, J, S, anchors, paths_idx_list, wq_min, wq_max,
              epoch_list, sweep_rows, run_nsga, device):
    """학습 곡선 모드: jsonl 캐시 (train_continual.py 가 매 epoch 적재) 에서 sweep 점을 읽고
    zero-shot/NSGA 참조점은 1회씩 평가. sweep 행은 ckpt 를 안 읽음 — jsonl 이 단일 소스.

    반환: (curve_agg, static_agg). curve_agg[row][epoch][metric] = list(per-path).
    """
    need_stale = any(_ROW_FEAT[r] == 'stale' for r in sweep_rows)

    curve_agg = {r: {e: {'HV': [], 'IGD+': [], 'Makespan': [], 'Quality': [], 'Time': []}
                     for e in epoch_list} for r in sweep_rows}

    # ── 정적 참조 메타: zero-shot + NSGA ──
    static_names = []
    static_meta = {}                       # name → {'ckpt': path, 'feat': feat_helper key}
    if args.ckpt_src:
        static_names.append('zero-shot')
        static_meta['zero-shot'] = {'ckpt': args.ckpt_src, 'feat': 'target'}
        print(f"[continual] zero-shot ckpt <- {args.ckpt_src}")
    if run_nsga:
        static_names.append('NSGA')
    static_agg = {name: {'HV': [], 'IGD+': [], 'Makespan': [], 'Quality': [], 'Time': []}
                  for name in static_names}

    # ── 점 캐시 로드 + 그룹 검증키 매치 ──
    cur_metas = _build_cur_metas(args=args, machines=machines, wq_min=wq_min, wq_max=wq_max,
                                 anchors=anchors)
    cache_data, file_meta = _load_continual_cache(J, args.q_idx, args.p_idx)
    file_groups = file_meta.get('groups', {})
    file_artifacts = file_meta.get('artifacts', {})
    refresh = bool(args.cache_refresh)
    model_ok = (not refresh) and not cache_meta_mismatch(file_groups.get('model'),
                                                          cur_metas['model'])
    nsga_ok = (not refresh) and not cache_meta_mismatch(file_groups.get('nsga'),
                                                         cur_metas['nsga'])
    if refresh:
        print("[cache] --cache_refresh → 캐시 무시·재계산 (jsonl 에 새 라인 append)")
    elif cache_data:
        if not (model_ok or nsga_ok):
            miss_m = cache_meta_mismatch(file_groups.get('model'), cur_metas['model'])
            miss_n = cache_meta_mismatch(file_groups.get('nsga'), cur_metas['nsga'])
            print(f"[cache] jsonl 있지만 그룹 키 mismatch — model:{miss_m[:3]} nsga:{miss_n[:3]} → 재계산 append")
        else:
            kinds = ([n for n in ('model', 'nsga') if (n == 'model' and model_ok) or
                      (n == 'nsga' and nsga_ok)])
            print(f"[cache] 활성 그룹: {kinds}  (file={_continual_cache_path(J, args.q_idx, args.p_idx)})")
            print(f"[cache] 옛 점 {len(cache_data) // 3}개 로드 (jsonl 한 줄 = 한 점)")

    # Append-only 라 미터치 슬러그는 jsonl 안 라인 그대로 살아있음 (preserve-merge 불필요).
    new_artifacts = dict(file_artifacts)    # 옛 artifacts 출발점, 새 평가마다 갱신

    # ── 정책 lazy 로드 — 캐시 풀히트면 한 번도 안 부른다 ──
    _model_cache = {}               # name → loaded policy
    _tmp_env = [None]

    def _get_static_model(name):
        if name not in _model_cache:
            if _tmp_env[0] is None:
                _tmp_env[0] = HFSPGraphEnv(num_jobs=J, machine_cnt_list=machines, device=device)
            _model_cache[name] = load_policy(static_meta[name]['ckpt'], _tmp_env[0], device)
        return _model_cache[name]

    for paths_idx in paths_idx_list:
        print(f"\n========== paths_{paths_idx} ==========")
        gq_tgt, feat_helpers, env, env_edge_lookup_t, proc, wq_1 = _setup_path(
            device=device, J=J, S=S, machines=machines,
            q_idx=args.q_idx, src_q_idx=args.src_q_idx,
            paths_idx=paths_idx, src_paths_idx=args.src_paths_idx, p_idx=args.p_idx,
            wq_min=wq_min, wq_max=wq_max, mask_value=args.mask_value,
            seed=args.seed, need_stale=need_stale)

        path_static_pts = {}

        # 정적 참조 행 (zero-shot)
        for name in static_names:
            if name == 'NSGA':
                continue
            def _compute_static(name=name):
                model = _get_static_model(name)
                qh = feat_helpers[static_meta[name]['feat']]
                return _eval_policy(model, qh, env, env_edge_lookup_t, device,
                                    gq_tgt, proc, wq_1, args.num_lambdas, args.samples,
                                    args.seed, args.yield_mode)
            ms, yld, t, hit = _cached_or_eval(
                slug=_STATIC_SLUG[name], epoch=-1, paths_idx=paths_idx,
                ckpt_path=static_meta[name]['ckpt'],
                group_ok=model_ok, refresh=refresh,
                cache_data=cache_data, file_artifacts=file_artifacts,
                new_artifacts=new_artifacts,
                compute_fn=_compute_static,
                J=J, q_idx=args.q_idx, p_idx=args.p_idx)
            path_static_pts[name] = (ms, yld)
            md = compute_metrics(ms, yld, anchors)
            for k in ('HV', 'Makespan', 'Quality'):
                static_agg[name][k].append(md[k])
            static_agg[name]['Time'].append(t)
            tag = ' [cache]' if hit else ''
            print(f"  {name:13s} HV={md['HV']:.4f}  ms={md['Makespan']:.1f}  "
                  f"q={md['Quality']:.4f}  t={t:.2f}s{tag}")

        # NSGA
        if run_nsga:
            def _compute_nsga():
                (msn, yldn), tn = timed(lambda: nsga_front(
                    env, gq_tgt, proc, wq_1, list(machines), J, S, anchors,
                    args.nsga_pop, args.nsga_gen, args.ga_seed, args.yield_mode), device)
                return msn, yldn, tn
            ms, yld, t, hit = _cached_or_eval(
                slug=_STATIC_SLUG['NSGA'], epoch=-1, paths_idx=paths_idx, ckpt_path=None,
                group_ok=nsga_ok, refresh=refresh,
                cache_data=cache_data, file_artifacts=file_artifacts,
                new_artifacts=new_artifacts,
                compute_fn=_compute_nsga,
                J=J, q_idx=args.q_idx, p_idx=args.p_idx)
            path_static_pts['NSGA'] = (ms, yld)
            md = compute_metrics(ms, yld, anchors)
            for k in ('HV', 'Makespan', 'Quality'):
                static_agg['NSGA'][k].append(md[k])
            static_agg['NSGA']['Time'].append(t)
            tag = ' [cache]' if hit else ''
            print(f"  NSGA          HV={md['HV']:.4f}  ms={md['Makespan']:.1f}  "
                  f"q={md['Quality']:.4f}  t={t:.2f}s{tag}")

        # Sweep 행 × epoch — jsonl 캐시 only. 점이 없으면 그 (행, epoch) 는 스킵.
        # (train_continual.py 가 매 epoch 적재해두지 않은 행은 그래프에서 자연 제외.)
        path_curve_pts = {r: {} for r in sweep_rows}
        for e in epoch_list:
            parts = [f"  epoch {e:>4d}"]
            for r in sweep_rows:
                pk = f'p{paths_idx}_{_SWEEP_SLUG[r]}_e{e}'
                if f'{pk}_ms' not in cache_data:
                    parts.append(f"{r}: -")
                    continue
                ms = cache_data[f'{pk}_ms']
                yld = cache_data[f'{pk}_yld']
                t = float(cache_data[f'{pk}_t'])
                path_curve_pts[r][e] = (ms, yld)
                md = compute_metrics(ms, yld, anchors)
                curve_agg[r][e]['HV'].append(md['HV'])
                curve_agg[r][e]['Makespan'].append(md['Makespan'])
                curve_agg[r][e]['Quality'].append(md['Quality'])
                curve_agg[r][e]['Time'].append(t)
                parts.append(f"{r}: HV={md['HV']:.3f}·")
            print('  '.join(parts))

        # 공통 best-known front (이 path 의 모든 점 합집합) → 행/에폭별 IGD+.
        union_pts = dict(path_static_pts)
        for r in sweep_rows:
            for e, (ms, yld) in path_curve_pts[r].items():
                union_pts[f'{r}@{e}'] = (ms, yld)
        ref_front = build_reference_front(union_pts, anchors)

        for r in sweep_rows:
            for e, (ms, yld) in path_curve_pts[r].items():
                igd = igd_plus_metric(ms, yld, ref_front, anchors)
                curve_agg[r][e]['IGD+'].append(igd)
        for name, (ms, yld) in path_static_pts.items():
            igd = igd_plus_metric(ms, yld, ref_front, anchors)
            static_agg[name]['IGD+'].append(igd)

    # 최종 meta 저장 (점 데이터는 sweep 도중 jsonl 에 라인 단위로 이미 append 완료).
    try:
        _save_continual_meta(J, args.q_idx, args.p_idx, cur_metas, new_artifacts)
        print(f"\nsaved -> {_continual_cache_path(J, args.q_idx, args.p_idx)}  (점 jsonl)")
        print(f"saved -> {_continual_meta_path(J, args.q_idx, args.p_idx)}  (meta)")
    except Exception as e:
        print(f"[warn] meta 저장 실패: {e}")

    return curve_agg, static_agg


# =====================================================
# Entry
# =====================================================
def _parse_idx_spec(s: str) -> list[int]:
    """'1' / '1~10' / '1~100:5' (stride) / '1,3,5' 지원."""
    s = s.strip()
    if '~' in s:
        rest, _, step = s.partition(':')
        lo, hi = rest.split('~')
        stride = int(step) if step else 1
        return list(range(int(lo), int(hi) + 1, stride))
    if ',' in s:
        return [int(x) for x in s.split(',')]
    return [int(s)]


def main():
    p = argparse.ArgumentParser(
        description="웨이퍼 지속학습 적응-효율 비교 표/그림 "
                    "(scratch / full-adapt / stale-feat / masked-feat / zero-shot / NSGA). "
                    "모든 방법을 타깃 (P1,Q1) 에서 동일 Q1 GT scorer 로 평가.")
    # 타깃 인스턴스 (기본 Q1, P1).
    p.add_argument('--num_jobs', type=int, default=25)
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7',
                   help="stage별 머신 수. base(5,3,7,3,5,7) 초과 시 (m%%base) 복제 증강.")
    p.add_argument('--q_idx', type=int, default=2, choices=[1, 2, 3],
                   help="타깃 품질 시나리오 (이어 학습할 새 환경).")
    p.add_argument('--p_idx', type=int, default=2, choices=[1, 2, 3],
                   help="타깃 proc_time 인스턴스.")
    p.add_argument('--paths_idx', type=str, default='10',
                   help="타깃 historical_paths 인덱스. '1' / '1~10' / '1,3,5'. 여러 개면 path 평균.")
    # stale 피처(옛 환경) 소스.
    p.add_argument('--src_q_idx', type=int, default=3, choices=[1, 2, 3],
                   help="stale-feat 행이 쓰는 옛(소스) 품질 시나리오. 기본 Q3.")
    p.add_argument('--src_paths_idx', type=int, default=1,
                   help="stale 피처용 소스 historical_paths 인덱스.")
    p.add_argument('--mask_value', type=float, default=-1.0,
                   help="masked-feat 행의 피처 상수. 기본 -1 (학습 시 -1 마스킹과 일치). "
                        "z-score 는 학습 시 ≥0 이라 -1 은 OOD sentinel; 0 으로 바꿔 in-dist 도 가능.")
    # 정책 체크포인트 — sweep(jsonl-only) 모드에서는 무시되고 snapshot 모드에서만 .pt 로 로드됨.
    p.add_argument('--ckpt_scratch', type=str, default='',
                   help="snapshot 모드 전용: (Q1p10) 처음부터 학습한 정책 .pt. (sweep 모드는 jsonl 만 읽음)")
    p.add_argument('--ckpt_adapt', type=str, default='',
                   help="snapshot 모드 전용: full-adapt 정책 .pt = 제안. (sweep 모드는 jsonl 만 읽음)")
    p.add_argument('--ckpt_stale', type=str, default='',
                   help="snapshot 모드 전용: stale-feat 정책 .pt. (sweep 모드는 jsonl 만 읽음)")
    p.add_argument('--ckpt_masked', type=str, default='',
                   help="snapshot 모드 전용: masked-feat 정책 .pt. (sweep 모드는 jsonl 만 읽음)")
    p.add_argument('--ckpt_src', type=str, default='checkpoints/0523_baseline.pt',
                   help="zero-shot: (P3,Q3) 사전학습 정책 (추가 학습 없이 평가; 학습곡선 t=0). "
                        "기본 0523_baseline.pt — 모든 continual ckpt 의 init 베이스. 빈값 → 제외.")
    # ── 학습 곡선(sweep) 모드 ──
    p.add_argument('--epoch_sweep', type=str, default='1~100',
                   help="비우면 단일 스냅샷 표 모드. 지정 시 학습 곡선 모드 (jsonl-only): "
                        "'1~100:5' (stride) 또는 '1,5,10,20'. 점은 train_continual.py 가 매 "
                        "epoch 적재한 continual_points_W{J}_Q{q}P{p}.jsonl 에서만 읽음.")
    # 평가 설정.
    p.add_argument('--yield_mode', type=str, default='raw', choices=['raw', 'percentile'])
    p.add_argument('--num_lambdas', type=int, default=32, help="모델 λ grid 크기.")
    p.add_argument('--samples', type=int, default=64,
                   help="모델 λ 당 sample 수 (>1=stochastic, 1=greedy).")
    p.add_argument('--no_nsga', default=True,
                   help="True 면 NSGA 참조 행을 건너뜀.")
    p.add_argument('--cache_refresh', default=False,
                   help="점 캐시 무시·재계산+덮어쓰기. 기본 False (캐시 히트 시 즉시 재사용해 플롯만 다시 그림). "
                        "ckpt mtime/size 변경은 자동 감지되니 보통은 안 켜도 됨.")
    p.add_argument('--nsga_pop', type=int, default=100)
    p.add_argument('--nsga_gen', type=int, default=300)
    p.add_argument('--xlim', type=str, default='40,180',
                   help="makespan anchor 'm_best,m_ref'. 타깃 인스턴스에 맞춰 조정.")
    p.add_argument('--ylim', type=str, default='0,1',
                   help="yield anchor 'q_ref,q_best'. percentile 모드는 0,100 으로 강제.")
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ga_seed', type=int, default=0)
    p.add_argument('--no_plot', default=False, help="True 면 overlay 플롯 저장 안 함.")
    args = p.parse_args()

    machines = [int(x) for x in args.machines.split(',')]
    J, S, M = args.num_jobs, len(machines), sum(machines)
    paths_idx_list = _parse_idx_spec(args.paths_idx)
    wq_min, wq_max = [float(x) for x in args.wafer_quality.split(',')]

    m_best, m_ref = [float(x) for x in args.xlim.split(',')]
    q_ref, q_best = [float(x) for x in args.ylim.split(',')]
    if args.yield_mode == 'percentile':
        q_ref, q_best = 0.0, 100.0
    anchors = (m_best, m_ref, q_best, q_ref)

    # 행별 ckpt 매핑 (빈값 → 그 행 제외). sweep 모드면 dir, 아니면 file.
    ckpt_map = {
        'scratch':     args.ckpt_scratch,
        'full-adapt':  args.ckpt_adapt,
        'stale-feat':  args.ckpt_stale,
        'masked-feat': args.ckpt_masked,
        'zero-shot':   args.ckpt_src,
    }
    run_nsga = not args.no_nsga

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f"[continual] device={device}  W={J}  machines={machines}  "
          f"타깃=(Q{args.q_idx},P{args.p_idx})  paths={paths_idx_list}  yield={args.yield_mode}")
    print(f"[continual] anchors: m=[best={m_best}, ref={m_ref}]  q=[ref={q_ref}, best={q_best}]")
    print(f"[continual] 모델: λ×{args.num_lambdas}, samples/λ={args.samples}  |  "
          f"NSGA-II: pop={args.nsga_pop}, gen={args.nsga_gen}  (run_nsga={run_nsga})")
    print(f"[continual] stale 피처 소스=(Q{args.src_q_idx}, paths_{args.src_paths_idx})  "
          f"mask_value={args.mask_value}")

    # ── 학습 곡선(sweep) 모드 분기 ──
    if args.epoch_sweep:
        epoch_list = _parse_idx_spec(args.epoch_sweep)
        # sweep 행 = canonical 4행. 데이터는 jsonl 캐시에서만 읽으므로 ckpt dir 불필요.
        # train_continual.py 가 적재하지 않은 행/에폭은 자동 스킵 (그래프에서 자연 제외).
        sweep_rows = ['scratch', 'full-adapt', 'stale-feat', 'masked-feat']
        print(f"[continual] sweep 모드 (jsonl-only): epochs={epoch_list[0]}~{epoch_list[-1]} "
              f"(n={len(epoch_list)})  행={sweep_rows}")

        curve_agg, static_agg = run_sweep(
            args=args, machines=machines, J=J, S=S, anchors=anchors,
            paths_idx_list=paths_idx_list, wq_min=wq_min, wq_max=wq_max,
            epoch_list=epoch_list, sweep_rows=sweep_rows, run_nsga=run_nsga,
            device=device)
        tag = f'W{J}_Q{args.q_idx}P{args.p_idx}'
        csv_path = f'test_results/continual_curve_{tag}.csv'
        png_path = f'test_results/continual_curve_{tag}.png'
        write_curve_csv(curve_agg, static_agg, epoch_list, sweep_rows, csv_path)
        print(f"saved -> {csv_path}")
        if not args.no_plot:
            title = (f"Continual learning curve → (Q{args.q_idx}, P{args.p_idx}), "
                     f"paths={paths_idx_list[0]}~{paths_idx_list[-1]} (n={len(paths_idx_list)})")
            plot_curves(curve_agg, static_agg, epoch_list, sweep_rows, png_path, title)
            print(f"saved -> {png_path}")

        # 적응 효율 요약: full-adapt 의 마지막 epoch HV vs zero-shot.
        if 'full-adapt' in curve_agg and static_agg.get('zero-shot', {}).get('HV'):
            last_e = epoch_list[-1]
            fa_vals = curve_agg['full-adapt'][last_e]['HV']
            if fa_vals:
                fa = float(np.nanmean(fa_vals))
                zs = float(np.nanmean(static_agg['zero-shot']['HV']))
                print(f"[continual] 적응 이득(HV @epoch={last_e}): full-adapt {fa:.4f} − "
                      f"zero-shot {zs:.4f} = {fa - zs:+.4f}")
        print("\n=== done (sweep) ===")
        return

    # ── 단일 스냅샷(표) 모드 ──
    policy_rows = [r for r in ALL_ROWS if r != 'NSGA' and ckpt_map[r]]
    if not policy_rows:
        raise SystemExit(
            "정책 ckpt 가 하나도 없습니다 — --ckpt_scratch/--ckpt_adapt/--ckpt_stale/"
            "--ckpt_masked/--ckpt_src 중 최소 하나를 지정하세요.")
    display_rows = policy_rows + (['NSGA'] if run_nsga else [])
    print(f"[continual] 행: {display_rows}")
    for r in policy_rows:
        print(f"             {r:11s} <- {ckpt_map[r]}  (피처={_ROW_FEAT[r]})")

    # 정책은 그래프 모델이라 env/인스턴스 무관 → 한 번만 로드해 재사용.
    tmp_env = HFSPGraphEnv(num_jobs=J, machine_cnt_list=machines, device=device)
    policies = {r: load_policy(ckpt_map[r], tmp_env, device) for r in policy_rows}

    # ── path 별 평가 → 집계 ──
    agg = {r: {'HV': [], 'IGD+': [], 'Makespan': [], 'Quality': [], 'Time': []}
           for r in display_rows}
    last_pts = None
    for paths_idx in paths_idx_list:
        print(f"\n========== paths_{paths_idx} ==========")
        out = evaluate_one_path(
            policies=policies, feat_src=_ROW_FEAT, device=device,
            J=J, S=S, machines=machines,
            q_idx=args.q_idx, src_q_idx=args.src_q_idx,
            paths_idx=paths_idx, src_paths_idx=args.src_paths_idx, p_idx=args.p_idx,
            anchors=anchors, seed=args.seed, ga_seed=args.ga_seed,
            wq_min=wq_min, wq_max=wq_max, mask_value=args.mask_value,
            num_lambdas=args.num_lambdas, samples=args.samples,
            nsga_pop=args.nsga_pop, nsga_gen=args.nsga_gen,
            yield_mode=args.yield_mode, run_nsga=run_nsga)

        present = [r for r in display_rows if r in out]
        # 1st pass: 방법별 독립 지표(HV/Makespan/Quality) + 점집합.
        last_pts = {r: (out[r][0], out[r][1]) for r in present}
        md = {r: compute_metrics(out[r][0], out[r][1], anchors) for r in present}
        # 2nd pass: 행 점을 합쳐 best-known front → 행별 IGD+ (공통 기준).
        ref_front = build_reference_front(last_pts, anchors)
        for r in present:
            ms, yld, t = out[r]
            igd = igd_plus_metric(ms, yld, ref_front, anchors)
            agg[r]['HV'].append(md[r]['HV'])
            agg[r]['IGD+'].append(igd)
            agg[r]['Makespan'].append(md[r]['Makespan'])
            agg[r]['Quality'].append(md[r]['Quality'])
            agg[r]['Time'].append(t)
            igd_str = '—' if np.isnan(igd) else f"{igd:.4f}"
            print(f"  {r:11s}  HV={md[r]['HV']:.4f}  IGD+={igd_str}  "
                  f"ms={md[r]['Makespan']:.1f}  q={md[r]['Quality']:.4f}  "
                  f"t={t:.2f}s  |pts|={np.asarray(ms).size}")

    # ── 표 출력 + 저장 ──
    header = (f"Continual (transfer) | N={J}, M={machines}, "
              f"타깃 (Q{args.q_idx},P{args.p_idx}), yield={args.yield_mode}")
    table_str, rows = build_table(agg, header, len(paths_idx_list), display_rows)
    print(table_str)

    # 적응 효율 요약: full-adapt 가 zero-shot 대비 얼마나 HV 를 끌어올렸나.
    if 'full-adapt' in agg and 'zero-shot' in agg and agg['full-adapt']['HV']:
        fa = float(np.nanmean(agg['full-adapt']['HV']))
        zs = float(np.nanmean(agg['zero-shot']['HV']))
        print(f"[continual] 적응 이득(HV): full-adapt {fa:.4f} − zero-shot {zs:.4f} "
              f"= {fa - zs:+.4f}")

    csv_path = f'test_results/continual_W{J}_Q{args.q_idx}P{args.p_idx}_table.csv'
    try:
        import pandas as pd
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"saved -> {csv_path}")
    except Exception as e:
        print(f"[warn] CSV 저장 실패: {e}")

    # ── overlay (단일 path 일 때만) ──
    if not args.no_plot and len(paths_idx_list) == 1 and last_pts is not None:
        png_path = f'test_results/continual_W{J}_Q{args.q_idx}P{args.p_idx}_p{paths_idx_list[0]}.png'
        plot_overlay(last_pts, anchors, png_path,
                     title=f"Continual transfer → (Q{args.q_idx}, P{args.p_idx}), paths_{paths_idx_list[0]}",
                     rows=[r for r in display_rows if r in last_pts])
        print(f"saved -> {png_path}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
