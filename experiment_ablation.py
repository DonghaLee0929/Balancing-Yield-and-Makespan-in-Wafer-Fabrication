"""
experiment_ablation.py — Ablation(부정확 보상) vs Baseline(정상 학습) 의 epoch × HVN 비교.

스토리:
  · Ablation   : 부정확 보상(paths_2) 로 학습, paths_1 GT 로 평가 — train_ablation.py 산물.
  · Baseline   : 보상·평가 모두 paths_1 GT 인 *정상* 학습 — train_ASIL.py 산물(과거 실행 로그).
  두 곡선을 같은 epoch 축에 겹쳐 부정확 보상이 진짜 yield Pareto 를 얼마나 못 끌어올리는지
  (혹은 의외로 끌어올리는지) 한눈에 본다.

입력:
  · ablation JSONL : 각 줄 = 한 epoch, `hv_norm` (B,) → batch 평균을 epoch HVN 으로.
  · baseline LOG   : train_ASIL 콘솔 로그. `eval <epoch>  hv=... hvN=<mean>±<std> ...`
                     꼴 줄을 정규식으로 모두 뽑아 (epoch, mean, std) 시리즈로.

시각화: experiment_continual.plot_curves 의 양식 채용 — 중앙형 이동평균 곡선 +
       표준편차 음영 band, 상단 bold suptitle. 패널은 HV 1 개만 ("오직 HV").
       baseline 은 20 epoch 간격 sparse 점이라 marker+line 으로 별도 표시 (smoothing X).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

# Windows 콘솔(cp949) 한글/유니코드 안전.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# experiment_continual.py 와 동일 폰트 (STIX serif, ↑↓ 글리프 + bold 지원).
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

# experiment_continual._PLOT_* 와 동일 스타일 컨벤션.
_PLOT_SMOOTH_WINDOW = 7
_PLOT_LINE_WIDTH = 2.0
_PLOT_BAND_ALPHA = 0.18
_PLOT_LEGEND_FONTSIZE = 12
_PLOT_SUPTITLE_FONTSIZE = 16
_PLOT_FIGSIZE = (8.5, 5.2)            # 1-panel + legend 공간 (continual 은 3-panel 15.0)
_ABLATION_COLOR = 'crimson'           # continual 의 full-adapt 와 동일 (ablation 주역 톤)
_BASELINE_COLOR = 'tab:blue'          # continual 의 NSGA 색 — 참조/정상 baseline 톤

DEFAULT_LOG  = "train_results/ablation_p2train_p1eval_pareto_points.jsonl"
DEFAULT_BASE = "train_results/output.log"
DEFAULT_OUT  = "train_results/ablation_p2train_p1eval_hvn_curve.png"

# eval 줄 파서. ± 가 mojibake(`Â±`) 일 수도 있고 그냥 ± 일 수도 → [^0-9]+ 로 흡수.
# 예: "eval   20  hv=28.259±10.740 hvN=0.507±0.037 ms@0=163.0 ..."
_EVAL_RE = re.compile(
    r'^eval\s+(\d+)\s+.*?hvN=(\d+\.\d+)[^0-9]+(\d+\.\d+)',
    re.IGNORECASE,
)


def _smooth(a, window):
    """중앙형 이동평균. (experiment_continual._smooth 와 동일.)"""
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
    """중앙형 rolling std — batch 가 단일 점일 때 noise envelope fallback."""
    a = np.asarray(a, dtype=np.float64)
    if window <= 1 or a.size <= 1:
        return np.zeros_like(a)
    half = window // 2
    out = np.zeros_like(a)
    for i in range(a.size):
        lo, hi = max(0, i - half), min(a.size, i + half + 1)
        out[i] = np.nanstd(a[lo:hi]) if hi - lo > 1 else 0.0
    return out


def load_jsonl(path):
    """JSONL → (epochs, hv_norm_mean, hv_norm_std). batch 축(B,) 을 mean/std 로 축약.

    포맷 (train_ablation.py 산물):
      첫 줄  = {"meta": {...}}  — skip (key 'epoch' 없음으로 식별)
      이후 줄 = {"epoch": e, "makespan":[L,K], "quality":[L,K], "hvn":[K], ...}
                  hvn 키 사용 (구버전 'hv_norm' 도 fallback 으로 허용).
    """
    epochs, mean, std = [], [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if 'epoch' not in rec:  # meta / 기타 헤더 줄 skip
                continue
            hvn_raw = rec.get('hvn', rec.get('hv_norm'))
            if hvn_raw is None:
                continue
            hvn = np.array(hvn_raw, dtype=np.float64)
            epochs.append(int(rec['epoch']))
            mean.append(float(np.nanmean(hvn)))
            std.append(float(np.nanstd(hvn)) if hvn.size > 1 else 0.0)
    return (np.array(epochs, dtype=np.float64),
            np.array(mean,   dtype=np.float64),
            np.array(std,    dtype=np.float64))


def load_baseline_log(path, max_epoch=None):
    """train_ASIL 콘솔 로그에서 `eval` 줄을 모두 뽑아 (epochs, hvN_mean, hvN_std).

    파일이 mojibake(`Â±`) 인코딩이어도 무관 — 정규식이 숫자 사이의 ± 영역을 [^0-9]+ 로 흡수.
    `errors='replace'` 로 깨진 바이트가 있어도 줄 자체는 그대로 매칭됨.

    max_epoch : 지정 시 epoch <= max_epoch 인 점만 반환. baseline 이 ablation 보다 더 오래
                돌아간 경우 동일 epoch 축으로 자르기 위함.
    """
    epochs, mean, std = [], [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = _EVAL_RE.match(line.strip())
            if not m:
                continue
            e = int(m.group(1))
            if max_epoch is not None and e > max_epoch:
                continue
            epochs.append(e)
            mean.append(float(m.group(2)))
            std.append(float(m.group(3)))
    return (np.array(epochs, dtype=np.float64),
            np.array(mean,   dtype=np.float64),
            np.array(std,    dtype=np.float64))


def plot_hvn(ab_epochs, ab_mean, ab_std,
             bl_epochs, bl_mean, bl_std,
             save_path, title):
    """epoch × HVN 단일 패널. continual.plot_curves 의 HV 패널 양식.

    ablation : dense (매 epoch) → smoothing + std band.
    baseline : sparse (eval_interval=20) → smoothing 없이 marker+line + std band.
    """
    fig, ax = plt.subplots(figsize=_PLOT_FIGSIZE)

    # --- ablation: 부정확 보상 학습 ---
    ab_smooth = _smooth(ab_mean, _PLOT_SMOOTH_WINDOW)
    if np.any(ab_std > 0):
        ab_band = _smooth(ab_std, _PLOT_SMOOTH_WINDOW)
    else:
        ab_band = _rolling_std(ab_mean, _PLOT_SMOOTH_WINDOW)
    ax.plot(ab_epochs, ab_smooth, '-', color=_ABLATION_COLOR,
            lw=_PLOT_LINE_WIDTH, label='Inaccurate prediction model')
    if np.any(ab_band > 0):
        ax.fill_between(ab_epochs, ab_smooth - ab_band, ab_smooth + ab_band,
                        color=_ABLATION_COLOR, alpha=_PLOT_BAND_ALPHA, linewidth=0)

    # --- baseline: 정상 학습 (eval 20epoch 마다) ---
    if bl_epochs.size > 0:
        ax.plot(bl_epochs, bl_mean, '-', color=_BASELINE_COLOR,
                lw=_PLOT_LINE_WIDTH,
                label='Baseline')
        if np.any(bl_std > 0):
            ax.fill_between(bl_epochs, bl_mean - bl_std, bl_mean + bl_std,
                            color=_BASELINE_COLOR, alpha=_PLOT_BAND_ALPHA, linewidth=0)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('HV ↑')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=_PLOT_LEGEND_FONTSIZE, frameon=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.suptitle(title, fontsize=_PLOT_SUPTITLE_FONTSIZE, fontweight='bold', y=0.97)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log',      type=str, default=DEFAULT_LOG,
                   help='ablation JSONL (train_ablation.py 산물)')
    p.add_argument('--baseline', type=str, default=DEFAULT_BASE,
                   help='정상 학습 콘솔 로그 (train_ASIL.py 의 eval 줄 파싱)')
    p.add_argument('--out',      type=str, default=DEFAULT_OUT)
    p.add_argument('--title',    type=str,
                   default='Learning curve comparison for ablation study')
    args = p.parse_args()

    ab_e, ab_m, ab_s = load_jsonl(args.log)
    if ab_e.size == 0:
        raise SystemExit(f"[ablation-plot] empty log: {args.log}")
    print(f"[ablation-plot] ablation: {ab_e.size} epochs from {args.log}")
    print(f"  HVN range mean=[{ab_m.min():.4f}, {ab_m.max():.4f}] "
          f"first={ab_m[0]:.4f} last={ab_m[-1]:.4f}")

    if os.path.exists(args.baseline):
        # baseline 이 ablation 보다 더 길어도 같은 epoch 축으로 자른다.
        bl_e, bl_m, bl_s = load_baseline_log(args.baseline,
                                             max_epoch=int(ab_e.max()))
        print(f"[ablation-plot] baseline: {bl_e.size} eval points from {args.baseline} "
              f"(epoch <= {int(ab_e.max())})")
        if bl_e.size > 0:
            print(f"  HVN range mean=[{bl_m.min():.4f}, {bl_m.max():.4f}] "
                  f"first={bl_m[0]:.4f} last={bl_m[-1]:.4f}")
    else:
        print(f"[ablation-plot] baseline log not found ({args.baseline}) — skipping overlay")
        bl_e = bl_m = bl_s = np.array([], dtype=np.float64)

    plot_hvn(ab_e, ab_m, ab_s, bl_e, bl_m, bl_s, args.out, args.title)
    print(f"[ablation-plot] saved -> {args.out}")


if __name__ == "__main__":
    main()
