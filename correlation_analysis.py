"""
λ-sweep 결과로 상관관계 히트맵 2종을 그린다 (슬라이드 재현, 2-objective 버전).

이 프로젝트는 목적함수가 2개(makespan↓, yield↑)이고 선호 가중치가 스칼라 λ 하나라서,
슬라이드의 3×3 대신 2×2 행렬이 나온다. λ-sweep 데이터는 test.py 의 evaluate_pareto 를
single_view='all' 로 호출해 그대로 얻는다 (각 (λ, sample) 의 makespan/yield 점 전부).

  1) Correlation Matrix: Objective Values
       corr among [makespan, yield]                    → (2×2)
  2) Weight-Value Cross Correlation
       corr( [w_makespan=1−λ, w_yield=λ], [makespan, yield] )  → (2×2)
       ⚠ w_makespan = 1 − w_yield 이므로 두 행은 부호만 반대 (구조상 필연).

makespan/yield 의미·계산은 test.py 와 100% 동일 (같은 함수를 그대로 호출).
"""

from __future__ import annotations
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')                 # 반드시 test 임포트(=pyplot 로드) 보다 먼저
import matplotlib.pyplot as plt

from test import evaluate_pareto       # λ-sweep 머신러리 그대로 재사용


# =====================================================
# 데이터 수집 — test.py 의 λ-sweep 을 그대로 호출
# =====================================================
def collect_lambda_sweep(*, ckpt, p_idx, q_idx, paths_idx, num_jobs, machines,
                         num_lambdas, samples, wq_min, wq_max, yield_mode,
                         sweep_png):
    """evaluate_pareto(single_view='all') → 각 (λ, sample) 의 (makespan, yield).

    반환: lam (total,), makespan (total,), yield (total,)  — 모두 같은 순서.
      total = num_lambdas * samples,  순서는 λ repeat_interleave(samples) 와 동일.
    """
    ms, yld, _front = evaluate_pareto(
        ckpt_path=ckpt, p_idx=p_idx, q_idx=q_idx, paths_idx_list=[paths_idx],
        num_jobs=num_jobs, machine_cnt_list=machines,
        num_lambdas=num_lambdas, samples=samples,
        wq_min=wq_min, wq_max=wq_max, save_path=sweep_png,
        method='model', yield_mode=yield_mode, single_view='all')
    # evaluate_pareto 내부: lam_all = linspace(0,1,N).repeat_interleave(samples)
    lam = np.repeat(np.linspace(0.0, 1.0, num_lambdas), samples)
    return lam, np.asarray(ms, dtype=np.float64), np.asarray(yld, dtype=np.float64)


# =====================================================
# 상관계수 계산
# =====================================================
def cross_correlation(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """W (N, nw), V (N, nv) → M (nw, nv), M[i,j] = corr(W[:,i], V[:,j])."""
    nw, nv = W.shape[1], V.shape[1]
    M = np.empty((nw, nv), dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        for i in range(nw):
            for j in range(nv):
                M[i, j] = np.corrcoef(W[:, i], V[:, j])[0, 1]
    return M


# =====================================================
# 히트맵 (슬라이드 스타일: 발산형 컬러맵, 값 주석, colorbar)
# =====================================================
def plot_heatmap(fig, ax, mat, row_labels, col_labels, title):
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = '—' if np.isnan(v) else f'{v:.3f}'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if (not np.isnan(v) and abs(v) > 0.5) else 'black',
                    fontsize=12)
    ax.set_title(title, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation')
    return im


def main():
    p = argparse.ArgumentParser(description="λ-sweep 상관관계 히트맵 (2-objective)")
    # test.py CLI 기본값과 1:1 동일 → 인자 없이 돌리면 test.py 와 같은 인스턴스.
    p.add_argument('--ckpt',        type=str, default='checkpoints/0528_baseline.pt')
    p.add_argument('--num_jobs',    type=int, default=25)
    p.add_argument('--machines',    type=str, default='5,3,7,3,5,7')
    p.add_argument('--p_idx',       type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--q_idx',       type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--paths_idx',   type=int, default=1)
    p.add_argument('--num_lambdas', type=int, default=64)
    p.add_argument('--samples',     type=int, default=64,
                   help='λ 당 trajectory 수. 1=greedy(점 num_lambdas개), >1=stochastic cloud.')
    p.add_argument('--yield_mode',  type=str, default='raw', choices=['raw', 'percentile'])
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00')
    args = p.parse_args()

    machines = [int(x) for x in args.machines.split(',')]
    wq_min, wq_max = [float(x) for x in args.wafer_quality.split(',')]

    tag = f'J{args.num_jobs}_Q{args.q_idx}_P{args.p_idx}_p{args.paths_idx}_s{args.samples}'
    sweep_png = f'test_results/corr_sweep_{tag}.png'   # evaluate_pareto 가 부가로 저장하는 Pareto 그림

    lam, ms, yld = collect_lambda_sweep(
        ckpt=args.ckpt, p_idx=args.p_idx, q_idx=args.q_idx, paths_idx=args.paths_idx,
        num_jobs=args.num_jobs, machines=machines,
        num_lambdas=args.num_lambdas, samples=args.samples,
        wq_min=wq_min, wq_max=wq_max, yield_mode=args.yield_mode, sweep_png=sweep_png)

    # ── 목적함수 값 행렬 (makespan, yield) ──
    values = np.stack([ms, yld], axis=1)                       # (N, 2)
    obj_corr = np.corrcoef(values, rowvar=False)               # (2, 2)

    # ── weight-value 교차 상관: weights = [1−λ, λ] ──
    weights = np.stack([1.0 - lam, lam], axis=1)               # (N, 2)
    cross_corr = cross_correlation(weights, values)            # (2, 2)

    # ── 콘솔 출력 ──
    print(f"\n[corr] N points = {len(ms)}  "
          f"makespan[{ms.min():.1f},{ms.max():.1f}]  yield[{yld.min():.4f},{yld.max():.4f}]")
    print(f"[corr] corr(makespan, yield) = {obj_corr[0, 1]:.4f}")
    print(f"[corr] corr(λ, makespan)     = {np.corrcoef(lam, ms)[0, 1]:.4f}")
    print(f"[corr] corr(λ, yield)        = {np.corrcoef(lam, yld)[0, 1]:.4f}")

    # ── 그림 ──
    obj_labels = ['Makespan', 'Yield']
    w_labels   = ['Makespan w. (1−λ)', 'Yield w. (λ)']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    plot_heatmap(fig, ax1, obj_corr, obj_labels, obj_labels,
                 'Correlation Matrix: Objective Values')
    plot_heatmap(fig, ax2, cross_corr, w_labels, obj_labels,
                 'Weight-Value Cross Correlation')
    ax2.set_xlabel('Objective Value')
    ax2.set_ylabel('Objective Weight')
    fig.suptitle(f'λ-sweep correlations  W={args.num_jobs}, (Q{args.q_idx},P{args.p_idx}), '
                 f'paths_{args.paths_idx}, λ×{args.num_lambdas}, s×{args.samples}',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = f'test_results/correlation_{tag}.png'
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ── 원자료 CSV (재플롯·검증용) ──
    out_csv = f'test_results/correlation_data_{tag}.csv'
    np.savetxt(out_csv, np.stack([lam, ms, yld], axis=1),
               delimiter=',', header='lambda,makespan,yield', comments='')
    print(f"\nsaved -> {out_png}")
    print(f"saved -> {out_csv}")
    print(f"saved -> {sweep_png}  (evaluate_pareto Pareto plot)")


if __name__ == "__main__":
    main()
