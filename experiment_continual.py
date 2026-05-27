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

from HFSPGraphEnv import HFSPGraphEnv
from FFSPModel import FFSPModel
from HFSPWrapper import make_env_edge_lookup, get_feat_dims
from quality_augment import GroundTruthQuality, load_proc_time_augmented

# test.py 의 모델/EST rollout 평가 머신 재사용.
from test import _run_single_experiment, compute_pareto_front
# nsga2.py 의 NSGA-II decoder/loop 재사용.
from nsga2 import HFSPDecoder, run_nsga2
# experiment_table.py 의 지표(HV/IGD+)·타이밍·포맷 그대로 재사용 → 표 수치 1:1 일치.
from experiment_table import (
    compute_metrics,
    build_reference_front,
    igd_plus_metric,
    timed,
    _fmt,
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

# 그림용 색/마커.
_ROW_STYLE = {
    'scratch':     dict(color='tab:gray',   marker='o'),
    'full-adapt':  dict(color='crimson',    marker='s'),
    'stale-feat':  dict(color='tab:orange', marker='^'),
    'masked-feat': dict(color='tab:purple', marker='v'),
    'zero-shot':   dict(color='tab:brown',  marker='D'),
    'NSGA':        dict(color='tab:blue',   marker='P'),
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


def main():
    p = argparse.ArgumentParser(
        description="웨이퍼 지속학습 적응-효율 비교 표/그림 "
                    "(scratch / full-adapt / stale-feat / masked-feat / zero-shot / NSGA). "
                    "모든 방법을 타깃 (P1,Q1) 에서 동일 Q1 GT scorer 로 평가.")
    # 타깃 인스턴스 (기본 Q1, P1).
    p.add_argument('--num_jobs', type=int, default=25)
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7',
                   help="stage별 머신 수. base(5,3,7,3,5,7) 초과 시 (m%%base) 복제 증강.")
    p.add_argument('--q_idx', type=int, default=1, choices=[1, 2, 3],
                   help="타깃 품질 시나리오 (이어 학습할 새 환경).")
    p.add_argument('--p_idx', type=int, default=1, choices=[1, 2, 3],
                   help="타깃 proc_time 인스턴스.")
    p.add_argument('--paths_idx', type=str, default='1',
                   help="타깃 historical_paths 인덱스. '1' / '1~10' / '1,3,5'. 여러 개면 path 평균.")
    # stale 피처(옛 환경) 소스.
    p.add_argument('--src_q_idx', type=int, default=3, choices=[1, 2, 3],
                   help="stale-feat 행이 쓰는 옛(소스) 품질 시나리오. 기본 Q3.")
    p.add_argument('--src_paths_idx', type=int, default=1,
                   help="stale 피처용 소스 historical_paths 인덱스.")
    p.add_argument('--mask_value', type=float, default=-1.0,
                   help="masked-feat 행의 피처 상수. 기본 -1 (학습 시 -1 마스킹과 일치). "
                        "z-score 는 학습 시 ≥0 이라 -1 은 OOD sentinel; 0 으로 바꿔 in-dist 도 가능.")
    # 정책 체크포인트 (빈값이면 그 행은 건너뜀 — 가진 것만 비교 가능).
    p.add_argument('--ckpt_scratch', type=str, default='',
                   help="(P1,Q1) 처음부터 학습한 정책.")
    p.add_argument('--ckpt_adapt', type=str, default='',
                   help="full-adapt: (P3,Q3)→(P1,Q1) 이어학습(피처·보상 모두 Q1) 정책 = 제안.")
    p.add_argument('--ckpt_stale', type=str, default='',
                   help="stale-feat: 보상만 Q1·피처는 Q3 유지로 이어학습한 정책.")
    p.add_argument('--ckpt_masked', type=str, default='',
                   help="masked-feat: 보상만 Q1·피처는 -1 마스킹으로 이어학습한 정책.")
    p.add_argument('--ckpt_src', type=str, default='',
                   help="zero-shot: (P3,Q3) 사전학습 정책 (추가 학습 없이 평가; 학습곡선 t=0).")
    # 평가 설정.
    p.add_argument('--yield_mode', type=str, default='raw', choices=['raw', 'percentile'])
    p.add_argument('--num_lambdas', type=int, default=32, help="모델 λ grid 크기.")
    p.add_argument('--samples', type=int, default=64,
                   help="모델 λ 당 sample 수 (>1=stochastic, 1=greedy).")
    p.add_argument('--no_nsga', default=False,
                   help="True 면 NSGA 참조 행을 건너뜀.")
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

    # 행별 ckpt 매핑 (빈값 → 그 행 제외).
    ckpt_map = {
        'scratch':     args.ckpt_scratch,
        'full-adapt':  args.ckpt_adapt,
        'stale-feat':  args.ckpt_stale,
        'masked-feat': args.ckpt_masked,
        'zero-shot':   args.ckpt_src,
    }
    policy_rows = [r for r in ALL_ROWS if r != 'NSGA' and ckpt_map[r]]
    if not policy_rows:
        raise SystemExit(
            "정책 ckpt 가 하나도 없습니다 — --ckpt_scratch/--ckpt_adapt/--ckpt_stale/"
            "--ckpt_masked/--ckpt_src 중 최소 하나를 지정하세요.")
    run_nsga = not args.no_nsga
    display_rows = policy_rows + (['NSGA'] if run_nsga else [])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f"[continual] device={device}  W={J}  machines={machines}  "
          f"타깃=(Q{args.q_idx},P{args.p_idx})  paths={paths_idx_list}  yield={args.yield_mode}")
    print(f"[continual] anchors: m=[best={m_best}, ref={m_ref}]  q=[ref={q_ref}, best={q_best}]")
    print(f"[continual] 모델: λ×{args.num_lambdas}, samples/λ={args.samples}  |  "
          f"NSGA-II: pop={args.nsga_pop}, gen={args.nsga_gen}  (run_nsga={run_nsga})")
    print(f"[continual] 행: {display_rows}")
    print(f"[continual] stale 피처 소스=(Q{args.src_q_idx}, paths_{args.src_paths_idx})  "
          f"mask_value={args.mask_value}")
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
