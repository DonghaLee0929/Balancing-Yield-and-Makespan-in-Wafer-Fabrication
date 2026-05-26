"""
experiment_shift.py — 품질 시나리오 변화(shift)에 대한 모델 성능 비교 표.

고정 인스턴스 (기본 Q3, P3, paths_1, W=25, machines=5,3,7,3,5,7) 에서, 가장 중요한
*마지막 stage* 의 품질 인자를 세 시나리오로 '변형(shift)' 시킨 뒤, 네 방법의 Pareto
front 지표(HV/IGD+/Makespan/Quality/Time)를 시나리오마다 한 표로 비교한다.

시나리오 (모두 마지막 stage 만 변형):
  1. 최고 품질 기계의 품질 인자를 그 stage 최저 기계보다 0.01 작게 (best → min - 0.01).
  2. 마지막 stage 품질 인자의 순위를 거꾸로 (값 집합은 그대로, rank reverse).
  3. 마지막 stage 최고 품질 기계 1개 제거 (그 stage 기계 수 7→6; env·proc_time·quality 모두 drop).

비교 방법 (각 시나리오마다 한 표, 위 이미지의 4행):
  pred=base   : base 정책 + base 예측모델(best_hb_model.zip). shift 를 모르는 모델 —
                예측모델이 *옛* 품질 landscape 를 보지만 채점은 shifted ground truth.
  pred=shift  : base 정책 + shift 에 맞춘 예측모델(--pred_s{n} 의 .zip; 기본값=base 예측모델).
                예측모델 재학습은 추후 추가 — 지금은 시나리오별 모델 경로만 받는다.
  gt=shift    : shift 로 처음부터 학습한 정책(--ckpt_gt_s{n}) + shifted ground truth oracle.
  NSGA (GT)   : shifted ground truth 로 평가하는 NSGA-II.

핵심 설계 — `quality_helper`(정책이 *보는* 품질 신호) 와 `scorer`(*실제* 채점) 를 분리한다:
    pred=base/shift : quality_helper = 예측모델(QualityHelper),  scorer = shifted GT.
    gt=shift        : quality_helper = scorer = shifted GT (oracle).
    NSGA            : scorer = shifted GT.
모든 방법이 동일 인스턴스(같은 proc_time·wafer_quality·shifted GT scorer·HV anchor)를 보므로
'예측 모델이 shift 를 얼마나 따라잡는가' 를 공정 비교할 수 있다. test.py / nsga2.py /
experiment_table.py 의 평가 코드를 그대로 import 해 수치가 그 스크립트들과 1:1 일치한다.
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

from HFSPGraphEnv import HFSPGraphEnv
from FFSPModel import FFSPModel
from HFSPWrapper import QualityHelper, make_env_edge_lookup, get_feat_dims
from quality_augment import GroundTruthQuality, load_proc_time_augmented

# test.py 의 모델/EST rollout 평가 머신 재사용.
from test import _run_single_experiment
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


# 표의 4행 (위 이미지 순서). gt=shift 는 노란 강조 행.
ROWS = ['pred=base', 'pred=shift', 'gt=shift', 'NSGA (GT)']
HIGHLIGHT_ROW = 'gt=shift'
SCENARIO_DESC = {
    1: '최고품질 기계 → (그 stage 최저 - 0.01)',
    2: '마지막 stage 품질 순위 reverse',
    3: '마지막 stage 최고품질 기계 1개 제거 (기계 수 -1)',
}


# =====================================================
# 시나리오 적용 — shifted ground-truth quality + proc_time 구성
# =====================================================
def _recompute_zq_local(gq: GroundTruthQuality) -> None:
    """gq.step_q 가 바뀐 뒤 stage-내 z-score(zq_local_t) 와 lazy 캐시를 다시 만든다.

    GroundTruthQuality.__init__ 의 z-score 로직과 동일 (unbiased std, min→0 shift).
    """
    S = gq.num_stages
    max_t = gq.step_q.shape[1]
    zq = np.zeros((S, max_t), dtype=np.float64)
    for s in range(S):
        cnt = gq.machine_cnt_list[s]
        qs = gq.step_q[s, :cnt]
        if cnt > 1 and np.ptp(qs) > 0:
            mu, std = qs.mean(), qs.std(ddof=1)
            z = (qs - mu) / max(std, 1e-8) * gq.z_scale
            z = z - z.min()
        else:
            z = np.zeros(cnt, dtype=np.float64)
        zq[s, :cnt] = z
    gq.zq_local_t = torch.tensor(zq, dtype=torch.float32, device=gq.device)
    gq.step_q_t = torch.tensor(gq.step_q, dtype=torch.float32, device=gq.device)
    gq._zq_global = None              # compute_machine_quality 캐시 무효화
    gq._all_products_sorted = None    # percentile 채점 캐시 무효화


def apply_scenario(gq: GroundTruthQuality, scenario: int) -> int | None:
    """gq 의 마지막 stage step_q 를 시나리오대로 in-place 변형.

    1: 최고품질 기계를 (그 stage 최저 - 0.01) 로,  2: 순위 reverse(값 집합 유지),
    3: 최고품질 기계 1개 제거 → machine_cnt_list[last] 1 감소.
    Returns: 제거한 기계 local index (scenario 3) 또는 None (1·2).
    """
    last = gq.num_stages - 1
    cnt = gq.machine_cnt_list[last]
    q = gq.step_q[last, :cnt].astype(np.float64).copy()
    removed = None

    if scenario == 1:
        b = int(np.argmax(q))
        q[b] = q.min() - 0.01                       # 최고 → 최저보다 0.01 작게
        gq.step_q[last, :cnt] = q
    elif scenario == 2:
        order = np.argsort(q)                       # 오름차순 인덱스 (order[0]=최저 기계)
        q[order] = np.sort(q)[::-1]                 # 최저 기계에 최고값 … 순위 거꾸로
        gq.step_q[last, :cnt] = q
    elif scenario == 3:
        b = int(np.argmax(q))
        removed = b
        q_red = np.delete(q, b)                     # 최고품질 기계 제거 → cnt-1 개
        gq.machine_cnt_list[last] = cnt - 1
        gq.step_q[last, :] = 0.0
        gq.step_q[last, :cnt - 1] = q_red
    else:
        raise ValueError(f"scenario must be 1/2/3, got {scenario}")

    _recompute_zq_local(gq)
    return removed


def build_shifted_instance(scenario, base_csv, p_csv, num_jobs, base_machines,
                           device, wq_min, wq_max):
    """시나리오별 (shifted GT scorer, proc_time, 유효 machine_cnt_list) 구성.

    scorer(=GroundTruthQuality) 는 base CSV 로 만든 뒤 마지막 stage 를 시나리오대로 변형.
    proc_time 은 base machines 로 로드 후, scenario 3 이면 제거된 기계의 열을 drop 해 정렬 유지.
    """
    S = len(base_machines)
    gq = GroundTruthQuality(num_stages=S, device=device, csv_path=base_csv,
                            machine_cnt_list=list(base_machines),
                            wafer_quality_min=wq_min, wafer_quality_max=wq_max)
    removed = apply_scenario(gq, scenario)
    machines_eff = list(gq.machine_cnt_list)

    proc = load_proc_time_augmented(p_csv, num_jobs, base_machines, device)
    if scenario == 3 and removed is not None:
        last = S - 1
        keep = [m for m in range(base_machines[last]) if m != removed]
        proc[last] = proc[last][:, :, keep].contiguous()      # (1, J, cnt-1), q_red 와 동일 순서
    return gq, proc, machines_eff


# =====================================================
# 모델·예측모델 로딩
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


def make_quality_helper(zip_path, num_stages, device):
    """예측모델 .zip + 같은 폴더의 wafer_quality.json 으로 QualityHelper 로드."""
    wafer_path = os.path.join(os.path.dirname(zip_path), 'wafer_quality.json')
    qh = QualityHelper(num_stages=num_stages, device=device,
                       model_path=zip_path, wafer_path=wafer_path)
    if not qh.is_active:
        raise FileNotFoundError(
            f"[shift] 예측모델 로드 실패 — {zip_path} / {wafer_path} 확인")
    return qh


# =====================================================
# 방법별 front 산출 (모델 λ-sweep / NSGA)
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


def nsga_front(env, scorer, proc, wq_1, machines_eff, J, S, anchors,
               nsga_pop, nsga_gen, ga_seed, yield_mode):
    """shifted GT scorer 기반 NSGA-II → (ms, gt_yield). experiment_table.py 와 동일 호출."""
    env_edge_lookup_np = env.env_edge_lookup.cpu().numpy()
    machine_offset_np = np.cumsum([0] + machines_eff[:-1]).astype(np.int64)
    wq_np_j = wq_1[0].cpu().numpy().astype(np.float64)
    decoder = HFSPDecoder(env, proc, env_edge_lookup_np, machine_offset_np,
                          scorer, wq_np_j, yield_mode=yield_mode,
                          yield_objective='ground_truth', quality_helper=None)
    _, _, ms_n, gt_n, _, _ = run_nsga2(
        decoder, J, S, machines_eff, pop_size=nsga_pop, n_gen=nsga_gen,
        seed=ga_seed, hv_anchors=anchors)
    return ms_n.astype(np.float32), gt_n.astype(np.float32)


# =====================================================
# 한 path 인스턴스에 대해 4행 모두 실행
# =====================================================
def evaluate_one_path(*, scenario, base_policy, device, J, S, base_machines,
                      base_csv, p_csv, anchors, seed, ga_seed, wq_min, wq_max,
                      num_lambdas, samples, nsga_pop, nsga_gen, yield_mode,
                      pred_base_zip, pred_shift_zip, gt_ckpt):
    """scenario 의 한 path → {row: (ms, yld, time_s)} dict."""
    gq, proc, machines_eff = build_shifted_instance(
        scenario, base_csv, p_csv, J, base_machines, device, wq_min, wq_max)

    env = HFSPGraphEnv(num_jobs=J, machine_cnt_list=machines_eff, device=device)
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)

    # 모든 방법이 동일 wafer_quality 를 보도록 한 번만 샘플 (shifted GT 에서).
    wq_1 = gq.sample_wafer_quality(B=1, num_jobs=J, seed=seed, device=device)   # (1, J)

    qh_base = make_quality_helper(pred_base_zip, S, device)        # 옛 품질 landscape
    qh_shift = make_quality_helper(pred_shift_zip, S, device)      # shift 에 맞춘 (기본=base)
    gt_policy = load_policy(gt_ckpt, env, device)                  # shift 로 학습한 정책

    out = {}

    # pred=base : base 정책 + base 예측모델, shifted GT 채점.
    (ms, yld), t = timed(lambda: model_front(
        base_policy, env, env_edge_lookup_t, device, qh_base, gq, proc, wq_1,
        num_lambdas, samples, seed, yield_mode), device)
    out['pred=base'] = (ms, yld, t)

    # pred=shift : base 정책 + shift 예측모델, shifted GT 채점.
    (ms, yld), t = timed(lambda: model_front(
        base_policy, env, env_edge_lookup_t, device, qh_shift, gq, proc, wq_1,
        num_lambdas, samples, seed, yield_mode), device)
    out['pred=shift'] = (ms, yld, t)

    # gt=shift : shift 학습 정책 + shifted GT oracle (quality_helper = scorer = gq).
    (ms, yld), t = timed(lambda: model_front(
        gt_policy, env, env_edge_lookup_t, device, gq, gq, proc, wq_1,
        num_lambdas, samples, seed, yield_mode), device)
    out['gt=shift'] = (ms, yld, t)

    # NSGA (GT) : shifted GT scorer.
    (ms, yld), t = timed(lambda: nsga_front(
        env, gq, proc, wq_1, machines_eff, J, S, anchors,
        nsga_pop, nsga_gen, ga_seed, yield_mode), device)
    out['NSGA (GT)'] = (ms, yld, t)

    return out, machines_eff


# =====================================================
# 표 렌더링
# =====================================================
def build_table(agg, header, n_paths):
    """agg[row][metric] = list(per-path values) → 출력 문자열 + rows(list of dict)."""
    cols = ['Method', 'HV ↑', 'IGD+ ↓', 'Makespan ↓', 'Quality ↑', 'Time (s)']
    decimals = {'HV ↑': 4, 'IGD+ ↓': 4, 'Makespan ↓': 1, 'Quality ↑': 4, 'Time (s)': 2}
    key_map = {'HV ↑': 'HV', 'IGD+ ↓': 'IGD+', 'Makespan ↓': 'Makespan',
               'Quality ↑': 'Quality', 'Time (s)': 'Time'}

    rows = []
    for r in ROWS:
        label = r + (' *' if r == HIGHLIGHT_ROW else '')
        row = {'Method': label}
        for c in cols[1:]:
            row[c] = _fmt(agg[r][key_map[c]], decimals[c])
        rows.append(row)

    try:
        import pandas as pd
        body = pd.DataFrame(rows, columns=cols).set_index('Method').to_string()
    except Exception:
        widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
        line = '  '.join(c.ljust(widths[c]) for c in cols)
        sep = '  '.join('-' * widths[c] for c in cols)
        body = '\n'.join([line, sep] + [
            '  '.join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows])

    n_tag = (f"(avg over {n_paths} paths, mean±std)" if n_paths > 1
             else "(single path)")
    out = f"\n{header}\n{n_tag}\n{body}\n(* = gt=shift, 노란 강조 행)\n"
    return out, rows


# =====================================================
# 한 시나리오 전체 (path 평균)
# =====================================================
def run_scenario(scenario, *, base_policy, device, args, paths_idx_list,
                 anchors, base_machines, wq_min, wq_max,
                 pred_base_zip, pred_shift_zip, gt_ckpt):
    J, S = args.num_jobs, len(base_machines)
    p_csv = f'quality_data/P_{args.p_idx}.csv'

    print(f"\n############## Scenario {scenario}: {SCENARIO_DESC[scenario]} ##############")
    if os.path.abspath(pred_shift_zip) == os.path.abspath(pred_base_zip):
        print(f"[note] pred=shift 예측모델이 base 와 동일 — adapted 모델 학습 후 "
              f"--pred_s{scenario} 로 교체하세요 (지금은 pred=base 와 같은 결과).")

    agg = {r: {'HV': [], 'IGD+': [], 'Makespan': [], 'Quality': [], 'Time': []}
           for r in ROWS}
    machines_eff = None
    for paths_idx in paths_idx_list:
        base_csv = f'quality_data/Q_{args.q_idx}/historical_paths_{paths_idx}.csv'
        print(f"\n---------- scenario {scenario} | paths_{paths_idx} ----------")
        out, machines_eff = evaluate_one_path(
            scenario=scenario, base_policy=base_policy, device=device,
            J=J, S=S, base_machines=base_machines, base_csv=base_csv, p_csv=p_csv,
            anchors=anchors, seed=args.seed, ga_seed=args.ga_seed,
            wq_min=wq_min, wq_max=wq_max,
            num_lambdas=args.num_lambdas, samples=args.samples,
            nsga_pop=args.nsga_pop, nsga_gen=args.nsga_gen,
            yield_mode=args.yield_mode,
            pred_base_zip=pred_base_zip, pred_shift_zip=pred_shift_zip,
            gt_ckpt=gt_ckpt)

        # 1st pass: 방법별 독립 지표(HV/Makespan/Quality) + 점집합.
        pts = {r: (out[r][0], out[r][1]) for r in ROWS}
        md = {r: compute_metrics(out[r][0], out[r][1], anchors) for r in ROWS}
        # 2nd pass: 4행 점을 합쳐 best-known front → 행별 IGD+ (공통 기준).
        ref_front = build_reference_front(pts, anchors)
        for r in ROWS:
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

    header = (f"Scenario {scenario} ({SCENARIO_DESC[scenario]})  |  "
              f"N={J}, M={machines_eff}, (Q{args.q_idx},P{args.p_idx}), "
              f"yield={args.yield_mode}")
    table_str, rows = build_table(agg, header, len(paths_idx_list))
    print(table_str)

    csv_path = f'test_results/shift_scenario{scenario}_W{J}_table.csv'
    try:
        import pandas as pd
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"saved -> {csv_path}")
    except Exception as e:
        print(f"[warn] CSV 저장 실패: {e}")
    return rows


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
        description="품질 시나리오 변화(shift) 대비 모델 성능 비교 표 "
                    "(pred=base / pred=shift / gt=shift / NSGA).")
    # 고정 인스턴스 (기본 Q3, P3, paths_1).
    p.add_argument('--num_jobs', type=int, default=25)
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7',
                   help="base stage별 머신 수 (마지막 stage 가 변형 대상).")
    p.add_argument('--q_idx', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--p_idx', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--paths_idx', type=str, default='1',
                   help="historical_paths 인덱스. '1' / '1~10' / '1,3,5'. 여러 개면 path 평균.")
    p.add_argument('--scenarios', type=str, default='1,2,3',
                   help="실행할 시나리오. 예 '1,2,3' / '2'.")
    # 정책 체크포인트.
    p.add_argument('--ckpt_base', type=str,
                   default='checkpoints/saved_new_new_baseline.pt',
                   help="pred=base / pred=shift 가 공유하는 base 정책.")
    p.add_argument('--ckpt_gt', type=str, default='',
                   help="gt=shift 정책 공통 기본값 (빈값=ckpt_base). 시나리오별로 "
                        "--ckpt_gt_s{n} 로 덮어쓸 수 있음.")
    p.add_argument('--ckpt_gt_s1', type=str, default='')
    p.add_argument('--ckpt_gt_s2', type=str, default='')
    p.add_argument('--ckpt_gt_s3', type=str, default='')
    # 예측모델 (.zip; 같은 폴더의 wafer_quality.json 동반).
    p.add_argument('--pred_base', type=str,
                   default='quality_results/Q_3/paths_1/best_hb_model.zip',
                   help="pred=base 예측모델 (옛 품질 landscape).")
    p.add_argument('--pred_s1', type=str, default='',
                   help="시나리오1 adapted 예측모델 (빈값=pred_base).")
    p.add_argument('--pred_s2', type=str, default='')
    p.add_argument('--pred_s3', type=str, default='')
    # 평가 설정.
    p.add_argument('--yield_mode', type=str, default='raw', choices=['raw', 'percentile'])
    p.add_argument('--num_lambdas', type=int, default=32, help="모델 λ grid 크기.")
    p.add_argument('--samples', type=int, default=64,
                   help="모델 λ 당 sample 수 (>1=stochastic, 1=greedy). 클수록 front 풍부·느림.")
    p.add_argument('--nsga_pop', type=int, default=100)
    p.add_argument('--nsga_gen', type=int, default=10)
    p.add_argument('--xlim', type=str, default='100,600',
                   help="makespan anchor 'm_best,m_ref'.")
    p.add_argument('--ylim', type=str, default='0,1',
                   help="yield anchor 'q_ref,q_best'. percentile 모드는 0,100 으로 강제.")
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ga_seed', type=int, default=0)
    args = p.parse_args()

    base_machines = [int(x) for x in args.machines.split(',')]
    paths_idx_list = _parse_idx_spec(args.paths_idx)
    scenarios = _parse_idx_spec(args.scenarios)
    wq_min, wq_max = [float(x) for x in args.wafer_quality.split(',')]

    m_best, m_ref = [float(x) for x in args.xlim.split(',')]
    q_ref, q_best = [float(x) for x in args.ylim.split(',')]
    if args.yield_mode == 'percentile':
        q_ref, q_best = 0.0, 100.0
    anchors = (m_best, m_ref, q_best, q_ref)

    # 시나리오별 정책/예측모델 경로 매핑 (빈값 → 기본값으로 폴백).
    gt_ckpt_map = {1: args.ckpt_gt_s1, 2: args.ckpt_gt_s2, 3: args.ckpt_gt_s3}
    pred_zip_map = {1: args.pred_s1, 2: args.pred_s2, 3: args.pred_s3}
    gt_default = args.ckpt_gt or args.ckpt_base

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    print(f"[shift] device={device}  W={args.num_jobs}  base_machines={base_machines}  "
          f"(Q{args.q_idx},P{args.p_idx})  paths={paths_idx_list}  "
          f"scenarios={scenarios}  yield={args.yield_mode}")
    print(f"[shift] anchors: m=[best={m_best}, ref={m_ref}]  q=[ref={q_ref}, best={q_best}]")
    print(f"[shift] base 정책={args.ckpt_base}")
    print(f"[shift] 모델: λ×{args.num_lambdas}, samples/λ={args.samples}  |  "
          f"NSGA-II: pop={args.nsga_pop}, gen={args.nsga_gen}")

    # base 정책은 시나리오·env 무관(그래프 모델) → 한 번만 로드해 재사용.
    tmp_env = HFSPGraphEnv(num_jobs=args.num_jobs,
                           machine_cnt_list=base_machines, device=device)
    base_policy = load_policy(args.ckpt_base, tmp_env, device)

    all_rows = {}
    for scenario in scenarios:
        gt_ckpt = gt_ckpt_map[scenario] or gt_default
        pred_shift_zip = pred_zip_map[scenario] or args.pred_base
        all_rows[scenario] = run_scenario(
            scenario, base_policy=base_policy, device=device, args=args,
            paths_idx_list=paths_idx_list, anchors=anchors,
            base_machines=base_machines, wq_min=wq_min, wq_max=wq_max,
            pred_base_zip=args.pred_base, pred_shift_zip=pred_shift_zip,
            gt_ckpt=gt_ckpt)

    print("\n=== done ===")


if __name__ == "__main__":
    main()
