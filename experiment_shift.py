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

# 그림 폰트를 experiment_pareto.py 와 동일하게 STIX serif (Times 류, ↑/↓/λ 글리프 풀 커버).
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

from HFSPGraphEnv import HFSPGraphEnv
from FFSPModel import FFSPModel
from HFSPWrapper import QualityHelper, make_env_edge_lookup, get_feat_dims
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
    save_baseline_cache,
    load_baseline_cache,
    cache_meta_mismatch,
    COMPARISON_DIR,
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


def apply_scenario(gq: GroundTruthQuality, scenario: int, s1_delta: float = 0.1) -> int | None:
    """gq 의 마지막 stage step_q 를 시나리오대로 in-place 변형.

    1: 최고품질 기계를 (그 stage 최저 - s1_delta) 로,  2: 순위 reverse(값 집합 유지),
    3: 최고품질 기계 1개 제거 → machine_cnt_list[last] 1 감소.
    Returns: 제거한 기계 local index (scenario 3) 또는 None (1·2).
    """
    last = gq.num_stages - 1
    cnt = gq.machine_cnt_list[last]
    q = gq.step_q[last, :cnt].astype(np.float64).copy()
    removed = None

    if scenario == 1:
        b = int(np.argmax(q))
        q[b] = q.min() - s1_delta                   # 최고 → 최저보다 s1_delta 작게
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
                           device, wq_min, wq_max, s1_delta=0.1):
    """시나리오별 (shifted GT scorer, proc_time, 유효 machine_cnt_list) 구성.

    scorer(=GroundTruthQuality) 는 base CSV 로 만든 뒤 마지막 stage 를 시나리오대로 변형.
    proc_time 은 base machines 로 로드 후, scenario 3 이면 제거된 기계의 열을 drop 해 정렬 유지.
    """
    S = len(base_machines)
    gq = GroundTruthQuality(num_stages=S, device=device, csv_path=base_csv,
                            machine_cnt_list=list(base_machines),
                            wafer_quality_min=wq_min, wafer_quality_max=wq_max)
    removed = apply_scenario(gq, scenario, s1_delta=s1_delta)
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
# 캐시 — 시나리오당 1개 npz(shift_points_s{n}_W{J}.npz)에 행별·path별 점 저장.
# 키(meta)를 model(pred/gt)·nsga 두 그룹으로 나눠 한쪽 설정이 바뀌어도 다른 쪽 점은 보존.
# meta 없는 옛 번들({slug}_ms concat)은 '설정 동일' 로 신뢰해 그대로 재사용(재계산 0).
# =====================================================
ROW_SLUG = {'pred=base': 'predbase', 'pred=shift': 'predshift',
            'gt=shift': 'gtshift', 'NSGA (GT)': 'nsga'}
ROW_GROUP = {'pred=base': 'model', 'pred=shift': 'model',
             'gt=shift': 'model', 'NSGA (GT)': 'nsga'}


def effective_machines(scenario, base_machines):
    """build_shifted_instance 가 만들 machine_cnt_list 를 모델 로드 없이 미리 계산.
    시나리오 3 은 마지막 stage 머신 1개 제거(-1), 그 외는 base 그대로."""
    m = list(base_machines)
    if scenario == 3:
        m[-1] -= 1
    return m


def _artifact_sig(path):
    """캐시 키용 아티팩트 식별자 — 경로+mtime+size. 파일이 바뀌면(재학습) 키가 달라져 무효화."""
    try:
        st = os.stat(path)
        return {'path': str(path), 'mtime': float(st.st_mtime), 'size': int(st.st_size)}
    except OSError:
        return {'path': str(path), 'mtime': None, 'size': None}


def cache_group_metas(*, scenario, s1_delta, q_idx, p_idx, J, base_machines, machines_eff,
                      num_lambdas, samples, seed, ga_seed, yield_mode, wq_min, wq_max, anchors,
                      nsga_pop, nsga_gen, ckpt_base, pred_base_zip, pred_shift_zip, gt_ckpt):
    """model(pred/gt)·nsga 두 그룹의 무효화 키. 각 그룹 점에 영향 주는 설정만 담는다."""
    base = {'scenario': int(scenario), 's1_delta': float(s1_delta),
            'q_idx': int(q_idx), 'p_idx': int(p_idx), 'J': int(J),
            'base_machines': list(base_machines), 'machines_eff': list(machines_eff),
            'yield_mode': str(yield_mode), 'wq_min': float(wq_min), 'wq_max': float(wq_max)}
    model = {**base, 'num_lambdas': int(num_lambdas), 'samples': int(samples), 'seed': int(seed),
             'artifacts': {'ckpt_base': _artifact_sig(ckpt_base),
                           'pred_base': _artifact_sig(pred_base_zip),
                           'pred_shift': _artifact_sig(pred_shift_zip),
                           'gt_ckpt': _artifact_sig(gt_ckpt)}}
    nsga = {**base, 'nsga_pop': int(nsga_pop), 'nsga_gen': int(nsga_gen),
            'ga_seed': int(ga_seed), 'seed': int(seed), 'anchors': [float(a) for a in anchors]}
    return {'model': model, 'nsga': nsga}


def scenario_cache_path(scenario, J):
    """시나리오당 캐시 파일 (옛 plot 번들과 동일 경로 → 기존 점 그대로 재사용)."""
    return os.path.join(COMPARISON_DIR, f"shift_points_s{scenario}_W{J}.npz")


def load_scenario_cache(scenario, J):
    """→ (data {key:array}, group_metas {'model':..,'nsga':..} 또는 {}=옛 번들). 없으면 ({},{})."""
    path = scenario_cache_path(scenario, J)
    if not os.path.exists(path):
        return {}, {}
    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    metas = {}
    if 'meta' in data:
        try:
            metas = json.loads(str(data['meta'].item()))
        except Exception:
            metas = {}
    return data, metas


def save_scenario_cache(scenario, J, data, group_metas):
    """data(점들) + group_metas(검증키) 를 npz 로 저장."""
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    out = {k: v for k, v in data.items() if k != 'meta'}
    out['meta'] = np.asarray(json.dumps(group_metas))
    np.savez(scenario_cache_path(scenario, J), **out)


def _try_load_row(data, file_metas, paths_idx, row, cur_metas, refresh):
    """캐시에서 (row, paths_idx) 점 로드 → (ms,yld,t) 또는 None(계산 필요).

    file_metas 비어있으면(옛 번들) 검증 없이 신뢰. 새 포맷이면 그룹 키가 일치해야 로드.
    옛 번들은 path 구분 없는 concat 키({slug}_ms) → paths_idx==1 로만 재사용.
    """
    if refresh:
        return None
    slug, group = ROW_SLUG[row], ROW_GROUP[row]
    if file_metas:                                   # 새 포맷 — 그룹 키 검증
        stored = file_metas.get(group)
        if stored is None or cache_meta_mismatch(stored, cur_metas[group]):
            return None
    key = f'p{paths_idx}_{slug}_ms'
    if key in data:
        t = float(data[f'p{paths_idx}_{slug}_t']) if f'p{paths_idx}_{slug}_t' in data else 0.0
        return (np.asarray(data[key], np.float32),
                np.asarray(data[f'p{paths_idx}_{slug}_yld'], np.float32), t)
    if paths_idx == 1 and not file_metas and f'{slug}_ms' in data:    # 옛 번들 concat → p1
        return (np.asarray(data[f'{slug}_ms'], np.float32),
                np.asarray(data[f'{slug}_yld'], np.float32), 0.0)
    return None


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
                      pred_base_zip, pred_shift_zip, gt_ckpt, s1_delta, rows_to_compute):
    """rows_to_compute 에 든 행만 계산해 ({row:(ms,yld,t)}, machines_eff) 반환.

    캐시 로드/저장은 run_scenario 가 시나리오당 1파일로 관리한다. 필요한 행이 쓰는
    모델만 로드해 불필요한 setup 을 피한다.
    """
    gq, proc, machines_eff = build_shifted_instance(
        scenario, base_csv, p_csv, J, base_machines, device, wq_min, wq_max, s1_delta=s1_delta)
    env = HFSPGraphEnv(num_jobs=J, machine_cnt_list=machines_eff, device=device)
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)
    # 모든 방법이 동일 wafer_quality 를 보도록 한 번만 샘플 (shifted GT 에서).
    wq_1 = gq.sample_wafer_quality(B=1, num_jobs=J, seed=seed, device=device)   # (1, J)

    out = {}
    if 'pred=base' in rows_to_compute:               # base 정책 + base 예측모델, shifted GT 채점
        qh = make_quality_helper(pred_base_zip, S, device)
        (ms, yld), t = timed(lambda: model_front(
            base_policy, env, env_edge_lookup_t, device, qh, gq, proc, wq_1,
            num_lambdas, samples, seed, yield_mode), device)
        out['pred=base'] = (ms, yld, t)
    if 'pred=shift' in rows_to_compute:              # base 정책 + shift 예측모델
        qh = make_quality_helper(pred_shift_zip, S, device)
        (ms, yld), t = timed(lambda: model_front(
            base_policy, env, env_edge_lookup_t, device, qh, gq, proc, wq_1,
            num_lambdas, samples, seed, yield_mode), device)
        out['pred=shift'] = (ms, yld, t)
    if 'gt=shift' in rows_to_compute:                # gt 정책 + shifted GT oracle (helper=scorer=gq)
        gt_policy = load_policy(gt_ckpt, env, device)
        (ms, yld), t = timed(lambda: model_front(
            gt_policy, env, env_edge_lookup_t, device, gq, gq, proc, wq_1,
            num_lambdas, samples, seed, yield_mode), device)
        out['gt=shift'] = (ms, yld, t)
    if 'NSGA (GT)' in rows_to_compute:               # shifted GT scorer
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


# 4행 색맵 (Pareto 플롯). pred=base 파랑 / pred=shift 빨강 / gt=shift 초록 / NSGA 주황.
ROW_COLORS = {'pred=base': 'tab:gray', 'pred=shift': 'tab:red',
              'gt=shift': 'tab:green', 'NSGA (GT)': 'tab:orange'}

# 플롯에 그릴 행 — gt=shift / NSGA 제외, pred=base vs pred=shift 만 시각화.
PLOT_ROWS = ['pred=base', 'pred=shift']
# 플롯 범례 표기 (데이터 키 ROWS 는 그대로, 표시 라벨만). continual 의 _LEGEND_LABEL 패턴.
PLOT_LABEL = {'pred=base': 'Pred=base', 'pred=shift': 'Pred=shift'}
SCENARIO_DESC_EN = {
    1: 'best-quality machine -> (stage min - {delta})',
    2: 'last-stage quality rank reversed',
    3: 'remove last-stage best-quality machine (machines -1)',
}

# 그림 크기 — experiment_pareto.py 와 동일 (점 대비 플롯 영역이 너무 커 보이지 않게).
#   단일: FIGSIZE_SINGLE = (6.2, 4.3),  결합: per-subplot ≈ 4.33×3.8 (pareto grid 13/3, 7.6/2).
FIGSIZE_SINGLE = (6.2, 4.3)
FIGSIZE_COMBINED_PER_COL = 4.33                   # subplot 1 개당 폭
FIGSIZE_COMBINED_HEIGHT = 5.0                     # 1 행 전체 높이 (suptitle + 하단 범례 포함)

# 폰트 크기 — experiment_continual.py 와 동일 컨벤션 (축 라벨 14 / suptitle 20 / 범례 12).
_PLOT_AXLABEL_FONTSIZE = 14    # 축 라벨 (Makespan ↓ / Yield ↑)
_PLOT_TITLE_FONTSIZE = 15      # 각 subplot 제목 (시나리오 이름)
_PLOT_SUPTITLE_FONTSIZE = 20   # 상단 큰 제목 — continual 과 동일
_PLOT_LEGEND_FONTSIZE = 12     # 하단 공통 범례 — continual 과 동일


def _draw_scenario_fronts(ax, pts, title, with_legend=True,
                          cloud_frac=0.0, cloud_seed=0):
    """단일 axes 에 PLOT_ROWS 의 Pareto front (+옵션: dominated 점 n% cloud) 를 그린다.

    cloud_frac (0.0~1.0): dominated(비-front) 점들 중 무작위로 이만큼만 옅은 산점도로 추가.
    0=완전 클린(Pareto front 만, 기본), 1=원본 전부. front 자체는 *항상 full 데이터* 로 계산해
    표시되는 cloud 와 무관하게 front 의 정확성이 보존된다.
    with_legend=False: ax 자체 범례 생략 — 결합 플롯에서 fig.legend 로 하단 공통 범례 사용.
    """
    rng = np.random.default_rng(cloud_seed) if cloud_frac > 0 else None
    for r in PLOT_ROWS:
        ms = np.asarray(pts[r][0], np.float64).ravel()
        yld = np.asarray(pts[r][1], np.float64).ravel()
        if ms.size == 0:
            continue
        c = ROW_COLORS[r]
        front = compute_pareto_front(ms, yld)
        if cloud_frac > 0:                              # dominated 점만 n% 옅게
            mask = np.ones(ms.size, dtype=bool)
            mask[front] = False
            dom_idx = np.where(mask)[0]
            if dom_idx.size > 0:
                k = max(1, int(round(dom_idx.size * min(cloud_frac, 1.0))))
                pick = (dom_idx if k >= dom_idx.size
                        else rng.choice(dom_idx, size=k, replace=False))
                ax.scatter(ms[pick], yld[pick], s=10, alpha=0.22, color=c)
        lbl = PLOT_LABEL.get(r, r)
        if front.size >= 2:
            order = np.argsort(ms[front])              # makespan 오름차순 연결
            ax.plot(ms[front][order], yld[front][order], '-o', ms=5, lw=1.6,
                    color=c, label=lbl)
        elif front.size == 1:
            ax.scatter(ms[front], yld[front], s=90, marker='*',
                       color=c, edgecolor='black', zorder=5, label=lbl)
    ax.set_xlabel('Makespan ↓', fontsize=_PLOT_AXLABEL_FONTSIZE)
    ax.set_ylabel('Yield ↑', fontsize=_PLOT_AXLABEL_FONTSIZE)
    ax.set_title(title, fontsize=_PLOT_TITLE_FONTSIZE, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    if with_legend:
        ax.legend(loc='best', fontsize=_PLOT_LEGEND_FONTSIZE)


def plot_scenario_fronts(pts, save_path, title, cloud_frac=0.0, cloud_seed=0):
    """PLOT_ROWS 의 점집합을 한 그림에 overlay — Pareto front (+ 옵션 cloud)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    _draw_scenario_fronts(ax, pts, title, cloud_frac=cloud_frac, cloud_seed=cloud_seed)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_combined_fronts(items, save_path, suptitle='', cloud_frac=0.0, cloud_seed=0):
    """items=[(pts, subtitle), ...] → 가로 N 개 서브플롯 + 큰 suptitle + 하단 공통 범례.

    experiment_pareto.py 의 batch grid 스타일을 따라 했다 — 각 subplot 은 bold subtitle 만,
    범례는 figure 하단(upper center, y=0.13)에 ncol=N 으로 한 번만.
    """
    n = len(items)
    fig, axes = plt.subplots(
        1, n,
        figsize=(FIGSIZE_COMBINED_PER_COL * n, FIGSIZE_COMBINED_HEIGHT),
        squeeze=False)
    for ax, (pts, subtitle) in zip(axes[0], items):
        _draw_scenario_fronts(ax, pts, subtitle, with_legend=False,
                              cloud_frac=cloud_frac, cloud_seed=cloud_seed)

    # 하단 공통 범례 — Pareto front 선/마커를 그대로 미러링.
    handles = [
        Line2D([0], [0], linestyle='-', marker='o', markersize=6, lw=1.6,
               color=ROW_COLORS['pred=base'], label=PLOT_LABEL['pred=base']),
        Line2D([0], [0], linestyle='-', marker='o', markersize=6, lw=1.6,
               color=ROW_COLORS['pred=shift'], label=PLOT_LABEL['pred=shift']),
    ]
    fig.tight_layout(rect=[0, 0.13, 1, 0.94])     # 아래 범례 + 위 suptitle 공간
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.15),
               ncol=len(handles), frameon=True,
               prop=dict(size=_PLOT_LEGEND_FONTSIZE, weight='bold'),
               columnspacing=0.8, handletextpad=0.4)
    if suptitle:
        fig.suptitle(suptitle, fontsize=_PLOT_SUPTITLE_FONTSIZE,
                     fontweight='bold', y=0.99)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# =====================================================
# 한 시나리오 전체 (path 평균)
# =====================================================
def run_scenario(scenario, *, base_policy, device, args, paths_idx_list,
                 anchors, base_machines, wq_min, wq_max,
                 pred_base_zip, pred_shift_zip, gt_ckpt):
    J, S = args.num_jobs, len(base_machines)
    p_csv = f'quality_data/P_{args.p_idx}.csv'
    desc = (f"최고품질 기계 → (그 stage 최저 - {args.s1_delta})" if scenario == 1
            else SCENARIO_DESC[scenario])
    machines_eff = effective_machines(scenario, base_machines)

    print(f"\n############## Scenario {scenario}: {desc} ##############")
    if os.path.abspath(pred_shift_zip) == os.path.abspath(pred_base_zip):
        print(f"[note] pred=shift 예측모델이 base 와 동일 — adapted 모델 학습 후 "
              f"--pred_s{scenario} 로 교체하세요 (지금은 pred=base 와 같은 결과).")

    # 시나리오당 1파일 캐시 로드 + 현재 설정 키. meta 없으면 옛 번들 → 신뢰 재사용(재계산 0).
    cur_metas = cache_group_metas(
        scenario=scenario, s1_delta=args.s1_delta, q_idx=args.q_idx, p_idx=args.p_idx, J=J,
        base_machines=base_machines, machines_eff=machines_eff,
        num_lambdas=args.num_lambdas, samples=args.samples, seed=args.seed, ga_seed=args.ga_seed,
        yield_mode=args.yield_mode, wq_min=wq_min, wq_max=wq_max, anchors=anchors,
        nsga_pop=args.nsga_pop, nsga_gen=args.nsga_gen, ckpt_base=args.ckpt_base,
        pred_base_zip=pred_base_zip, pred_shift_zip=pred_shift_zip, gt_ckpt=gt_ckpt)
    cache_data, file_metas = load_scenario_cache(scenario, J)
    if cache_data and not file_metas:
        print("  [cache] 옛 번들(검증키 없음) — 설정 동일 가정하고 그대로 재사용")

    agg = {r: {'HV': [], 'IGD+': [], 'Makespan': [], 'Quality': [], 'Time': []}
           for r in ROWS}
    plot_pts = {r: ([], []) for r in ROWS}          # path 별 점 누적 (Pareto 플롯용)
    save_data = {}                                  # 새 포맷으로 다시 저장할 점들
    for paths_idx in paths_idx_list:
        base_csv = f'quality_data/Q_{args.q_idx}/historical_paths_{paths_idx}.csv'
        print(f"\n---------- scenario {scenario} | paths_{paths_idx} ----------")

        out, to_compute = {}, []
        for r in ROWS:
            refresh = args.nsga_refresh if r == 'NSGA (GT)' else args.model_refresh
            loaded = _try_load_row(cache_data, file_metas, paths_idx, r, cur_metas, refresh)
            if loaded is not None:
                out[r] = loaded
            else:
                to_compute.append(r)
        if to_compute:
            print(f"  [compute] {to_compute}")
            computed, machines_eff = evaluate_one_path(
                scenario=scenario, base_policy=base_policy, device=device,
                J=J, S=S, base_machines=base_machines, base_csv=base_csv, p_csv=p_csv,
                anchors=anchors, seed=args.seed, ga_seed=args.ga_seed,
                wq_min=wq_min, wq_max=wq_max,
                num_lambdas=args.num_lambdas, samples=args.samples,
                nsga_pop=args.nsga_pop, nsga_gen=args.nsga_gen, yield_mode=args.yield_mode,
                pred_base_zip=pred_base_zip, pred_shift_zip=pred_shift_zip,
                gt_ckpt=gt_ckpt, s1_delta=args.s1_delta, rows_to_compute=to_compute)
            out.update(computed)
        else:
            print(f"  [cache] 4행 모두 히트 (p{paths_idx}) → 계산 생략")

        for r in ROWS:                              # 새 포맷(path별)으로 적재
            slug = ROW_SLUG[r]
            ms, yld, t = out[r]
            save_data[f'p{paths_idx}_{slug}_ms'] = np.asarray(ms, np.float32).ravel()
            save_data[f'p{paths_idx}_{slug}_yld'] = np.asarray(yld, np.float32).ravel()
            save_data[f'p{paths_idx}_{slug}_t'] = np.asarray(float(t), np.float32)

        # 1st pass: 방법별 독립 지표 + 점집합 / 2nd pass: 공통 ref front → 행별 IGD+.
        pts = {r: (out[r][0], out[r][1]) for r in ROWS}
        md = {r: compute_metrics(out[r][0], out[r][1], anchors) for r in ROWS}
        ref_front = build_reference_front(pts, anchors)
        for r in ROWS:
            ms, yld, t = out[r]
            igd = igd_plus_metric(ms, yld, ref_front, anchors)
            agg[r]['HV'].append(md[r]['HV'])
            agg[r]['IGD+'].append(igd)
            agg[r]['Makespan'].append(md[r]['Makespan'])
            agg[r]['Quality'].append(md[r]['Quality'])
            agg[r]['Time'].append(t)
            plot_pts[r][0].append(np.asarray(ms, np.float64).ravel())
            plot_pts[r][1].append(np.asarray(yld, np.float64).ravel())
            igd_str = '—' if np.isnan(igd) else f"{igd:.4f}"
            print(f"  {r:11s}  HV={md[r]['HV']:.4f}  IGD+={igd_str}  "
                  f"ms={md[r]['Makespan']:.1f}  q={md[r]['Quality']:.4f}  "
                  f"t={t:.2f}s  |pts|={np.asarray(ms).size}")

    try:                                            # 캐시 저장(옛 번들 → 새 포맷 승격)
        save_scenario_cache(scenario, J, save_data, cur_metas)
        print(f"saved -> {scenario_cache_path(scenario, J)}  (점 캐시)")
    except Exception as e:
        print(f"[warn] 점 캐시 저장 실패: {e}")

    header = (f"Scenario {scenario} ({desc})  |  "
              f"N={J}, M={machines_eff}, (Q{args.q_idx},P{args.p_idx}), "
              f"yield={args.yield_mode}")
    table_str, rows = build_table(agg, header, len(paths_idx_list))
    print(table_str)

    # 시나리오별 CSV/PNG 는 더 이상 저장하지 않는다 — main() 이 모든 시나리오를 합쳐
    # shift_combined 표(CSV) + 그림(PNG) 하나씩만 출력한다.
    pts_cat = {r: (np.concatenate(plot_pts[r][0]) if plot_pts[r][0] else np.array([]),
                   np.concatenate(plot_pts[r][1]) if plot_pts[r][1] else np.array([]))
               for r in ROWS}
    desc_en = (SCENARIO_DESC_EN[1].format(delta=args.s1_delta) if scenario == 1
               else SCENARIO_DESC_EN[scenario])
    plot_title = (f"Scenario {scenario} ({desc_en})  |  "
                  f"N={J}, M={machines_eff}, (Q{args.q_idx},P{args.p_idx}), "
                  f"yield={args.yield_mode}")
    return rows, pts_cat, plot_title


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


def default_shift_pred(q_idx: int, scenario: int) -> str:
    """shift_pretrain.py 가 저장하는 시나리오별 adapted 예측모델 관례 경로 (존재할 때만 반환).

    빈 문자열이면 호출부에서 pred_base 로 폴백 → adapted 모델이 아직 없으면 기존 동작 유지.
    """
    path = f'quality_results/Q_{q_idx}/shift_s{scenario}/best_hb_model.zip'
    return path if os.path.exists(path) else ''


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
    p.add_argument('--s1_delta', type=float, default=0.1,
                   help="시나리오1 강도: 최고품질 기계를 (그 stage 최저 - s1_delta) 로. "
                        "키울수록 stale 예측모델(pred=base) 페널티 ↑. "
                        "⚠ shift_pretrain.py 의 --s1_delta 와 동일 값으로 학습해야 공정 비교.")
    # 정책 체크포인트.
    p.add_argument('--ckpt_base', type=str,
                   default='checkpoints/0523_baseline.pt',
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
                   help="시나리오1 adapted 예측모델. 빈값이면 "
                        "quality_results/Q_{q_idx}/shift_s1/best_hb_model.zip "
                        "(shift_pretrain.py 산출물) 가 있으면 그걸, 없으면 pred_base 를 사용.")
    p.add_argument('--pred_s2', type=str, default='')
    p.add_argument('--pred_s3', type=str, default='')
    # 평가 설정.
    p.add_argument('--yield_mode', type=str, default='raw', choices=['raw', 'percentile'])
    p.add_argument('--num_lambdas', type=int, default=32, help="모델 λ grid 크기.")
    p.add_argument('--samples', type=int, default=64,
                   help="모델 λ 당 sample 수 (>1=stochastic, 1=greedy). 클수록 front 풍부·느림.")
    p.add_argument('--nsga_pop', type=int, default=100)
    p.add_argument('--nsga_gen', type=int, default=200)
    p.add_argument('--model_refresh', default=False,
                   help="모델 행(pred=base/shift, gt=shift) 캐시 무시·재계산+덮어쓰기. 켜려면 --model_refresh 1.")
    p.add_argument('--nsga_refresh', default=False,
                   help="NSGA 캐시 무시·재계산+덮어쓰기. 켜려면 --nsga_refresh 1.")
    p.add_argument('--xlim', type=str, default='100,600',
                   help="makespan anchor 'm_best,m_ref'.")
    p.add_argument('--ylim', type=str, default='0,1',
                   help="yield anchor 'q_ref,q_best'. percentile 모드는 0,100 으로 강제.")
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ga_seed', type=int, default=0)
    p.add_argument('--cloud_frac', type=float, default=0.05,
                   help="Pareto front 옆에 표시할 dominated 점들의 비율 0~1 "
                        "(0=front 만 깔끔, 1=원본 전부). 시각 노이즈 vs 분포 단서 트레이드오프.")
    args = p.parse_args()
    if not (0.0 <= args.cloud_frac <= 1.0):
        raise ValueError(f"--cloud_frac 은 0~1, got {args.cloud_frac}")

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
    if 1 in scenarios:
        print(f"[shift] scenario1 s1_delta={args.s1_delta} "
              f"(pred=shift 예측모델도 같은 s1_delta 로 학습됐는지 확인)")

    # base 정책은 시나리오·env 무관(그래프 모델) → 한 번만 로드해 재사용.
    tmp_env = HFSPGraphEnv(num_jobs=args.num_jobs,
                           machine_cnt_list=base_machines, device=device)
    base_policy = load_policy(args.ckpt_base, tmp_env, device)

    all_rows = {}
    combined_items = []                                 # (pts_cat, short subtitle) 시나리오 순서대로
    for scenario in scenarios:
        gt_ckpt = gt_ckpt_map[scenario] or gt_default
        pred_shift_zip = (pred_zip_map[scenario]
                          or default_shift_pred(args.q_idx, scenario)
                          or args.pred_base)
        rows, pts_cat, _ = run_scenario(
            scenario, base_policy=base_policy, device=device, args=args,
            paths_idx_list=paths_idx_list, anchors=anchors,
            base_machines=base_machines, wq_min=wq_min, wq_max=wq_max,
            pred_base_zip=args.pred_base, pred_shift_zip=pred_shift_zip,
            gt_ckpt=gt_ckpt)
        all_rows[scenario] = rows
        combined_items.append((pts_cat, f'Scenario {scenario}'))

    # 결합 표(CSV) — 모든 시나리오 행을 Scenario 열을 붙여 하나로.
    combined_csv = f'test_results/shift_combined_W{args.num_jobs}_table.csv'
    try:
        import pandas as pd
        combined_rows = []
        for scenario in scenarios:
            for row in all_rows[scenario]:
                combined_rows.append({'Scenario': scenario, **row})
        os.makedirs(os.path.dirname(combined_csv) or '.', exist_ok=True)
        pd.DataFrame(combined_rows).to_csv(combined_csv, index=False, encoding='utf-8-sig')
        print(f"\nsaved -> {combined_csv}  (결합 표)")
    except Exception as e:
        print(f"[warn] 결합 CSV 저장 실패: {e}")

    # 결합 그림(PNG) — 시나리오들을 가로로 나란히.
    combined_path = f'test_results/shift_combined_W{args.num_jobs}_pareto.png'
    suptitle = 'Pareto Frontier under Distribution Shift'
    try:
        plot_combined_fronts(combined_items, combined_path, suptitle=suptitle,
                             cloud_frac=args.cloud_frac, cloud_seed=args.seed)
        print(f"saved -> {combined_path}  ({len(combined_items)} 시나리오 결합 플롯)")
    except Exception as e:
        print(f"[warn] 결합 Pareto 플롯 저장 실패: {e}")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
