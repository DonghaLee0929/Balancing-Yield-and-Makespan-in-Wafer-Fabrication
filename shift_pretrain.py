"""
shift_pretrain.py — pred=shift 예측모델 재학습 파이프라인.

experiment_shift.py 의 각 품질 시나리오(1/2/3)에 대해, base 정책은 그대로 두고
*예측모델(QualityHelper)* 만 shift 에 맞춰 다시 fit 하기 위한 산출물을 만든다:

  1) shifted step_q 확보 — experiment_shift.apply_scenario 를 그대로 재사용해 마지막
     stage 품질 인자를 변형. 이렇게 해야 예측모델의 *학습 분포* 가 실험의 *채점 scorer*
     (shifted GroundTruthQuality) 와 1:1 동일해진다.
  2) base historical_paths CSV 의 Yield 라벨을 shifted step_q 로 재계산해 새 CSV 저장.
     - 시나리오 1·2: 마지막 stage 머신의 품질 인자만 갱신 → Yield = ∏(stage quality)×Wafer.
       머신 집합·OneHot 차원 불변.
     - 시나리오 3: 제거된 머신을 쓴 행을 drop 하고 남은 머신을 1..K 로 연속 재번호.
       (machine_cnt = Step열 max 로 잡히므로 번호에 구멍이 있으면 평가 env(머신 -1개)와
        OneHot 차원이 어긋난다. build_shifted_instance 의 keep 순서와 정렬을 맞춘다.)
  3) quality_prediction.analysis(csv_path, out_dir) 로 그 CSV 에서 예측모델 재학습 →
     best_hb_model.zip + wafer_quality.json 생성.

생성물(기본 q_idx=3, paths_idx=1):
  CSV   : quality_data/Q_{q}/shift_s{n}_historical_paths_{p}.csv
  model : quality_results/Q_{q}/shift_s{n}/best_hb_model.zip (+ wafer_quality.json)
끝에 experiment_shift.py 에 그대로 붙여넣을 --pred_s{n} 경로를 출력한다.

예) python shift_pretrain.py --scenarios 1,2,3
    python experiment_shift.py --pred_s1 quality_results/Q_3/shift_s1/best_hb_model.zip ...
"""

from __future__ import annotations
import os
import sys
import argparse

# 로컬 모듈이 stdlib 에 가려지지 않도록 스크립트 디렉터리를 최우선에.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 콘솔(cp949)에서 유니코드 출력 시 UnicodeEncodeError 방지 (analysis 의 한글/화살표 포함).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch

from quality_augment import GroundTruthQuality
from experiment_shift import apply_scenario, SCENARIO_DESC, _parse_idx_spec
from quality_prediction import analysis


def make_shifted_csv(scenario, base_csv, out_csv, base_machines, wq_min, wq_max,
                     s1_delta=0.01):
    """base CSV 의 Yield 라벨을 shifted step_q 로 재계산해 out_csv 에 저장.

    Returns: (machines_eff, removed_local_idx | None, n_rows_in, n_rows_out).
    """
    S = len(base_machines)
    last = S - 1
    # apply_scenario 가 gq.step_q(np) 를 in-place 변형 + (시나리오3) machine_cnt_list -1.
    gq = GroundTruthQuality(
        num_stages=S, device=torch.device('cpu'), csv_path=base_csv,
        machine_cnt_list=list(base_machines),
        wafer_quality_min=wq_min, wafer_quality_max=wq_max)
    removed = apply_scenario(gq, scenario, s1_delta=s1_delta)   # 시나리오3: 제거 머신 local idx, 그 외 None
    machines_eff = list(gq.machine_cnt_list)

    df = pd.read_csv(base_csv)
    n_in = len(df)
    step_col = f'Step {S}'                               # 마지막 stage 머신번호 (1-based)
    last_q_col = f'Step {S} quality'                     # 마지막 stage 품질 인자
    q_cols = [f'Step {s} quality' for s in range(1, S + 1)]

    if scenario in (1, 2):
        m_local = df[step_col].to_numpy() - 1            # 0-based local
        df[last_q_col] = gq.step_q[last, m_local]        # shifted 품질로 갱신
    elif scenario == 3:
        b = int(removed)
        df = df[df[step_col] != b + 1].copy()            # 제거 머신(1-based=b+1) 사용 행 drop
        m_old = df[step_col].to_numpy() - 1
        new_local = np.where(m_old < b, m_old, m_old - 1)  # keep 순서대로 1..K-1 연속 재번호
        df[step_col] = new_local + 1
        df[last_q_col] = gq.step_q[last, new_local]       # == 원 품질(값 불변), 정렬만 맞춤
    else:
        raise ValueError(f"scenario must be 1/2/3, got {scenario}")

    # Yield = ∏(stage quality) × Wafer quality  (곱셈 ground-truth 모델, quality_augment 참조).
    df['Yield'] = df[q_cols].prod(axis=1) * df['Wafer quality']

    # 평가 정합성 점검: 마지막 stage 머신번호가 1..machines_eff[last] 를 빠짐없이 덮는지.
    uniq = sorted(int(v) for v in df[step_col].unique())
    expect = list(range(1, machines_eff[last] + 1))
    if uniq != expect:
        print(f"[warn] 마지막 stage 머신번호 {uniq} != {expect} — "
              f"OneHot 차원이 평가 env({machines_eff[last]}머신)와 어긋날 수 있음.")

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    return machines_eff, removed, n_in, len(df)


def main():
    p = argparse.ArgumentParser(
        description="pred=shift 예측모델 재학습 (shifted Yield 라벨 CSV → quality_prediction).")
    p.add_argument('--q_idx', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--paths_idx', type=int, default=1,
                   help="base historical_paths 인덱스 (pred_base 와 동일하게 단일 파일).")
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7',
                   help="base stage별 머신 수 (experiment_shift 와 일치시킬 것).")
    p.add_argument('--scenarios', type=str, default='1,2,3',
                   help="재학습할 시나리오. '1,2,3' / '2' / '1~3'.")
    p.add_argument('--s1_delta', type=float, default=0.01,
                   help="시나리오1 강도: 최고품질 기계를 (그 stage 최저 - s1_delta) 로. "
                        "⚠ experiment_shift.py 의 --s1_delta 와 동일 값으로 평가해야 공정 비교.")
    p.add_argument('--wafer_quality', type=str, default='0.99,1.00')
    args = p.parse_args()

    base_machines = [int(x) for x in args.machines.split(',')]
    scenarios = _parse_idx_spec(args.scenarios)
    wq_min, wq_max = [float(x) for x in args.wafer_quality.split(',')]
    base_csv = f'quality_data/Q_{args.q_idx}/historical_paths_{args.paths_idx}.csv'

    if not os.path.exists(base_csv):
        raise FileNotFoundError(f"base CSV 없음 → {base_csv}")

    print(f"[pretrain] base CSV={base_csv}  machines={base_machines}  scenarios={scenarios}")

    pred_paths = {}
    for sc in scenarios:
        out_csv = (f'quality_data/Q_{args.q_idx}/'
                   f'shift_s{sc}_historical_paths_{args.paths_idx}.csv')
        out_dir = f'quality_results/Q_{args.q_idx}/shift_s{sc}'

        desc = (f"최고품질 기계 → (그 stage 최저 - {args.s1_delta})" if sc == 1
                else SCENARIO_DESC[sc])
        print(f"\n############## Scenario {sc}: {desc} ##############")
        machines_eff, removed, n_in, n_out = make_shifted_csv(
            sc, base_csv, out_csv, base_machines, wq_min, wq_max, s1_delta=args.s1_delta)
        drop_tag = (f", removed machine local={removed} (rows {n_in}→{n_out})"
                    if sc == 3 else "")
        print(f"[pretrain] shifted CSV -> {out_csv}  (machines_eff={machines_eff}{drop_tag})")

        analysis(csv_path=out_csv, out_dir=out_dir)      # → best_hb_model.zip + wafer_quality.json
        pred_paths[sc] = f'{out_dir}/best_hb_model.zip'

    print("\n=== shift 예측모델 준비 완료 — experiment_shift.py 에 아래 인자 추가 ===")
    flags = ' '.join(f'--pred_s{sc} {pred_paths[sc]}' for sc in scenarios)
    print(f"python experiment_shift.py {flags}")


if __name__ == "__main__":
    main()
