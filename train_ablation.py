"""
Ablation — 부정확한 reward 예측기로 학습, GT 예측기로 yield 만 측정.

setup:
  학습 rollout 의 보상(yield)         = PATHS_TRAIN 예측기  (부정확, cross-domain)
  Eval rollout 의 features/wafer 샘플 = PATHS_TRAIN 예측기  (학습 분포와 동일)
  Eval Pareto 점의 yield 좌표(GT)     = PATHS_EVAL  예측기  (정확, R²≈0.999)

→ "정책이 잘못된 보상으로 학습된 채, eval input 은 학습 분포 그대로 두고 yield 만
   GT 로 측정했을 때 Pareto front 가 얼마나 무너지나" 를 매 epoch schedule dump 로
   추적 (post-hoc 으로 어떤 GT 예측기든 갈아끼워 재평가 가능).

설계:
  train_ASIL.train() 은 quality_helper 외부 주입을 이미 지원 → 학습용은 qh_train 직주입.
  eval_pareto 는 quality_helper 가 (sample_wafer / machine_quality / compute_yield)
  세 군데에서 호출됨 → HybridQualityHelper 로 라우팅 분리 후 train_ASIL 모듈 namespace
  의 eval_pareto 심볼을 monkeypatch.
  anchor 들도 train_ASIL 의 module-global 을 main() 진입 시 덮어쓴다.
  (train_ASIL.py / HFSPWrapper.py 등 기존 파일 무수정.)
"""

from __future__ import annotations
import argparse
import json
import os
import torch

import train_ASIL
from HFSPWrapper import QualityHelper


# =====================================================
# Anchors — ablation 전용. 두 그룹의 역할이 완전히 분리됨. 사용자가 직접 조정.
#
# [1] REWARD_* — Reward scaling 앵커. sil_pomo_rollout 안 (m̂, q̂) ∈ [0,1] 정규화 분모.
#     m̂ = (REWARD_M_WORST - ms) / (REWARD_M_WORST - REWARD_M_BEST), q̂ 도 동일 꼴.
#     ASIL POMO baseline 이 instance-내 advantage 를 표준화하므로 *학습 방향* 자체엔
#     거의 영향 없음 (project_reward_anchor_scale_invariance 메모 참조). 그래도 노출.
#
# [2] HVN_* — Pareto eval 의 normalized HV(`hvN`) 박스 corner. (ms, q) → [0,1]² 매핑 후
#     hvN 계산. raw HV(HV_REF_*) 와 완전 독립 — run 간 비교용 단일 척도.
#     makespan: [HVN_M_BEST, HVN_M_WORST] → [0,1]  (낮을수록 좋음)
#     yield:    [HVN_Q_WORST, HVN_Q_BEST] → [0,1]  (높을수록 좋음)
#
# [3] HV_REF_* — raw HV(`hv`) 의 reference point (nadir = 최악 코너). best-ckpt 선택 기준
#     (train_ASIL 안에서 hv_mean > best_eval). 절대 단위 → run 간 비교에는 [2] hvN 사용.
#     makespan 이 HV_REF_M 초과 또는 yield 가 HV_REF_Q 미만인 점 → raw HV 기여 0.
#
# [4] PLOT_* — Pareto scatter 그림 축 범위. HV/hvN 계산과 완전 독립, 순수 시각화용.
# =====================================================
# [1] Reward scaling
REWARD_M_BEST  = 100.0    # makespan best  (낮을수록 좋음 → 정규화 분자 상한)
REWARD_M_WORST = 500.0    # makespan worst
REWARD_Q_BEST  = 0.85     # yield best     (높을수록 좋음)
REWARD_Q_WORST = 0.50     # yield worst

# [2] HVN normalization (paths_1 GT 기준 평가용)
HVN_M_BEST  = 100.0
HVN_M_WORST = 600.0
HVN_Q_WORST = 0.0
HVN_Q_BEST  = 1.0

# [3] Raw HV reference (best-ckpt 선택용)
HV_REF_M = 600.0
HV_REF_Q = 0.50

# [4] Plot axes
PLOT_M_MIN = 100.0
PLOT_M_MAX = 550.0
PLOT_Q_MIN = 0.50
PLOT_Q_MAX = 0.85

# =====================================================
# Quality predictor 선택
# =====================================================
Q_ID         = 3
PATHS_TRAIN  = 10  # 학습 reward 예측기 — 부정확 (cross-domain)
PATHS_EVAL   = 1   # 평가 GT 예측기   — 정확 (환경과 동일)

# Per-epoch eval dump (단일 jsonl).
#   첫 줄  = 메타.  {"meta": {lambdas, num_jobs, num_stages, machine_cnt_list, K}}
#   이후 줄 = 한 epoch.  {epoch, makespan [L,K], quality [L,K], hvn [K]}
#     quality = qh_eval (GT) 로 매긴 yield → 학습 reward(qh_train) 와 measurement gap 가 직접 보임.
#     hvn     = 정규화 HV (anchor box 기준 [0,1] 스케일) per-instance.
PARETO_POINTS_LOG = f"train_results/ablation_p{PATHS_TRAIN}train_p{PATHS_EVAL}eval_pareto_points.jsonl"


def build_helper(paths_id: int, num_stages: int, device) -> QualityHelper:
    return QualityHelper(
        num_stages=num_stages, device=device,
        model_path=f"quality_results/Q_{Q_ID}/paths_{paths_id}/best_hb_model.zip",
        wafer_path=f"quality_results/Q_{Q_ID}/paths_{paths_id}/wafer_quality.json",
    )


class HybridQualityHelper:
    """Eval 전용 hybrid — features(sample_wafer, machine_quality) 는 `feat` 헬퍼,
    yield 측정(compute_yield) 만 `yield_h` 헬퍼로 라우팅.

    eval_pareto 안의 호출 경로 (HFSPWrapper.py:585~600):
      sample_wafer_quality + _rollout_loop 내 compute_machine_quality → feat
        → 정책이 학습 때 보던 분포/피처 그대로 입력 (도메인 시프트 없음).
      compute_yield → yield_h
        → Pareto 점의 yield 좌표가 GT(=PATHS_EVAL) 기준 → HV/HVN 도 자동으로 GT 기준.

    wafer_quality_min/max 가 QualityHelper 에서 [0.99, 1.00] 으로 하드코딩 (HFSPWrapper.py:98-99)
    이라 feat 가 뽑은 wafer 샘플을 yield_h 모델에 그대로 먹여도 분포 일관성 유지.
    sil_pomo_rollout 의 학습 reward 는 train() 인자로 qh_train 직주입 — 여기 hybrid 무관.
    """

    def __init__(self, feat: QualityHelper, yield_h: QualityHelper):
        self._feat = feat
        self._yield = yield_h

    @property
    def is_active(self) -> bool:
        return self._feat.is_active and self._yield.is_active

    def sample_wafer_quality(self, *a, **kw):
        return self._feat.sample_wafer_quality(*a, **kw)

    def compute_machine_quality(self, *a, **kw):
        return self._feat.compute_machine_quality(*a, **kw)

    def compute_yield(self, *a, **kw):
        return self._yield.compute_yield(*a, **kw)


def patch_eval_helper(eval_helper, log_path: str,
                      num_jobs: int, machine_cnt_list: list[int]) -> None:
    """train_ASIL.eval_pareto 에 hybrid helper 강제 주입 + 매 epoch '할당' 저장.

    train_ASIL.py 가 `from HFSPWrapper import eval_pareto` 로 import 했기 때문에
    train_ASIL 모듈 namespace 의 eval_pareto 심볼만 갈아끼우면 됨.
    sil_pomo_rollout 은 train() 이 명시적으로 quality_helper 를 넘기므로 영향 없음.

    저장 형태 (단일 jsonl):
      첫 줄  : {"meta": {lambdas, num_jobs, num_stages, machine_cnt_list, K}}
               (run 동안 eval λ·문제 크기는 불변이라 1회면 충분.)
      이후 줄: {"epoch": e, "makespan": [L,K], "quality": [L,K], "hvn": [K]}
               quality 는 hybrid helper 의 yield_h(=GT) 산출 → 학습 reward 와의 gap 가
               그대로 보임. hvn 은 같은 (ms, quality) 로 계산된 정규화 HV.
    """
    orig = train_ASIL.eval_pareto
    epoch_counter = [0]
    meta_written = [False]
    num_stages = len(machine_cnt_list)

    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    # run 시작 시 파일 초기화 (truncate). 이전 ablation run 의 점이 섞이지 않게.
    open(log_path, 'w').close()

    def patched(*args, **kwargs):
        kwargs['quality_helper'] = eval_helper
        ms_lam, q_lam, hv, nd, hv_norm = orig(*args, **kwargs)
        epoch_counter[0] += 1

        with open(log_path, 'a') as f:
            if not meta_written[0]:
                lambdas_t = kwargs.get('lambdas')
                meta = {'meta': {
                    'lambdas':          lambdas_t.detach().cpu().tolist(),
                    'num_jobs':         num_jobs,
                    'num_stages':       num_stages,
                    'machine_cnt_list': list(machine_cnt_list),
                    'K':                int(ms_lam.shape[1]),  # instances per lambda
                }}
                f.write(json.dumps(meta) + '\n')
                meta_written[0] = True

            record = {
                'epoch':    epoch_counter[0],
                'makespan': ms_lam.detach().cpu().tolist(),     # (L,K)
                'quality':  q_lam.detach().cpu().tolist(),      # (L,K) — GT(=qh_eval) 산출
                'hvn':      hv_norm.detach().cpu().tolist(),    # (K,)  — 정규화 HV per-instance
            }
            f.write(json.dumps(record) + '\n')

        return ms_lam, q_lam, hv, nd, hv_norm

    train_ASIL.eval_pareto = patched


def override_train_asil_anchors() -> None:
    """train_ASIL 모듈 global 의 HVN / raw HV / plot anchor 를 위 상단 상수로 덮어쓴다.

    train_ASIL.py 의 train() 안에서 eval_pareto/plot_pareto 호출 시 이 상수들을
    module-global 로 lookup → train() 진입 전에 한 번 갈아끼우면 그대로 반영.
    (reward scaling anchor 는 train() 의 함수 인자라 main() 에서 직접 넘김.)
    """
    train_ASIL.HVN_M_BEST  = HVN_M_BEST
    train_ASIL.HVN_M_WORST = HVN_M_WORST
    train_ASIL.HVN_Q_WORST = HVN_Q_WORST
    train_ASIL.HVN_Q_BEST  = HVN_Q_BEST
    train_ASIL.HV_REF_M    = HV_REF_M
    train_ASIL.HV_REF_Q    = HV_REF_Q
    train_ASIL.PLOT_M_MIN  = PLOT_M_MIN
    train_ASIL.PLOT_M_MAX  = PLOT_M_MAX
    train_ASIL.PLOT_Q_MIN  = PLOT_Q_MIN
    train_ASIL.PLOT_Q_MAX  = PLOT_Q_MAX


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num_jobs', type=int, default=25)
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7')
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--n_accum', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=50)
    p.add_argument('--pomo_size', type=int, default=64)
    p.add_argument('--eval_interval', type=int, default=1,
                   help='Ablation: 매 epoch HVN(paths_1 GT) 곡선용. 기본 학습은 20.')
    p.add_argument('--eval_batch_size', type=int, default=32 * 10)
    p.add_argument('--pareto_lambdas', type=int, default=32)
    # NOTE: reward anchor / HVN anchor / HV_REF / PLOT 은 모두 파일 상단 상수에서 설정.
    #       CLI 인자로 노출하지 않음 — ablation 들이 anchor 조합을 다르게 가져갈 거라
    #       파일 상단에서 한눈에 잡는 게 더 명확.
    p.add_argument('--ckpt', type=str,
                   default='checkpoints/ablation_p2train_p1eval.pt')
    # WandB 와 PNG scatter 둘 다 끈다 — 콘솔 로그 + raw HV best ckpt 만 남는다.
    args = p.parse_args()

    machines = [int(x) for x in args.machines.split(',')]
    num_stages = len(machines)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[ablation] train reward predictor = paths_{PATHS_TRAIN}  (inaccurate)")
    print(f"[ablation] eval features predictor = paths_{PATHS_TRAIN}  (same as train — no domain shift)")
    print(f"[ablation] eval  yield  predictor = paths_{PATHS_EVAL}   (GT for measurement)")
    print(f"[anchors]  reward : m=[{REWARD_M_BEST},{REWARD_M_WORST}] "
          f"q=[{REWARD_Q_WORST},{REWARD_Q_BEST}]")
    print(f"[anchors]  HVN    : m=[{HVN_M_BEST},{HVN_M_WORST}] "
          f"q=[{HVN_Q_WORST},{HVN_Q_BEST}]")
    print(f"[anchors]  HV_REF : (m={HV_REF_M}, q={HV_REF_Q})")

    qh_train = build_helper(PATHS_TRAIN, num_stages, device)
    qh_eval  = build_helper(PATHS_EVAL,  num_stages, device)
    qh_eval_hybrid = HybridQualityHelper(feat=qh_train, yield_h=qh_eval)

    override_train_asil_anchors()
    patch_eval_helper(qh_eval_hybrid, PARETO_POINTS_LOG,
                      num_jobs=args.num_jobs, machine_cnt_list=machines)
    print(f"[log] -> {PARETO_POINTS_LOG} (첫 줄 meta, 이후 줄 = epoch당 op_machine+makespan)")

    # 매 epoch eval 이라 PNG scatter 가 400 개 쌓이는 걸 막기 위해 plot_pareto 를 no-op 로.
    # rec.log_pareto 는 wandb_enabled=False 면 어차피 early-return 이라 PNG 경로 안 씀.
    train_ASIL.plot_pareto = lambda *_a, **_k: None
    # 모델 ckpt 저장도 비활성화 — ablation 이라 모델 보관 불필요 (디스크/시간 낭비).
    # train_ASIL.train() 안의 torch.save({...}, ckpt_path) 호출이 no-op 으로 처리됨.
    # 스크립트는 train 종료 후 바로 끝나므로 글로벌 패치라도 부수 효과 없음.
    torch.save = lambda *_a, **_k: None

    train_ASIL.train(
        num_epochs=args.epochs,
        n_accum=args.n_accum,
        batch_size=args.batch_size,
        pomo_size=args.pomo_size,
        num_jobs=args.num_jobs,
        machine_cnt_list=machines,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
        pareto_lambda_count=args.pareto_lambdas,
        hv_m_best=REWARD_M_BEST,
        hv_m_worst=REWARD_M_WORST,
        hv_q_best=REWARD_Q_BEST,
        hv_q_worst=REWARD_Q_WORST,
        ckpt_path=args.ckpt,
        wandb_enabled=False,
        wandb_project='hfsp-asil-ablation',
        wandb_run_name='p2train_p1eval',
        quality_helper=qh_train,
    )


if __name__ == "__main__":
    main()
