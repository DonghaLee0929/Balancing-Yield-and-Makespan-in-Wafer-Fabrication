"""
train_continual.py — 웨이퍼 *지속학습(continual)* 이어학습 스크립트.

(P3, Q3) 에서 학습한 λ-conditioned 정책을 분포가 바뀐 새 환경에서 이어 학습한다. train_ASIL.py
의 학습 루프(train) 를 *그대로 재사용* 하되, 아래만 바꿔 끼운다:

  1) warm-start   : --init_ckpt 의 가중치로 모델을 초기화(이어학습). 없으면 fresh init(=scratch).
  2) 품질 예측기   : 보상(compute_yield)·피처(compute_machine_quality) 예측기를 타깃 Q 로 교체.
  3) 작업시간 분포 : --time_low/--time_high 로 proc_time 샘플 범위 교체 (P3=넓게, P1=좁게).
  4) 매-에폭 저장  : --save_dir 에 epoch_{e}.pt 를 매 에폭 저장 → 적응 '효율' 학습곡선용.

⚠ 이 레포의 학습은 proc_time 을 *랜덤 샘플*(torch.randint[low, high), high 배타적) 로 만든다.
   P1/P2/P3 CSV 는 eval 전용 — 학습엔 안 쓰인다. 따라서 '(P1) 으로 이어학습' = proc 샘플 범위를
   P1 분포로 바꾸는 것. 기본값 time_low=3, time_high=7 → 작업시간 ∈ {3,4,5,6}.
   (소스 P3 분포는 train_ASIL 기본 time_low=3, time_high=22 → {3,…,21}.)

피처/보상 예측기 분리 (4개 비교군 학습용):
   sil_pomo_rollout 의 quality_helper 가 피처(compute_machine_quality)·보상(compute_yield)·
   wafer_quality 를 모두 공급한다. CompositeQuality 로 *피처만* 다른 예측기로 라우팅해, 롤아웃
   코드 수정 없이 아래 조건을 학습한다 (보상·wafer_quality 는 항상 타깃 Q):
     --feat_mode full   : 피처=타깃 Q        (보상=타깃 Q)  → full-adapt (제안)
     --feat_mode stale  : 피처=옛 Q(--src_q_idx, 기본 Q3)  → stale-feat
     --feat_mode masked : 피처=상수(--mask_value, 기본 -1) → masked-feat
   여기에 --init_ckpt 유무로 scratch / 이어학습을 가른다:
     init 없음 + full  = scratch (타깃에서 처음부터)
     init 있음 + full  = full-adapt
     init 있음 + stale = stale-feat
     init 있음 + masked= masked-feat

학습된 epoch_{e}.pt 들은 experiment_continual.py 로 (ground-truth) 평가해 학습곡선을 그린다.
"""

from __future__ import annotations
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch

from HFSPWrapper import QualityHelper
from train_ASIL import train


# =====================================================
# 피처/보상 예측기 분리용 래퍼
# =====================================================
class MaskedQuality:
    """피처(compute_machine_quality)만 상수(mask_value)로 내보내는 feature helper.

    build_model_state 가 유효 엣지에 mask_value 를 채우고, 무효 엣지는 어차피 -1 패딩되므로
    mask_value=-1 이면 '피처 전부 -1 마스킹' 과 일치. (보상/채점엔 관여하지 않음.)
    """

    def __init__(self, mask_value: float = -1.0):
        self.mask_value = float(mask_value)

    def compute_machine_quality(self, env) -> torch.Tensor:
        return torch.full((env.batch_size, env.num_jobs, env.total_machines),
                          self.mask_value, dtype=torch.float32, device=env.device)


class CompositeQuality:
    """피처는 feature_helper, 보상·wafer_quality 는 reward_helper 로 라우팅하는 합성 helper.

    sil_pomo_rollout / eval_pareto / compute_random_baseline 가 호출하는 메서드만 위임한다:
      compute_machine_quality → feature_helper   (정책이 보는 엣지 피처)
      compute_yield / sample_wafer_quality / is_active / wafer_quality_* → reward_helper
    → 롤아웃 코드 수정 없이 '피처 예측기 ≠ 보상 예측기' 조건을 학습할 수 있다.
    """

    def __init__(self, feature_helper, reward_helper):
        self.feature_helper = feature_helper
        self.reward_helper = reward_helper

    @property
    def is_active(self) -> bool:
        return self.reward_helper.is_active

    @property
    def wafer_quality_min(self):
        return self.reward_helper.wafer_quality_min

    @property
    def wafer_quality_max(self):
        return self.reward_helper.wafer_quality_max

    def sample_wafer_quality(self, B, num_jobs, seed, device):
        return self.reward_helper.sample_wafer_quality(B, num_jobs, seed, device)

    def compute_yield(self, env, wafer_quality, aggregate="mean"):
        return self.reward_helper.compute_yield(env, wafer_quality, aggregate)

    def compute_machine_quality(self, env):
        return self.feature_helper.compute_machine_quality(env)


def _make_quality_helper(num_stages, device, q_idx, paths_idx) -> QualityHelper:
    """quality_results/Q_{q}/paths_{p}/ 의 예측모델(.zip)+wafer_quality.json 로 QualityHelper."""
    base = f"quality_results/Q_{q_idx}/paths_{paths_idx}"
    qh = QualityHelper(num_stages=num_stages, device=device,
                       model_path=f"{base}/best_hb_model.zip",
                       wafer_path=f"{base}/wafer_quality.json")
    if not qh.is_active:
        raise FileNotFoundError(f"예측모델 로드 실패 — {base}/best_hb_model.zip 확인")
    return qh


# =====================================================
# Entry
# =====================================================
def main():
    p = argparse.ArgumentParser(
        description="웨이퍼 지속학습 이어학습 (train_ASIL.train 재사용). "
                    "타깃 기본 Q1, 작업시간 {3,4,5,6}. 매 에폭 ckpt 저장.")
    # warm-start. # checkpoints/0523_baseline.pt
    p.add_argument('--init_ckpt', type=str, default='checkpoints/0523_baseline.pt',
                   help="이어학습 시작 ckpt (소스 (P3,Q3) 정책). 비우면 fresh init(=scratch).")
    # 인스턴스 구조.
    p.add_argument('--num_jobs', type=int, default=25)
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7')
    # 타깃 품질 예측기 (보상 + full 피처).
    p.add_argument('--q_idx', type=int, default=1, choices=[1, 2, 3],
                   help="타깃 품질 시나리오 → 보상(및 full 피처) 예측기 Q_{q}/paths_{p}.")
    p.add_argument('--paths_idx', type=int, default=1, help="타깃 예측기 paths 인덱스.")
    # 피처 예측기 조건.
    p.add_argument('--feat_mode', type=str, default='full',
                   choices=['full', 'stale', 'masked'],
                   help="full=피처도 타깃Q / stale=피처는 옛Q(--src_q_idx) / masked=피처 상수.")
    p.add_argument('--src_q_idx', type=int, default=3, choices=[1, 2, 3],
                   help="stale 피처용 옛(소스) 품질 시나리오. 기본 Q3.")
    p.add_argument('--src_paths_idx', type=int, default=1, help="stale 피처용 paths 인덱스.")
    p.add_argument('--mask_value', type=float, default=-1.0,
                   help="masked 피처 상수. 기본 -1 (학습 시 -1 마스킹).")
    # 작업시간(proc_time) 분포 — torch.randint[low, high), high 배타적.
    p.add_argument('--time_low', type=int, default=3,
                   help="작업시간 하한(포함). 기본 3.")
    p.add_argument('--time_high', type=int, default=7,
                   help="작업시간 상한(배타적). 기본 7 → 작업시간 ∈ {3,4,5,6}. "
                        "(소스 P3 분포는 train_ASIL 기본 22 → {3..21}.)")
    # 학습 하이퍼파라미터.
    p.add_argument('--epochs', type=int, default=100, help="이어학습 에폭 수.")
    p.add_argument('--n_accum', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=50)
    p.add_argument('--pomo_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=1)
    # 평가/저장.
    p.add_argument('--eval_interval', type=int, default=20)
    p.add_argument('--eval_batch_size', type=int, default=320,
                   help="Pareto eval 총 인스턴스 수. --pareto_lambdas 로 나눠떨어져야 함.")
    p.add_argument('--pareto_lambdas', type=int, default=32)
    p.add_argument('--save_interval', type=int, default=1,
                   help="매 N 에폭마다 epoch_{e}.pt 저장. 기본 1 = 매 에폭.")
    p.add_argument('--save_dir', type=str, default='',
                   help="epoch_{e}.pt 저장 디렉터리. 비우면 자동 명명.")
    p.add_argument('--hv_m_best', type=float, default=40.0)
    p.add_argument('--hv_m_worst', type=float, default=160.0)
    p.add_argument('--hv_q_best', type=float, default=0.97)
    p.add_argument('--hv_q_worst', type=float, default=0.85)
    # WandB (기본 off — 짧은 이어학습 다회 실행 편의).
    p.add_argument('--wandb_do', default=False, help="truthy 면 WandB 로깅 활성.")
    p.add_argument('--wandb_project', type=str, default='hfsp-continual')
    p.add_argument('--wandb_run_name', type=str, default=None)
    args = p.parse_args()

    machines = [int(x) for x in args.machines.split(',')]
    num_stages = len(machines)
    if args.time_high <= args.time_low:
        raise SystemExit(f"--time_high({args.time_high}) > --time_low({args.time_low}) 이어야 함")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 보상 예측기 = 타깃 Q ──
    reward_helper = _make_quality_helper(num_stages, device, args.q_idx, args.paths_idx)

    # ── 피처 예측기 (feat_mode) → quality_helper 합성 ──
    if args.feat_mode == 'full':
        quality_helper = reward_helper                       # 피처=보상=타깃 Q
    elif args.feat_mode == 'stale':
        feature_helper = _make_quality_helper(
            num_stages, device, args.src_q_idx, args.src_paths_idx)
        quality_helper = CompositeQuality(feature_helper, reward_helper)
    else:  # masked
        quality_helper = CompositeQuality(MaskedQuality(args.mask_value), reward_helper)

    # ── 저장 경로 ──
    init_tag = 'adapt' if args.init_ckpt else 'scratch'
    save_dir = args.save_dir or (
        f'checkpoints/continual_{init_tag}_{args.feat_mode}_Q{args.q_idx}_t{args.time_low}-{args.time_high}')
    ckpt_path = os.path.join(save_dir, 'best_hv.pt')         # eval-시점 best-HV 저장(추가)

    proc_set = list(range(args.time_low, args.time_high))
    print(f"[continual] device={device}  W={args.num_jobs}  machines={machines}")
    print(f"[continual] init={'(fresh/scratch)' if not args.init_ckpt else args.init_ckpt}")
    print(f"[continual] 타깃 보상 예측기=Q{args.q_idx}/paths_{args.paths_idx}  "
          f"feat_mode={args.feat_mode}"
          + (f" (피처=Q{args.src_q_idx}/paths_{args.src_paths_idx})" if args.feat_mode == 'stale'
             else f" (피처 상수={args.mask_value})" if args.feat_mode == 'masked' else ""))
    print(f"[continual] 작업시간 ∈ {proc_set}  (time_low={args.time_low}, time_high={args.time_high} 배타)")
    print(f"[continual] epochs={args.epochs}  매 {args.save_interval} 에폭 저장 → {save_dir}/epoch_*.pt")

    train(
        num_epochs=args.epochs,
        n_accum=args.n_accum,
        batch_size=args.batch_size,
        pomo_size=args.pomo_size,
        num_jobs=args.num_jobs,
        machine_cnt_list=machines,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
        pareto_lambda_count=args.pareto_lambdas,
        ckpt_path=ckpt_path,
        hv_m_best=args.hv_m_best,
        hv_m_worst=args.hv_m_worst,
        hv_q_best=args.hv_q_best,
        hv_q_worst=args.hv_q_worst,
        wandb_enabled=args.wandb_do,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        time_low=args.time_low,
        time_high=args.time_high,
        lr=args.lr,
        seed=args.seed,
        # ── continual 전용 주입 ──
        init_ckpt=args.init_ckpt or None,
        quality_helper=quality_helper,
        epoch_ckpt_dir=save_dir,
        save_interval=args.save_interval,
    )
    print("\n=== continual training done ===")


if __name__ == "__main__":
    main()
