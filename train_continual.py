"""
train_continual.py — 웨이퍼 *지속학습(continual)* 이어학습 스크립트.

(P3, Q3) 에서 학습한 λ-conditioned 정책을 분포가 바뀐 새 환경에서 이어 학습한다. train_ASIL.py
의 학습 루프(train) 를 *그대로 재사용* 하되, 아래만 바꿔 끼운다:

  1) warm-start   : --init_ckpt 의 가중치로 모델을 초기화(이어학습). 없으면 fresh init(=scratch).
  2) 품질 예측기   : 보상(compute_yield)·피처(compute_machine_quality) 예측기를 타깃 Q 로 교체.
  3) 작업시간 분포 : --time_low/--time_high 로 proc_time 샘플 범위 교체 (P3=넓게, P1=좁게).
  4) 매-에폭 평가  : 매 에폭 끝 hook 으로 학습 target 에서 즉시 eval → 점 (ms, yld, t) 를
                    test_results/comparison/continual_points_W{J}_Q{q}P{p}.jsonl 에 적재.
                    ckpt 는 저장하지 않음 — 학습곡선에 필요한 건 점 뿐.

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

매 에폭 적재된 점들은 experiment_continual.py 가 캐시 hit 으로 읽어 학습곡선을 그린다.
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

from HFSPWrapper import QualityHelper, make_env_edge_lookup
from HFSPGraphEnv import HFSPGraphEnv
from quality_augment import GroundTruthQuality, load_proc_time_augmented
from train_ASIL import train

# 매 epoch eval → jsonl 캐시 적재용. experiment_continual.py 의 슬러그·키·meta 컨벤션을
# *그대로* 따라야 후속 experiment_continual.py 가 cache hit 한다.
from experiment_continual import (
    MaskedQuality as _EvalMaskedQuality,
    model_front,
    _SWEEP_SLUG,
    _load_continual_cache,
    _append_eval_record,
    _save_continual_meta,
    _build_cur_metas,
)
from experiment_table import timed


# Eval target 의 (q_idx, paths_idx, p_idx) 는 args 에서 받아 학습 target 과 1:1 매치.
# 그 외 eval 설정은 experiment_continual.py 기본값과 일치해야 후속 cache hit.
_EVAL_NUM_LAMBDAS = 32
_EVAL_SAMPLES = 64
_EVAL_SEED = 0
_EVAL_YIELD_MODE = 'raw'
_EVAL_WQ_MIN = 0.99
_EVAL_WQ_MAX = 1.00
_EVAL_NSGA_POP = 100              # cur_metas['nsga'] 만들 때 정합용 (NSGA 자체는 안 돌림)
_EVAL_NSGA_GEN = 300
_EVAL_GA_SEED = 0
_EVAL_ANCHORS = (40.0, 180.0, 1.0, 0.0)    # experiment_continual.py 기본 --xlim/--ylim


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


def _resolve_row_slug(init_ckpt: str, feat_mode: str) -> tuple[str, str]:
    """이 학습 run 이 어느 비교 행에 해당하는지 → (row_name, slug)."""
    row = ('scratch' if not init_ckpt else
           'full-adapt' if feat_mode == 'full' else
           'stale-feat' if feat_mode == 'stale' else 'masked-feat')
    return row, _SWEEP_SLUG[row]


def _make_eval_hook(args, machines, num_stages, device):
    """매 epoch 모델을 학습 target (args.q_idx, args.paths_idx, args.p_idx) 에서 평가 →
    test_results/comparison/continual_points_W{J}_Q{q}P{p}.jsonl 에 즉시 적재.

    ckpt 안 만들고 점 (ms, yld, t) 만 저장 → 디스크 절약 + 후속 plot 즉시.
    experiment_continual.py 가 cache hit 으로 곡선 그릴 수 있게 같은 슬러그·meta 사용.

    jsonl 은 append-only 라 다른 row 의 옛 라인은 자동 보존 → 4개 row 따로 돌려도 캐시 누적.
    """
    row, slug = _resolve_row_slug(args.init_ckpt, args.feat_mode)
    q_idx, paths_idx, p_idx = args.q_idx, args.paths_idx, args.p_idx

    # ── 평가 인스턴스 (학습 target 과 동일) 한 번만 셋업 ──
    tgt_csv = f'quality_data/Q_{q_idx}/historical_paths_{paths_idx}.csv'
    gq_tgt = GroundTruthQuality(num_stages=num_stages, device=device, csv_path=tgt_csv,
                                machine_cnt_list=list(machines),
                                wafer_quality_min=_EVAL_WQ_MIN, wafer_quality_max=_EVAL_WQ_MAX)
    feat_helpers = {'target': gq_tgt, 'masked': _EvalMaskedQuality(args.mask_value)}
    if args.feat_mode == 'stale':
        src_csv = f'quality_data/Q_{args.src_q_idx}/historical_paths_{args.src_paths_idx}.csv'
        feat_helpers['stale'] = GroundTruthQuality(
            num_stages=num_stages, device=device, csv_path=src_csv,
            machine_cnt_list=list(machines),
            wafer_quality_min=_EVAL_WQ_MIN, wafer_quality_max=_EVAL_WQ_MAX)

    env = HFSPGraphEnv(num_jobs=args.num_jobs, machine_cnt_list=list(machines), device=device)
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)
    proc = load_proc_time_augmented(f'quality_data/P_{p_idx}.csv',
                                    args.num_jobs, list(machines))
    wq_1 = gq_tgt.sample_wafer_quality(B=1, num_jobs=args.num_jobs,
                                        seed=_EVAL_SEED, device=device)

    # eval 시 정책이 보는 피처 = 학습 조건과 동일
    eval_feat_key = ('target' if args.feat_mode == 'full' else
                     'stale' if args.feat_mode == 'stale' else 'masked')
    qh = feat_helpers[eval_feat_key]

    # ── 캐시 로드 + cur_metas. append-only 라 옛 라인은 자동 보존 (머지 로직 불필요) ──
    cache_data, file_meta = _load_continual_cache(args.num_jobs, q_idx, p_idx)
    new_artifacts = dict(file_meta.get('artifacts', {}))

    # experiment_continual.py 의 _build_cur_metas 가 args.* 로 접근하므로 같은 인터페이스 namespace
    class _A:
        num_jobs = args.num_jobs
        q_idx = args.q_idx
        p_idx = args.p_idx
        src_q_idx = args.src_q_idx
        src_paths_idx = args.src_paths_idx
        mask_value = args.mask_value
        num_lambdas = _EVAL_NUM_LAMBDAS
        samples = _EVAL_SAMPLES
        seed = _EVAL_SEED
        yield_mode = _EVAL_YIELD_MODE
        nsga_pop = _EVAL_NSGA_POP
        nsga_gen = _EVAL_NSGA_GEN
        ga_seed = _EVAL_GA_SEED
    cur_metas = _build_cur_metas(args=_A, machines=list(machines),
                                 wq_min=_EVAL_WQ_MIN, wq_max=_EVAL_WQ_MAX,
                                 anchors=_EVAL_ANCHORS)

    print(f"[continual-eval-hook] row={row}, slug={slug}  -> "
          f"(Q{q_idx}, P{p_idx}) paths_{paths_idx}  "
          f"λ×{_EVAL_NUM_LAMBDAS}, s={_EVAL_SAMPLES}, seed={_EVAL_SEED}, "
          f"feat={eval_feat_key}")
    print(f"[continual-eval-hook] 옛 jsonl 점 {len(cache_data) // 3}개 로드 (append-only 라 자동 보존)")

    def _hook(epoch, model):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                (ms, yld), t = timed(lambda: model_front(
                    model, env, env_edge_lookup_t, device, qh, gq_tgt, proc, wq_1,
                    _EVAL_NUM_LAMBDAS, _EVAL_SAMPLES, _EVAL_SEED, _EVAL_YIELD_MODE), device)
        finally:
            if was_training:
                model.train()

        # jsonl 한 줄 append — 매 epoch I/O 가 그 한 줄 뿐. 전체 rewrite 0.
        try:
            _append_eval_record(args.num_jobs, q_idx, p_idx, paths_idx, slug, epoch, ms, yld, t)
        except Exception as e:
            print(f"[warn] jsonl append 실패 (epoch={epoch}): {e}")

        # ckpt 파일이 없으므로 sentinel artifact — experiment_continual.py 의 _cached_or_eval 이
        # cur_sig.mtime is None 분기로 자동 신뢰함 (캐시 hit).
        new_artifacts[f'{slug}_e{epoch}'] = {
            'path': f'<train-direct: {row} epoch={epoch}>',
            'mtime': None, 'size': None}
        try:
            _save_continual_meta(args.num_jobs, q_idx, p_idx, cur_metas, new_artifacts)
        except Exception as e:
            print(f"[warn] meta 저장 실패 (epoch={epoch}): {e}")

    return _hook


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
                    "타깃 기본 Q1, 작업시간 {3,4,5,6}. 매 에폭 eval → jsonl 캐시 적재.")
    # warm-start.
    p.add_argument('--init_ckpt', type=str, default='',
                   help="이어학습 시작 ckpt (소스 (P3,Q3) 정책). 비우면 fresh init(=scratch).")
    # 인스턴스 구조.
    p.add_argument('--num_jobs', type=int, default=25)
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7')
    # 타깃 품질 예측기 (보상 + full 피처). source = Q3 path 1 (=baseline 학습 환경) 와 *다른*
    # path 를 골라야 진짜 분포 변화 (machine 상대성 차이) 가 나옴.
    p.add_argument('--q_idx', type=int, default=2, choices=[1, 2, 3],
                   help="타깃 품질 시나리오 → 보상(및 full 피처) 예측기 Q_{q}/paths_{p}.")
    p.add_argument('--p_idx', type=int, default=2, choices=[1, 2, 3],
                   help="**eval 전용** target proc_time 인스턴스 (학습엔 안 쓰임 — 학습은 "
                        "torch.randint[time_low, time_high) 랜덤). 매 epoch eval hook 의 P CSV.")
    p.add_argument('--paths_idx', type=int, default=10,
                   help="타깃 예측기 paths 인덱스. 기본 10 (source path=1 과 분리). "
                        "1 로 두면 Q3p1≈Q1p1 이라 stale↔full 격차 안 나옴 — 2026-05-29 발견.")
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
    p.add_argument('--time_high', type=int, default=15,
                   help="작업시간 상한(배타적). 작업시간 ∈ {3,4,...,14}. "
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
    # reward 정규화 anchor (train_ASIL __main__ 기본값과 동일).
    p.add_argument('--hv_m_best', type=float, default=100.0)
    p.add_argument('--hv_m_worst', type=float, default=500.0)
    p.add_argument('--hv_q_best', type=float, default=0.85)
    p.add_argument('--hv_q_worst', type=float, default=0.50)
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

    row, slug = _resolve_row_slug(args.init_ckpt, args.feat_mode)
    proc_set = list(range(args.time_low, args.time_high))
    print(f"[continual] device={device}  W={args.num_jobs}  machines={machines}")
    print(f"[continual] init={'(fresh/scratch)' if not args.init_ckpt else args.init_ckpt}")
    print(f"[continual] 행={row}  슬러그={slug}  feat_mode={args.feat_mode}"
          + (f"  (피처=Q{args.src_q_idx}/paths_{args.src_paths_idx})" if args.feat_mode == 'stale'
             else f"  (피처 상수={args.mask_value})" if args.feat_mode == 'masked' else ""))
    print(f"[continual] 타깃 보상 예측기=Q{args.q_idx}/paths_{args.paths_idx}")
    print(f"[continual] 작업시간 ∈ {proc_set}  (time_low={args.time_low}, "
          f"time_high={args.time_high} 배타)")
    print(f"[continual] epochs={args.epochs}  (ckpt 저장 안 함 — 매 epoch eval 점만 캐시 적재)")

    # 매 epoch 끝에 eval → continual_points_W{J}_Q{q}P{p}.jsonl 에 직접 적재.
    eval_hook = _make_eval_hook(args, machines, num_stages, device)

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
        ckpt_path=None,                  # best_hv.pt 저장 안 함 (점만 캐시)
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
        post_epoch_hook=eval_hook,       # 매 에폭 eval → jsonl 캐시 적재
    )
    print("\n=== continual training done ===")


if __name__ == "__main__":
    main()
