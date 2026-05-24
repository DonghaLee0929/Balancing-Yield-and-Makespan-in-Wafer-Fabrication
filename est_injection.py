"""
EST injection (archived) — 학습에서 제거된 EST-휴리스틱 주입 로직 보관소.

지금 학습 파이프라인(train_ASIL.py / HFSPWrapper.sil_pomo_rollout)에는 더 이상
연결돼 있지 않다. 도움이 안 된다고 판단되어 제거했지만, 나중에 다시 시도할 수도 있어
핵심 코드만 한 파일로 모아 둔다.

[원래 아이디어]
POMO 의 P 샘플 중 마지막 1개(p == P-1)를 매 step EST-우선 휴리스틱
(HFSPGraphEnv.sample_est_action) 의 action 으로 강제 대체했다. Phase-2 grad-replay 가
(B, P) reward 의 argmax 슬롯만 추적하므로, 휴리스틱 trajectory 가 best 가 되는 동안
모델이 그 actions 을 imitate(BC)하고, 모델이 휴리스틱을 따라잡으면 best 슬롯이 다시
모델 쪽으로 옮겨가 smooth handoff 가 일어나는 구조였다.

`--est_slot f` (f∈[0,1]) 로 처음 f·num_epochs epoch 동안만 켜는 curriculum 도 있었다.
None/false/off/0 → 처음부터 OFF, 1.0 → 학습 끝까지 ON.

[다시 붙이는 법 — 요약]
  train_ASIL.py:
    - train() 시그니처에 est_slot_frac 인자 추가
    - 루프 진입 전:  est_off_epoch = est_off_epoch_from_frac(est_slot_frac, num_epochs)
    - 매 epoch:      use_est_this_epoch = epoch <= est_off_epoch
    - argparse:      p.add_argument('--est_slot', type=parse_est_frac, default=0, ...)
    - train() 호출:  est_slot_frac=args.est_slot
  HFSPWrapper.sil_pomo_rollout:
    - 시그니처에 use_eft_slot: bool = False 추가
    - 루프 전:  est_slot_mask = make_est_slot_mask(B, P, device) if use_eft_slot else None
    - 매 step (model() 직후, history 기록 직전):
          if est_slot_mask is not None:
              edge_selected = override_with_est(env, edge_selected, est_slot_mask)
"""

import torch

from HFSPGraphEnv import sample_est_action


# =====================================================
# Curriculum schedule helpers (원래 train_ASIL.py 에 있던 부분)
# =====================================================
def parse_est_frac(s):
    """--est_slot argparse 용 파서. None/false/off/'' → None, 그 외 → float."""
    if s is None:
        return None
    sl = str(s).strip().lower()
    if sl in ('none', 'false', 'off', ''):
        return None
    return float(s)


def est_off_epoch_from_frac(est_slot_frac, num_epochs, verbose=True):
    """est_slot_frac∈[0,1] → injection 을 켜 두는 마지막 epoch(est_off_epoch).

    epoch <= est_off_epoch 동안만 inject ON. 0 이면 처음부터 OFF.
    """
    if est_slot_frac is None or est_slot_frac is False or float(est_slot_frac) <= 0.0:
        est_off_epoch = 0
    else:
        est_off_epoch = int(round(float(est_slot_frac) * num_epochs))

    if verbose:
        if est_off_epoch <= 0:
            print(f"[inject] inject OFF for entire run (frac={est_slot_frac})")
        elif est_off_epoch >= num_epochs:
            print(f"[inject] inject ON for entire run "
                  f"(frac={est_slot_frac}, total_epochs={num_epochs})")
        else:
            print(f"[inject] inject ON for epoch 1..{est_off_epoch} / {num_epochs}, OFF after "
                  f"(frac={est_slot_frac})")
    return est_off_epoch


# =====================================================
# Rollout-side override (원래 HFSPWrapper.sil_pomo_rollout 에 있던 부분)
# =====================================================
def make_est_slot_mask(B, P, device):
    """(B*P,) bool mask — 각 (scenario, λ) 의 마지막 P 슬롯(p == P-1)만 True.

    BP layout 은 (B, P) row-major → bp_idx % P == P-1 이 마지막 슬롯.
    """
    return (torch.arange(B * P, device=device) % P) == (P - 1)


def override_with_est(env, edge_selected, est_slot_mask):
    """est_slot_mask 가 True 인 슬롯의 action 을 EST 휴리스틱 action 으로 대체.

    sample_est_action 은 env-edge idx 를 돌려주므로 model action 공간
    (edge_selected = machine_idx*num_jobs + job_idx) 으로 역변환 후 덮어쓴다.
    """
    est_env_edge = sample_est_action(env, {'feasible_mask': env.feasible_mask})
    est_machine_idx = env.edge_machine[est_env_edge]
    est_op_idx = env.edge_op[est_env_edge]
    est_job_idx = est_op_idx // env.num_stages
    est_action = est_machine_idx * env.num_jobs + est_job_idx
    return torch.where(est_slot_mask, est_action, edge_selected)
