"""
HFSP multi-objective scheduler — train ASIL (Self-Imitation Learning) + POMO baseline.

목표 (makespan ↓ + yield ↑) 를 λ ∈ [0,1] 스칼라화로 묶어 λ-conditioned 정책 하나가 모든
trade-off 를 학습. eval 시 λ 를 0~1 로 sweep 하면 단일 모델에서 Pareto front 가 나옴.

Model: FFSPModel — MatNet bipartite attention.
  row = job nodes, col = machine nodes, edge feature = (proc_time, EST, EFT, quality_prior).
  decoder produces a joint softmax over (machine × job) edges and samples one per step.

Env adapter (HFSPGraphEnv → MatNet state):
  Env 는 op (job × stage) 단위지만 MatNet 은 job 단위로 본다. 매 step 각 job 의 "현재 active
  op" 의 feature 를 row 로 사용하고, 모든 stage 끝난 job 은 마지막 stage 의 op + is_done 플래그
  로 표기. action (machine, job) 이 선택되면 (machine, active_op_of_job) → env edge index 로
  변환해 env.step 호출.

Algorithm: ASIL — two-phase rollout per micro-rollout, gradient accumulation across micros.
  Phase 1: sampling rollout (no grad), B 인스턴스 × P 샘플 = BP 병렬 → trajectory 별 스칼라
           reward r = (1-λ)·m̂ + λ·q̂ 계산. (m̂, q̂ 는 [0,1] 정규화된 makespan/yield)
  Phase 2: 같은 instance 의 P 샘플 중 best (argmax r) 의 saved state 만 in-memory grad-replay 해 log-prob 수집.
           weight = (r_best - r_mean) / r_std  (instance-내 POMO baseline)
           loss   = -mean( weight · Σ_t log π(best trajectory) ) / n_accum

Epoch = 1 optimizer step.
  매 epoch 안에서 n_accum 회 micro-rollout → 각 micro 마다 fresh B λ 샘플 + loss/n_accum + backward.
  누적된 grad 로 clip + step 1번. 한 step 의 effective (scenario, λ) = n_accum × B.

Batch layout (per micro-rollout):
  B unique scenarios (proc_time + wafer_quality) × B unique λ, 1:1 매칭.
  P = POMO samples per (scenario, λ). 한 micro 의 BP = B·P instance 들이 병렬로 굴러감.
"""

from __future__ import annotations
import os
import time
import torch

from HFSPGraphEnv import HFSPGraphEnv, sample_random_actions
from HFSPWrapper import (
    sil_pomo_rollout, make_env_edge_lookup, get_feat_dims, get_feat_names,
    QualityHelper, eval_pareto, plot_pareto,
)
from FFSPModel import FFSPModel
from DataRecorder import DataRecorder


# =====================================================
# Default FFSPModel hyperparameters.
# embedding/head/qkv/ff_hidden 은 MatNet 논문 default 를 따른 값.
# ms_layer_init 은 attention bias projection 의 작은 초기화 상수 (재현용).
# =====================================================
def default_model_params(embedding_dim=128, head_num=8, qkv_dim=16,
                        encoder_layer_num=2, ff_hidden_dim=256,
                        logit_clipping=10):
    return {
        'embedding_dim': embedding_dim,
        'sqrt_embedding_dim': float(embedding_dim) ** 0.5,
        'encoder_layer_num': encoder_layer_num,
        'qkv_dim': qkv_dim,
        'sqrt_qkv_dim': float(qkv_dim) ** 0.5,
        'head_num': head_num,
        'logit_clipping': logit_clipping,
        'ff_hidden_dim': ff_hidden_dim,
        'eval_type': 'argmax',
        'ms_hidden_dim': 16,
        'ms_layer1_init': (1 / 2) ** 0.5,
        'ms_layer2_init': (1 / 16) ** 0.5,
        # CCO block
        'cco_num_ff_experts': 4,        # 4개의 FF 전문가
        'cco_top_k': 2,                 # 상위 2개의 전문가에게 라우팅
    }

# =====================================================
# Random baseline rollout → makespan/yield 정규화 anchor 추정.
# 학습과 동일한 B scenarios × P samples 구조의 BP 인스턴스를 만들고,
# 랜덤 정책으로 끝까지 굴린 결과의 (min, max) 를 정규화 분모로 사용. seed=0 으로 deterministic.
# =====================================================
def compute_random_baseline(env, quality_helper, B, P, seed=0):
    BP = B * P
    torch.manual_seed(seed)
    pt_unique = env.sample_edge_proc_time(B, 0, identical_job=True)       # (B, NE) torch
    pt_bp = pt_unique.repeat_interleave(P, dim=0)                          # (BP, NE)
    env.reset(seed=seed, batch_size=BP, edge_proc_time=pt_bp)
    wq_unique = quality_helper.sample_wafer_quality(B, env.num_jobs, seed, env.device)
    if wq_unique is not None:
        wq_bp = wq_unique.repeat_interleave(P, dim=0)
    else:
        wq_bp = None
    rand_obs = env._get_obs()
    rand_total = torch.zeros(BP, dtype=torch.long, device=env.device)
    # 총 step 수 = env.num_ops (J·S) 로 고정 — done.all() 동기화 sink 제거.
    T = env.num_ops
    for t in range(T):
        a = sample_random_actions(rand_obs['feasible_mask'])
        rand_obs, r, _ = env.step(a, last=(t == T - 1))
        rand_total += r
    rand_ms = (-rand_total).float()                                       # (BP,) torch
    rand_q  = quality_helper.compute_yield(env, wq_bp, aggregate="mean")  # (BP,) torch
    m_min = float(rand_ms.min().item())
    m_max = float(rand_ms.max().item())
    q_min = float(rand_q.min().item()) if quality_helper.is_active else 0.0
    q_max = float(rand_q.max().item()) if quality_helper.is_active else 1.0
    print(f"[random ref] ms mean={rand_ms.mean().item():.2f} min={m_min:.1f} max={m_max:.1f}")
    print(f"[random ref] q  mean={rand_q.mean().item():.4f} min={q_min:.4f} max={q_max:.4f}")
    return m_min, m_max, q_min, q_max


# =====================================================
# Train loop — ASIL + POMO baseline + gradient accumulation.
# 1 epoch = 1 optimizer step. 매 epoch 안에서 n_accum 회 micro-rollout → grad 누적 → step 1번.
# 한 step 의 effective (scenario, λ) 페어 = n_accum × B.
# eval_interval epoch 마다 λ sweep 으로 Pareto eval 수행, HV 갱신 시 ckpt 저장.
# =====================================================
def train(num_epochs=100,
          n_accum=5,
          batch_size=50,
          pomo_size=24,
          num_jobs=20,
          machine_cnt_list=(5, 3, 7, 3, 5, 7),
          time_low=3, time_high=22,
          lr=3e-4,
          weight_decay=0.1,
          adam_betas=(0.9, 0.95),
          warmup_ratio=0.1,
          eval_interval=10,
          eval_batch_size=256,
          pareto_lambda_count=32,
          hv_m_best=30.0,
          hv_m_ref=110.0,
          hv_q_best=1.0,
          hv_q_ref=0.90,
          ckpt_path='checkpoints/hfsp_pomo.pt',
          seed=1,
          est_slot_frac=0.3,
          wandb_enabled=True,
          wandb_project='hfsp-asil',
          wandb_run_name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    B = int(batch_size)
    P = int(pomo_size)
    BP = B * P

    # λ schedule
    # - 매 iter Uniform[0,1] 에서 B 개 새로 뽑아 (B,) instance-level λ 생성.
    # - B 시나리오 × B 람다 1:1 매칭 → 한 micro 안 모든 row 가 unique (scenario, λ) 페어.
    # - 같은 (scenario, λ) 안의 P 샘플로 POMO baseline variance 계산.
    print(f"[layout/accum] B={B} unique (scenario, λ) pairs × P={P} pomo samples")
    print(f"[layout/step]  n_accum={n_accum} → {n_accum*B} unique pairs per optimizer step")

    # ── EST curriculum ──
    # est_slot_frac in [0,1] → 처음 frac·num_epochs epoch 동안만 P 샘플 마지막 1개를
    # sample_est_v2_action 으로 대체 (BC 신호 주입). cutoff 이후 모델 자력 학습으로 전환.
    # None / False / 0 이면 처음부터 inject 안 함. 1.0 이면 학습 끝까지 켜짐.
    if est_slot_frac is None or est_slot_frac is False or float(est_slot_frac) <= 0.0:
        est_off_epoch = 0
    else:
        est_off_epoch = int(round(float(est_slot_frac) * num_epochs))
    if est_off_epoch <= 0:
        print(f"[inject] inject OFF for entire run (frac={est_slot_frac})")
    elif est_off_epoch >= num_epochs:
        print(f"[inject] inject ON for entire run "
              f"(frac={est_slot_frac}, total_epochs={num_epochs})")
    else:
        print(f"[inject] inject ON for epoch 1..{est_off_epoch} / {num_epochs}, OFF after "
              f"(frac={est_slot_frac})")

    env = HFSPGraphEnv(
        num_jobs=num_jobs, machine_cnt_list=list(machine_cnt_list),
        device=device,
        time_low=time_low, time_high=time_high,
    )
    # Quality prediction resources — separate from env.
    # env 는 -makespan 만 반환하고 yield 는 모름. episode 끝에 wrapper 가 완성된 op→machine
    # path + wafer_quality 를 helper.compute_yield 로 넘겨 sklearn pipeline 으로 예측 받음.
    quality_helper = QualityHelper(
        num_stages=env.num_stages,
        device=device,
        model_path="quality_results/Q_3/paths_1/best_hb_model.zip",
        wafer_path="quality_results/Q_3/paths_1/wafer_quality.json",
    )
    row_feat_dim, col_feat_dim, edge_feat_dim = get_feat_dims(env)
    print(f"[init] B={B}  P={P}  BP={BP}  "
          f"row_feat_dim={row_feat_dim}  col_feat_dim={col_feat_dim}  "
          f"edge_feat_dim={edge_feat_dim}  "
          f"NJ={env.num_jobs}  NM={env.total_machines}  NE={env.num_edges}  "
          f"device={device}")

    # Model / optimizer / LR scheduler init.
    model_params = default_model_params()
    model = FFSPModel(row_feat_dim, col_feat_dim, edge_feat_dim, **model_params).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=tuple(adam_betas), weight_decay=weight_decay,
    )
    # Linear warmup → linear decay (paper sec.):
    #   warmup_epochs = warmup_ratio · num_epochs (e.g. 0.1·E = 400 when E=4000).
    #   epoch ∈ [1, warmup_epochs]   → lr 선형 증가 0 → lr_peak
    #   epoch ∈ (warmup_epochs, E]   → lr 선형 감소 lr_peak → 0
    # scheduler.step() 은 매 epoch (= 1 optimizer step) 끝에 한 번 호출.
    warmup_epochs = max(1, int(round(warmup_ratio * num_epochs)))
    def _lr_lambda(step):
        # step 은 0-indexed (LambdaLR.last_epoch). 최초 호출 시 step=0 → lr = base_lr * lambda(0).
        e = step + 1
        if e <= warmup_epochs:
            return e / warmup_epochs
        progress = (e - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return max(0.0, 1.0 - progress)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,}")

    # (op, machine) → env edge idx lookup — 구조적 상수라 한 번만 만들어 매 step 재사용.
    env_edge_lookup_t = make_env_edge_lookup(env).to(device)

    # Separate eval env (same params) — 평가 도중 학습 env 의 batch_size/state 를 건드리지
    # 않도록 분리. 파라미터가 epoch 동안 불변이므로 한 번만 생성.
    eval_env = HFSPGraphEnv(
        num_jobs=num_jobs, machine_cnt_list=list(machine_cnt_list),
        device=device,
        time_low=time_low, time_high=time_high,
    )
    eval_lookup_t = make_env_edge_lookup(eval_env).to(device)

    # WandB recorder — logs loss / makespan / quality / grad norms / feature saliency / Pareto.
    # wandb_enabled=False 면 no-op stub.
    rec = DataRecorder(
        project=wandb_project, run_name=wandb_run_name, enabled=wandb_enabled,
        config=dict(num_epochs=num_epochs, n_accum=n_accum, step_size=n_accum*B,
                    B=B, P=P,
                    num_jobs=num_jobs, machines=list(machine_cnt_list),
                    time_low=time_low, time_high=time_high, lr=lr,
                    weight_decay=weight_decay, adam_betas=list(adam_betas),
                    warmup_ratio=warmup_ratio, warmup_epochs=warmup_epochs,
                    seed=seed, num_ops=env.num_ops,
                    hv_m_best=hv_m_best, hv_m_ref=hv_m_ref,
                    hv_q_best=hv_q_best, hv_q_ref=hv_q_ref,
                    n_params=n_params, **model_params),
        feat_names=get_feat_names(env),
    )

    # ── Random baseline 확인용 ──
    compute_random_baseline(env, quality_helper, B, P, seed=0)

    # ── HV reference point (fixed across all epochs, user-specified) ──
    # 2D hypervolume 의 ref point 는 epoch 간 HV 가 같은 좌표계에서 비교되도록 고정.
    # m_ref = "worst makespan" anchor, q_ref = "worst yield" anchor.
    lam_pool = torch.linspace(0.0, 1.0, 101, dtype=torch.float32, device=device)
    pareto_lambdas = torch.linspace(0.0, 1.0, int(pareto_lambda_count),
                                    dtype=torch.float32, device=device)
    best_eval = -float('inf')    # track best HV across epochs (higher = better)
    t00 = time.time()
    total_eval_time = 0.0        # cumulative eval wallclock, excluded from epoch time

    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        model.train()
        use_est_this_epoch = epoch <= est_off_epoch

        # Gradient accumulation: n_accum 회 micro-rollout → grad 누적 → 1번 step.
        # 매 micro 마다 fresh B λ 샘플 → 한 step 의 effective λ = n_accum × B.
        optimizer.zero_grad()
        loss_buf = []
        ms_avg_buf = []
        ms_best_buf = []
        q_avg_buf = []
        q_best_buf = []
        for ac in range(n_accum):
            chunk_seed = seed + epoch * n_accum + ac
            # chunk_seed 로 generator 시드 → proc_time/wafer_quality 와 같은 재현성 축.
            gen = torch.Generator(device=device).manual_seed(chunk_seed)

            # linspace(0,1,101) pool 에서 B 개 인덱스를 균등 추출 → (B,) instance-level λ.
            lam_idx = torch.randint(0, 101, (B,), generator=gen, device=device)
            lambdas_b = lam_pool[lam_idx]

            # λ-conditioned SIL rollout — batch layout: B unique (scenario, λ) × P pomo samples.
            # 같은 (scenario, λ) 안 P 샘플의 reward 분산으로 POMO baseline 계산.
            # bf16 autocast: forward activation 만 bf16 저장 → 메모리 ~50% 절감.
            # weights/grad/optimizer state 는 fp32 유지. 3090 (Ampere) 네이티브 지원, GradScaler 불필요.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                log_prob_sum_best, G_ms, G_q, r_bp = sil_pomo_rollout(
                    env, model, env_edge_lookup_t, device,
                    seed=chunk_seed, B=B, P=P, lambdas=lambdas_b,
                    quality_helper=quality_helper,
                    m_best=hv_m_best, m_worst=hv_m_ref,
                    q_best=hv_q_best, q_worst=hv_q_ref,
                    recorder=rec,
                    use_eft_slot=use_est_this_epoch,
                )
            # Shapes:
            #   log_prob_sum_best : (B,)   — Σ_t log π for each instance's best trajectory.
            #   G_ms              : (BP,)  — env return = -makespan (higher is better).
            #   G_q               : (BP,)  — predicted mean yield per sample.
            #   r_bp              : (B, P) — scalarized reward; P 샘플은 같은 (시나리오, λ).

            # ── ASIL weight — instance-내 POMO baseline ──
            # 같은 (시나리오, λ) 의 P 샘플들에 대해 (best - mean) / std 로 표준화.
            # 의미: 한 instance 안에서 두드러지게 좋은 trajectory 일수록 더 강하게 학습.
            # std 로 나누는 건 batch 안 λ 별 reward 스케일 차이를 완화하는 normalization.
            r_mean = r_bp.mean(dim=1)                               # (B,)
            r_best = r_bp.max(dim=1).values                         # (B,)
            r_std  = r_bp.std(dim=1)                                # (B,) unbiased
            weight = (r_best - r_mean) / (r_std + 1e-8)             # (B,)
            log_prob_avg = log_prob_sum_best / env.num_ops          # (B,)
            # /n_accum: 큰 batch 의 mean-loss 와 동등한 grad 가 되도록 미리 스케일.
            loss_micro = -(weight.detach() * log_prob_avg).mean() / n_accum

            loss_micro.backward()                                   # grad 누적 (+=)
            # 로깅엔 unscaled loss 전달 (/n_accum 은 backward 용 스케일일 뿐).
            loss_unscaled = loss_micro.detach() * n_accum
            rec.log_step(loss_unscaled, G_ms, G_q, r_bp, lambdas_b, log_prob_sum_best, weight)
            rec.collect_saliency()

            # ── Logging stats — accumulate on GPU, sync once at epoch end ──
            ms_bp = (-G_ms).view(B, P)
            q_bp  = G_q.view(B, P)
            loss_buf.append(loss_unscaled)
            ms_avg_buf.append(ms_bp.mean())
            ms_best_buf.append(ms_bp.min(dim=1).values.mean())
            q_avg_buf.append(q_bp.mean())
            q_best_buf.append(q_bp.max(dim=1).values.mean())

        # 누적된 grad 로 step 1번. log_gradients 는 clip 직전 = 누적 완료 시점에서 측정.
        rec.log_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        loss_mean = torch.stack(loss_buf).mean().item()
        avg_ms = torch.stack(ms_avg_buf).mean().item()
        best_ms = torch.stack(ms_best_buf).mean().item()
        avg_q  = torch.stack(q_avg_buf).mean().item()
        best_q = torch.stack(q_best_buf).mean().item()
        cur_lr = optimizer.param_groups[0]['lr']
        ct = time.time()

        train_elapsed = ct - t0
        cum_train_time = (ct - t00) - total_eval_time
        log_msg = (f"epoch {epoch:4d}  loss={loss_mean:+8.4f}  "
                   f"ms avg={avg_ms:6.2f} best={best_ms:6.2f}  "
                   f"q avg={avg_q:.4f} best={best_q:.4f}  "
                   f"({train_elapsed:6.2f}s/ep) ({cum_train_time / 60:6.2f}m)")
        print(log_msg)

        # ── Periodic Pareto evaluation ──
        # λ 를 linspace(0,1, pareto_lambda_count) 로 sweep, 각 λ greedy rollout → (ms, q) 평면
        # 위 점 모음. 평가 항목: HV (hypervolume), endpoint (λ=0, λ=1 의 평균), ND (Pareto
        # 비지배 점 개수), scatter plot. eval_pareto 는 모든 λ 를 한 번의 (L*B) batch 로 처리.
        if epoch % eval_interval == 0 or epoch == num_epochs:
            t_eval_start = time.time()
            model.eval()
            ms_lam, q_lam, hv, nd = eval_pareto(
                eval_env, model, eval_lookup_t, device,
                batch_size=eval_batch_size, lambdas=pareto_lambdas,
                quality_helper=quality_helper, seed=0,
                m_ref=hv_m_ref, q_ref=hv_q_ref,
            )
            hv_mean = float(hv.mean())
            ep_ms = float(ms_lam[0].mean())     # λ=0 endpoint — makespan-only objective
            ep_q  = float(q_lam[-1].mean())     # λ=1 endpoint — yield-only objective
            # λ-conditioning gap. 0 근처 → collapse (λ 무시), 음수 → anti-conditioning (λ 반대로 해석)
            # 품질을 포기하고 속도에 집중했을 때, 시간이 얼마나 덜 걸리는지를 보여줍니다.
            d_ms = float(ms_lam[-1].mean()) - ep_ms
            # 속도를 포기하고 품질에 집중했을 때, 품질을 얼마나 더 끌어올릴 수 있는지를 보여줍니다.
            d_q  = ep_q  - float(q_lam[0].mean())
            hv_std = float(hv.std())
            nd_mean = float(nd.float().mean())
            log_msg = (f"eval  {epoch:4d}  hv={hv_mean:.3f}±{hv_std:.3f} "
                        f"ms@0={ep_ms:.1f} q@1={ep_q:.3f} "
                        f"Δms={d_ms:+.2f} Δq={d_q:+.3f} nd={nd_mean:.1f}")

            scatter_path = f"quality_results/pareto_epoch_{epoch}.png"
            plot_pareto(ms_lam, q_lam, pareto_lambdas,
                        hv_m_best, hv_m_ref, hv_q_best, hv_q_ref,
                        scatter_path, title=f"epoch {epoch}  HV={hv_mean:.3f}")
            rec.log_pareto(hv_mean, hv_std, ep_ms, ep_q,
                           nd_mean, scatter_path,
                           d_ms=d_ms, d_q=d_q)

            if hv_mean > best_eval:
                best_eval = hv_mean
                log_msg += " ← best"

            os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'row_feat_dim': row_feat_dim,
                'col_feat_dim': col_feat_dim,
                'edge_feat_dim': edge_feat_dim,
                'model_params': model_params,
                'best_hv': best_eval,
                'hv_m_ref': hv_m_ref, 'hv_q_ref': hv_q_ref,
                'epoch': epoch,
            }, ckpt_path)

            eval_elapsed = time.time() - t_eval_start
            total_eval_time += eval_elapsed
            log_msg += f"  (eval {eval_elapsed:5.2f}s)"
            print(log_msg)

        rec.log_epoch(train_elapsed, cur_lr)
        rec.flush(step=epoch)

    rec.finish()
    return model

def _parse_est_frac(s):
    if s is None: return None
    sl = str(s).strip().lower()
    if sl in ('none', 'false', 'off', ''): return None
    return float(s)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num_jobs', type=int, default=25, help='Jobs per scheduling instance.')
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7',
                   help='Comma-separated machine counts per stage. len = num_stages.')
    p.add_argument('--epochs', type=int, default=200,
                   help='Total training epochs. 1 epoch = 1 optimizer step.')
    p.add_argument('--n_accum', type=int, default=1,
                   help='Gradient accumulation: micro-rollouts per optimizer step. '
                        '매 micro 마다 fresh B λ 샘플 → effective (scenario, λ) per step = n_accum × batch_size.')
    p.add_argument('--batch_size', type=int, default=50,
                   help='B = unique (scenario, λ) pairs per micro-rollout. '
                        'B 시나리오 × B 람다 1:1 매칭, 각 페어에 P pomo 샘플.')
    p.add_argument('--pomo_size', type=int, default=64,
                   help='P = POMO samples per (scenario, λ) — same instance, different trajectories.')
    p.add_argument('--eval_interval', type=int, default=10,
                   help='Pareto eval every N epochs (also forces eval at last epoch).')
    p.add_argument('--eval_batch_size', type=int, default=32*10,
                   help='Total instances in Pareto eval (LB). Must be divisible by --pareto_lambdas. ')
    p.add_argument('--pareto_lambdas', type=int, default=32,
                   help='Number of λ values for Pareto sweep (linspace(0,1,N)). '
                        'HV/scatter/endpoint 모두 이걸 사용.')
    p.add_argument('--hv_m_best', type=float, default=100.0,
                   help='Best (lowest) makespan anchor — reward 정규화 + Pareto plot x-min.')
    p.add_argument('--hv_m_ref', type=float, default=500.0,
                   help='Worst (highest) makespan anchor — HV ref + reward 정규화 + plot x-max.')
    p.add_argument('--hv_q_best', type=float, default=0.80,
                   help='Best (highest) yield anchor — reward 정규화 + Pareto plot y-max.')
    p.add_argument('--hv_q_ref', type=float, default=0.50,
                   help='Worst (lowest) yield anchor — HV ref + reward 정규화 + plot y-min.')
    p.add_argument('--ckpt', type=str, default='checkpoints/hfsp_base_quality.pt',
                   help='Path to save best-HV checkpoint.')
    p.add_argument('--est_slot', type=_parse_est_frac, default=0,
                   help='EST heuristic injection curriculum. '
                        'Float f in [0,1]: 처음 f·epochs 동안만 P 샘플 마지막 1개를 sample_est_v2_action 으로 대체, '
                        'cutoff 이후 모델 자력 학습. None/false/off/0: 처음부터 안 씀. 1.0: 학습 끝까지 inject. '
                        '초반 빠르게 휴리스틱 수준 도달 후 그 이상을 학습하도록 자연스럽게 handoff.')
    p.add_argument('--wandb_do', default=True,
                   help='Pass this flag to DISABLE WandB logging.')
    p.add_argument('--wandb_project', type=str, default='hfsp-asil-quality', help='WandB project name.')
    p.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name (auto if None).')
    args = p.parse_args()

    machines = [int(x) for x in args.machines.split(',')]

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
        hv_m_best=args.hv_m_best,
        hv_m_ref=args.hv_m_ref,
        hv_q_best=args.hv_q_best,
        hv_q_ref=args.hv_q_ref,
        ckpt_path=args.ckpt,
        est_slot_frac=args.est_slot,
        wandb_enabled=args.wandb_do,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
