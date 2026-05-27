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
# HV 평가 / Pareto 그림 전용 고정 앵커 — 하드코딩, CLI 인자 아님.
# reward 정규화(--hv_m_best/_worst, --hv_q_best/_worst)나 학습 loss/reward scale 에는
# 전혀 영향 없음.
#
# [1] HV_REF_* — raw HV(`hv`) 의 reference point (nadir = 최악 코너). raw HV 는 절대 단위라
#     epoch/실험 간 비교하려면 같은 값으로 고정해야 함 → 바꾸지 말 것 (바꾸면 과거 run 과 hv 비교 깨짐).
#     makespan 이 HV_REF_M 초과 또는 yield 가 HV_REF_Q 미만인 점 → raw HV 기여 0.
#     (best-ckpt 선택이 이 raw hv 기준. 원하면 plot 의 ref 마커로도 그대로 사용 가능.)
# [2] HVN_* — normalized HV(`hvN`) 박스. (ms,q) 를 4 corner 로 [0,1]² 매핑 후 계산하는 run 간
#     비교용 스케일. raw HV reference 와 완전 분리 — 박스 corner 를 여기서 직접 다 지정.
#     makespan: [HVN_M_BEST, HVN_M_WORST]  → [0,1]  (낮을수록 좋음)
#     yield:    [HVN_Q_WORST, HVN_Q_BEST]  → [0,1]  (높을수록 좋음)
#     ※ yield 분모(HVN_Q_BEST=1.0)는 HV_REF_Q(0.50)와 분리 — q>0.50 여도 hvN≤1 유지.
# [3] PLOT_* — Pareto scatter 그림의 축 범위. [1]/[2] HV 계산과 완전 독립, 순수 시각화용이라
#     마음대로 바꿔도 hv/hvN 숫자에는 영향 없음 (보기 좋은 범위로 자유롭게 조절).
#     makespan x축: [PLOT_M_MIN, PLOT_M_MAX] / yield y축: [PLOT_Q_MIN, PLOT_Q_MAX]
# =====================================================
HV_REF_M    = 600.0   # [1] raw HV reference makespan (worst corner)
HV_REF_Q    = 0.50    # [1] raw HV reference yield    (worst corner)
HVN_M_BEST  = 100.0   # [2] hvN makespan 정규화 best  (하한) # Table과 비교용
HVN_M_WORST = 600.0   # [2] hvN makespan 정규화 worst (상한)
HVN_Q_WORST = 0.0     # [2] hvN yield   정규화 worst  (하한)
HVN_Q_BEST  = 1.0     # [2] hvN yield   정규화 best   (상한)
PLOT_M_MIN  = 100.0   # [3] plot makespan x축 min
PLOT_M_MAX  = 550.0   # [3] plot makespan x축 max
PLOT_Q_MIN  = 0.50    # [3] plot yield    y축 min
PLOT_Q_MAX  = 0.85    # [3] plot yield    y축 max


# =====================================================
# Default FFSPModel hyperparameters.
# embedding/head/qkv/ff_hidden 은 MatNet 논문 default 를 따른 값.
# ms_layer_init 은 attention bias projection 의 작은 초기화 상수 (재현용).
# =====================================================
def default_model_params(embedding_dim=128, head_num=8, qkv_dim=16,
                        encoder_layer_num=2, ff_hidden_dim=256,
                        logit_clipping=5):
    return {
        'embedding_dim': embedding_dim,
        'sqrt_embedding_dim': float(embedding_dim) ** 0.5,
        'encoder_layer_num': encoder_layer_num,
        'qkv_dim': qkv_dim,
        'sqrt_qkv_dim': float(qkv_dim) ** 0.5,
        'head_num': head_num,
        'logit_clipping': logit_clipping,
        'ff_hidden_dim': ff_hidden_dim,
        'eval_type': 'argmax',          # 어디에도 영향을 주지 않는 기본값
        # 학습 sampling: softmax, 학습 eval: argmax, test: samples가 1이면 argmax, 이외엔 softmax  
        'ms_hidden_dim': 16,
        'ms_layer1_init': (1 / 2) ** 0.5,
        'ms_layer2_init': (1 / 16) ** 0.5,
        # CCO block
        'cco_num_ff_experts': 4,        # 4개의 FF 전문가
        'cco_top_k': 2,                 # 상위 2개의 전문가에게 라우팅
        'cco_gate_input_dim': 2 * embedding_dim,  # gate 입력 = [h_c ; λ_emb] concat (per-token 라우팅)
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
def train(num_epochs, n_accum, batch_size, pomo_size, 
          num_jobs, machine_cnt_list, 
          eval_interval, eval_batch_size, pareto_lambda_count, ckpt_path,
          hv_m_best, hv_m_worst, hv_q_best, hv_q_worst,
          wandb_enabled, wandb_project, wandb_run_name,
          time_low=3, time_high=22,
          lr=3e-4,
          weight_decay=0.1,
          adam_betas=(0.9, 0.95),
          warmup_ratio=0.1,
          seed=1,
          # ── 아래는 train_continual.py 용 선택 인자 (모두 default = 기존 동작 그대로) ──
          init_ckpt=None,          # 주어지면 그 ckpt 가중치로 warm-start (이어학습). 없으면 fresh init.
          quality_helper=None,     # 주어지면 그대로 사용(피처/보상 예측기 주입). 없으면 Q_3/paths_1 로드.
          epoch_ckpt_dir=None,     # 주어지면 매 save_interval 에폭마다 epoch_{e}.pt 저장 (학습곡선용).
          save_interval=0,         # 0 = 매-에폭 저장 안 함(기존 동작). 1 = 매 에폭.
          ):
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

    env = HFSPGraphEnv(
        num_jobs=num_jobs, machine_cnt_list=list(machine_cnt_list),
        device=device,
        time_low=time_low, time_high=time_high,
    )
    # Quality prediction resources — separate from env.
    # env 는 -makespan 만 반환하고 yield 는 모름. episode 끝에 wrapper 가 완성된 op→machine
    # path + wafer_quality 를 helper.compute_yield 로 넘겨 sklearn pipeline 으로 예측 받음.
    # quality_helper 가 외부 주입되면(이어학습: 피처/보상 예측기 교체) 그대로 쓰고,
    # 아니면 기존처럼 Q_3/paths_1 예측기를 로드한다.
    if quality_helper is None:
        quality_helper = QualityHelper(
            num_stages=env.num_stages,
            device=device,
            model_path="quality_results/Q_3/paths_1/best_hb_model.zip",
            wafer_path="quality_results/Q_3/paths_1/wafer_quality.json",
        )
    else:
        print(f"[init] quality_helper 주입됨 → {type(quality_helper).__name__} "
              f"(continual: 피처/보상 예측기 외부 지정)")
    row_feat_dim, col_feat_dim, edge_feat_dim = get_feat_dims(env)
    print(f"[init] B={B}  P={P}  BP={BP}  "
          f"row_feat_dim={row_feat_dim}  col_feat_dim={col_feat_dim}  "
          f"edge_feat_dim={edge_feat_dim}  "
          f"NJ={env.num_jobs}  NM={env.total_machines}  NE={env.num_edges}  "
          f"device={device}")

    # Model / optimizer / LR scheduler init.
    # init_ckpt 가 있으면 그 ckpt 의 아키텍처(model_params/dims)로 빌드 후 가중치 로드(warm-start).
    # 없으면 기존처럼 default_model_params() 로 fresh init.
    if init_ckpt:
        _ck = torch.load(init_ckpt, map_location=device, weights_only=False)
        model_params = _ck['model_params']
        model = FFSPModel(_ck['row_feat_dim'], _ck['col_feat_dim'],
                          _ck.get('edge_feat_dim', edge_feat_dim), **model_params).to(device)
        model.load_state_dict(_ck['model'])
        print(f"[init] warm-start <- {init_ckpt}  "
              f"(소스 epoch={_ck.get('epoch', '?')}, best_hv={_ck.get('best_hv', '?')})")
    else:
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

    # WandB recorder — logs loss / makespan / quality / grad norms / feature saliency /
    # policy entropy / CCO MoE routing health / Pareto.
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
                    hv_m_best=hv_m_best, hv_m_worst=hv_m_worst,
                    hv_q_best=hv_q_best, hv_q_worst=hv_q_worst,
                    hv_ref_m=HV_REF_M, hv_ref_q=HV_REF_Q,
                    n_params=n_params, **model_params),
        feat_names=get_feat_names(env),
    )

    # ── Random baseline 확인용 ──
    compute_random_baseline(env, quality_helper, B, P, seed=0)

    # ── HV reference point / 정규화 박스는 파일 상단 하드코딩 상수 사용 (HV_REF_*, HVN_*). ──
    lam_num = 101
    lam_pool = torch.linspace(0.0, 1.0, lam_num, dtype=torch.float32, device=device)
    pareto_lambdas = torch.linspace(0.0, 1.0, int(pareto_lambda_count),
                                    dtype=torch.float32, device=device)
    best_eval = -float('inf')    # track best HV across epochs (higher = better)
    t00 = time.time()
    total_eval_time = 0.0        # cumulative eval wallclock, excluded from epoch time

    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        model.train()

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
            lam_idx = torch.randint(0, lam_num, (B,), generator=gen, device=device)
            lambdas_b = lam_pool[lam_idx]
            # lambdas_b[0] = 0
            # lambdas_b[-1] = 1

            # λ-conditioned SIL rollout — batch layout: B unique (scenario, λ) × P pomo samples.
            # 같은 (scenario, λ) 안 P 샘플의 reward 분산으로 POMO baseline 계산.
            # bf16 autocast: forward activation 만 bf16 저장 → 메모리 ~50% 절감.
            # weights/grad/optimizer state 는 fp32 유지. 3090 (Ampere) 네이티브 지원, GradScaler 불필요.
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                log_prob_sum_best, G_ms, G_q, r_bp = sil_pomo_rollout(
                    env, model, env_edge_lookup_t, device,
                    seed=chunk_seed, B=B, P=P, lambdas=lambdas_b,
                    quality_helper=quality_helper,
                    m_best=hv_m_best, m_worst=hv_m_worst,
                    q_best=hv_q_best, q_worst=hv_q_worst,
                    recorder=rec,
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
            rec.collect_entropy()

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
        rec.log_moe(model)   # CCO 라우팅 헬스 — epoch 동안 Phase-2 forward 누적분 pop
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        # ── (옵션) 매 save_interval 에폭마다 ckpt 저장 — continual 학습곡선용 ──
        # 아래 eval-시점 best-HV 저장과 별개로, step 직후 가중치를 epoch_{e}.pt 로 남긴다.
        if epoch_ckpt_dir and save_interval > 0 and epoch % save_interval == 0:
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'row_feat_dim': row_feat_dim,
                'col_feat_dim': col_feat_dim,
                'edge_feat_dim': edge_feat_dim,
                'model_params': model_params,
                'epoch': epoch,
            }, os.path.join(epoch_ckpt_dir, f'epoch_{epoch}.pt'))

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
            ms_lam, q_lam, hv, nd, hv_norm = eval_pareto(
                eval_env, model, eval_lookup_t, device,
                batch_size=eval_batch_size, lambdas=pareto_lambdas,
                quality_helper=quality_helper, seed=0,
                m_ref=HV_REF_M, q_ref=HV_REF_Q,
                norm_m_best=HVN_M_BEST, norm_m_worst=HVN_M_WORST,
                norm_q_worst=HVN_Q_WORST, norm_q_best=HVN_Q_BEST,
            )
            hv_mean = float(hv.mean())
            hv_norm_mean = float(hv_norm.mean())
            hv_norm_std = float(hv_norm.std())
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
                        f"hvN={hv_norm_mean:.3f}±{hv_norm_std:.3f} "
                        f"ms@0={ep_ms:.1f} q@1={ep_q:.3f} "
                        f"Δms={d_ms:+.2f} Δq={d_q:+.3f} nd={nd_mean:.1f}")

            scatter_path = f"train_results/pareto_epoch_{epoch}.png"
            # === Pareto plot 축 = 파일 상단 PLOT_* 전역변수 (HV 계산과 독립, 자유 조절) ===
            PLOT_XLIM = (PLOT_M_MIN, PLOT_M_MAX)     # makespan x축
            PLOT_YLIM = (PLOT_Q_MIN, PLOT_Q_MAX)     # yield    y축
            PLOT_REF  = None             # 예: (HV_REF_M, HV_REF_Q) 로 raw HV nadir 마커 표시
            plot_pareto(ms_lam, q_lam, pareto_lambdas,
                        scatter_path,
                        title=f"epoch {epoch}  HV={hv_mean:.3f}  normHV={hv_norm_mean:.3f}",
                        xlim=PLOT_XLIM, ylim=PLOT_YLIM, ref_point=PLOT_REF)
            rec.log_pareto(hv_mean, hv_std, ep_ms, ep_q,
                           nd_mean, scatter_path,
                           hv_norm_mean, hv_norm_std,
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
                'hv_ref_m': HV_REF_M, 'hv_ref_q': HV_REF_Q,
                'epoch': epoch,
            }, ckpt_path)

            eval_elapsed = time.time() - t_eval_start
            total_eval_time += eval_elapsed
            log_msg += f" ({eval_elapsed:5.2f}s)"
            print(log_msg)

        # Reward 정규화 anchor — 현재 고정값이라 flat line. ms/mean·q/mean 수렴을
        # m_min/m_max·q_min/q_max 경계와 같은 차트에서 비교하기 위한 reference.
        # 인자 순서 (m_min, m_max, q_min, q_max) ← (best/낮은 ms, worst/높은 ms, worst/낮은 q, best/높은 q).
        rec.log_anchors(hv_m_best, hv_m_worst, hv_q_worst, hv_q_best)
        rec.log_epoch(train_elapsed, cur_lr)
        rec.flush(step=epoch)

    rec.finish()
    return model

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--num_jobs', type=int, default=25, help='Jobs per scheduling instance.')
    p.add_argument('--machines', type=str, default='5,3,7,3,5,7',
                   help='Comma-separated machine counts per stage. len = num_stages.')
    p.add_argument('--epochs', type=int, default=400,
                   help='Total training epochs. 1 epoch = 1 optimizer step.')
    p.add_argument('--n_accum', type=int, default=1,
                   help='Gradient accumulation: micro-rollouts per optimizer step. '
                        '매 micro 마다 fresh B λ 샘플 → effective (scenario, λ) per step = n_accum × batch_size.')
    p.add_argument('--batch_size', type=int, default=50,
                   help='B = unique (scenario, λ) pairs per micro-rollout. '
                        'B 시나리오 × B 람다 1:1 매칭, 각 페어에 P pomo 샘플.')
    p.add_argument('--pomo_size', type=int, default=64,
                   help='P = POMO samples per (scenario, λ) — same instance, different trajectories.')
    p.add_argument('--eval_interval', type=int, default=20,
                   help='Pareto eval every N epochs (also forces eval at last epoch).')
    p.add_argument('--eval_batch_size', type=int, default=32*10,
                   help='Total instances in Pareto eval (LB). Must be divisible by --pareto_lambdas. ')
    p.add_argument('--pareto_lambdas', type=int, default=32,
                   help='Number of λ values for Pareto sweep (linspace(0,1,N)). '
                        'HV/scatter/endpoint 모두 이걸 사용.')
    p.add_argument('--hv_m_best', type=float, default=100.0,
                   help='Best (lowest) makespan anchor — reward 정규화 전용.')
    p.add_argument('--hv_m_worst', type=float, default=500.0,
                   help='Worst (highest) makespan anchor — reward 정규화 전용.')
    p.add_argument('--hv_q_best', type=float, default=0.85,
                   help='Best (highest) yield anchor — reward 정규화 전용.')
    p.add_argument('--hv_q_worst', type=float, default=0.50,
                   help='Worst (lowest) yield anchor — reward 정규화 전용.')
    # HV reference point(--hv_ref_m/_q)는 CLI 인자 제거 — 파일 상단 HV_REF_M/HV_REF_Q 하드코딩.
    p.add_argument('--ckpt', type=str, default='checkpoints/hfsp_base_quality.pt',
                   help='Path to save best-HV checkpoint.')
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
        hv_m_worst=args.hv_m_worst,
        hv_q_best=args.hv_q_best,
        hv_q_worst=args.hv_q_worst,
        ckpt_path=args.ckpt,
        wandb_enabled=args.wandb_do,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
