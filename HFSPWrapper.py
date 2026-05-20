"""
HFSP Graph RL wrapper — PyTorch Vectorized Edition
MatNet bipartite attention + multi-objective SIL.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import os
import torch
import torch.nn.functional as F
from hummingbird.ml import load

from HFSPGraphEnv import HFSPGraphEnv, sample_est_action
from FFSPModel import FFSPModel

# =====================================================
# Feature channels
# =====================================================
def get_feat_names(env: HFSPGraphEnv) -> dict:
    S = env.num_stages
    return {
        'row':  [f'active_stage_s{s}' for s in range(S)] + ['active_stage_done', 'remaining_proc', 'wafer_quality'],
        'col':  [f'stage_id_s{s}' for s in range(S)] + ['idle', 'time_to_idle'],
        'edge': ['proc_time', 'est', 'eft', 'quality_zscore'],
    }

def get_feat_dims(env: HFSPGraphEnv) -> tuple[int, int, int]:
    n = get_feat_names(env)
    return len(n['row']), len(n['col']), len(n['edge'])

# =====================================================
# Lightweight container
# =====================================================
@dataclass
class ModelState:
    BATCH_IDX: torch.Tensor        
    row_feature: torch.Tensor      
    col_feature: torch.Tensor      
    edge_feature: torch.Tensor     
    edge_mask: torch.Tensor        
    finished: torch.Tensor         
    lambdas: torch.Tensor          

# =====================================================
# Quality helper (Hummingbird PyTorch 사이킷런/판다스 추론 모델 래핑)
# =====================================================
class QualityHelper:
    def __init__(self, num_stages: int, device,
                 model_path: str = "quality_results/Q_3/paths_1/best_hb_model.zip",
                 wafer_path: str = "quality_results/Q_3/paths_1/wafer_quality.json"):
        self.num_stages = int(num_stages)
        self.quality_model       = None
        
        self._state_offsets_t = None
        self._stride_matrix_t = None
        self._full_table_t    = None
        self._done_idx        = None

        if not (os.path.exists(model_path) and os.path.exists(wafer_path)):
            print(f"[quality] 파일 없음 → 비활성화 ({model_path}, {wafer_path})")
            return

        with open(wafer_path, "r") as f:
            stats = json.load(f)

        self.step_cols          = list(stats["step_cols"])
        # JSON에 저장된 기계 개수 로드 (저장본 없으면 기본값 사용)
        self.machine_cnt_list   = stats.get("machine_cnt_list", [5,3,7,3,5,7]) 
        # 실제 파일의 분포로 샘플링하고 싶을 경우
        # self.wafer_quality_col  = stats["wafer_quality_col"]
        # _samples = stats["wafer_quality"]["samples"]
        # self.wafer_quality_min  = float(min(_samples))
        # self.wafer_quality_max  = float(max(_samples))
        self.wafer_quality_min  = 0.99
        self.wafer_quality_max  = 1.00

        if len(self.step_cols) != self.num_stages:
            print(f"[quality] step_cols={len(self.step_cols)} != num_stages={self.num_stages} → 비활성화")
            self.quality_model = None
            return
        
        # Hummingbird 모델 로드 및 GPU 할당
        self.quality_model = load(model_path, override_flag=True)
        self.quality_model.to(device)

        print(f"[quality] loaded {stats['model_name']} (Hummingbird PyTorch), "
              f"wafer_q range=[{self.wafer_quality_min:.4f}, {self.wafer_quality_max:.4f}]")

    @property
    def is_active(self) -> bool:
        return self.quality_model is not None

    def sample_wafer_quality(self, B: int, num_jobs: int, seed: int, device) -> torch.Tensor | None:
        """U(wafer_quality_min, wafer_quality_max) 에서 (B, num_jobs) 샘플. torch on device."""
        if not self.is_active:
            return None
        g = torch.Generator(device=device).manual_seed(int(seed))
        return torch.empty((B, num_jobs), dtype=torch.float32, device=device).uniform_(
            self.wafer_quality_min, self.wafer_quality_max, generator=g)

    def compute_yield(self, env, wafer_quality: torch.Tensor | None, aggregate: str = "mean") -> torch.Tensor:
        B, J, S = env.batch_size, env.num_jobs, env.num_stages
        zeros_bj = torch.zeros((B, J), dtype=torch.float32, device=env.device)
        zeros_b = torch.zeros(B, dtype=torch.float32, device=env.device)

        if not self.is_active or wafer_quality is None:
            return zeros_bj if aggregate == "none" else zeros_b

        completed = env.op_assigned.all(dim=1)
        if not completed.any():
            return zeros_bj if aggregate == "none" else zeros_b

        # Pandas, Numpy 연산 전면 제거 -> 100% 순수 GPU 텐서 파이프라인
        op_m_done = env.op_machine[completed].view(-1, J, S)
        local_m = env.machine_local[op_m_done.clamp(min=0)]  # (N_done, J, S)
        local_m_flat = local_m.view(-1, S) # (N_done * J, S)
        wq_flat = wafer_quality[completed].view(-1, 1) # (N_done * J, 1)

        # PyTorch F.one_hot을 이용한 범주형 인코딩
        one_hots = []
        for s, cnt in enumerate(self.machine_cnt_list):
            oh = F.one_hot(local_m_flat[:, s], num_classes=cnt).float()
            one_hots.append(oh)
            
        X_tensor = torch.cat(one_hots + [wq_flat], dim=1)

        with torch.no_grad():
            # 내부 torch 모듈 직접 호출 → numpy 변환 없이 GPU 텐서 그대로 받음.
            # .predict() 는 항상 numpy 로 변환해 반환하므로 사용 금지.
            preds = self.quality_model.model(X_tensor)
            if isinstance(preds, tuple):
                preds = preds[0]  # classifier 인 경우 (pred_label, proba) — pred_label 사용
            preds = preds.float()

        out_bj = zeros_bj.clone()
        out_bj[completed] = preds.view(-1, J)
        
        if aggregate == "none": return out_bj
        if aggregate == "sum": return out_bj.sum(dim=1)
        return out_bj.mean(dim=1)

    def prepare_path_cache(self, env, wq_value: float = 0.995):
        if self._full_table_t is not None or not self.is_active:
            return

        S = env.num_stages
        device = env.device
        m_stage = env.machine_stage                                                  # (M,) long, on device
        m_local = env.machine_local                                                  # (M,) long, on device
        m_per_stage = [int((m_stage == s).sum().item()) for s in range(S)]

        stage_globals = []
        for s in range(S):
            g_idx = torch.nonzero(m_stage == s, as_tuple=False).view(-1)
            stage_globals.append(g_idx[torch.argsort(m_local[g_idx])])

        # 0-based 카르테시안: 각 stage 의 machine 선택을 (m0, m1, ..., m_{S-1}) 로 펼침.
        grids = torch.meshgrid(*[torch.arange(n, device=device) for n in m_per_stage],
                               indexing='ij')
        all_paths = torch.stack([g.reshape(-1) for g in grids], dim=-1).to(torch.long)
        Npaths = all_paths.shape[0]

        wq_t = torch.full((Npaths, 1), wq_value, dtype=torch.float32, device=device)

        one_hots = []
        for s, cnt in enumerate(self.machine_cnt_list):
            oh = F.one_hot(all_paths[:, s], num_classes=cnt).float()
            one_hots.append(oh)

        X_tensor = torch.cat(one_hots + [wq_t], dim=1)

        with torch.no_grad():
            preds = self.quality_model.model(X_tensor)
            if isinstance(preds, tuple):
                preds = preds[0]
            y = preds.float()

        y_grid = y.reshape(*m_per_stage)                                             # 전부 GPU

        Z_SCALE = 1.0
        path_zscore = []
        for d in range(S):
            dims_to_avg = list(range(d + 1, S))
            marg = y_grid.mean(dim=dims_to_avg) if dims_to_avg else y_grid
            mu  = marg.mean(dim=-1, keepdim=True)
            std = marg.std(dim=-1, keepdim=True)
            z   = (marg - mu) / std.clamp(min=1e-8) * Z_SCALE
            path_zscore.append(z.to(torch.float32))

        M = env.total_machines
        prefix_counts = [1] * S
        for d in range(1, S):
            prefix_counts[d] = prefix_counts[d - 1] * m_per_stage[d - 1]
        state_offsets_list = [0]
        for c in prefix_counts:
            state_offsets_list.append(state_offsets_list[-1] + c)
        total_states = state_offsets_list[-1]

        stride_matrix = torch.zeros((S, S), dtype=torch.long, device=device)
        for d in range(1, S):
            strides = [1] * d
            for i in range(d - 2, -1, -1):
                strides[i] = strides[i + 1] * m_per_stage[i + 1]
            stride_matrix[d, :d] = torch.tensor(strides, dtype=torch.long, device=device)

        full_table = torch.zeros((total_states + 1, M), dtype=torch.float32, device=device)
        for d in range(S):
            flat_d = path_zscore[d].reshape(prefix_counts[d], m_per_stage[d])
            cols = stage_globals[d]
            rows = torch.arange(state_offsets_list[d],
                                state_offsets_list[d] + prefix_counts[d],
                                device=device)
            full_table[rows[:, None], cols[None, :]] = flat_d

        self._done_idx        = total_states
        self._state_offsets_t = torch.tensor(state_offsets_list, dtype=torch.long, device=device)
        self._stride_matrix_t = stride_matrix
        self._full_table_t    = full_table

    def compute_machine_quality(self, env) -> torch.Tensor | None:
        """(BP, J, M) — active depth + prefix 조건 stage-내 z-score prior. Fully torch on env.device."""
        if not self.is_active:
            return None
        self.prepare_path_cache(env)

        BP, J, S = env.batch_size, env.num_jobs, env.num_stages

        op_m_3d = env.op_machine.view(BP, J, S)
        assigned = op_m_3d != -1
        active_depth = assigned.sum(dim=2)                                          # (BP, J) ∈ [0, S]
        local_choice = torch.where(assigned, env.machine_local[op_m_3d.clamp(min=0)],
                                    torch.zeros_like(op_m_3d))

        prefix_linear_all = (local_choice.unsqueeze(-2)                              # (BP, J, 1, S)
                              * self._stride_matrix_t                                # (S, S) → broadcast
                              ).sum(dim=-1)                                          # (BP, J, S)
        depth_safe = active_depth.clamp(max=S - 1)
        prefix_linear = prefix_linear_all.gather(-1, depth_safe.unsqueeze(-1)).squeeze(-1)
        state_idx = self._state_offsets_t[depth_safe] + prefix_linear
        # active_depth == S (전체 완료) → done sentinel (all-zero row)
        state_idx = torch.where(active_depth == S,
                                 torch.full_like(state_idx, self._done_idx),
                                 state_idx)
        return self._full_table_t[state_idx]                                          # (BP, J, M)                                     # (BP, J, M)

# =====================================================
# Env adapter (env-precomputed pool 을 소비하는 얇은 어댑터)
# =====================================================
def make_env_edge_lookup(env: HFSPGraphEnv) -> torch.Tensor:
    """env 가 __init__ 에서 캐시한 (op, machine) → edge idx 룩업 반환."""
    return env.env_edge_lookup


def build_model_state(env: HFSPGraphEnv,
                      batch_idx: torch.Tensor,
                      B: int, P: int, device,
                      wafer_quality_t: torch.Tensor | None = None,
                      lambdas_t: torch.Tensor | None = None,
                      quality_helper: 'QualityHelper | None' = None) -> tuple[ModelState, torch.Tensor]:
    """env._compute_feature_pools 가 만들어둔 row/col/edge_pool 에 quality 두 채널만 덧붙여 ModelState 조립.

    env attrs 가져와 stack 만 함 — 시간 채널 계산/edge mask/active_op 은 env 가 이미 만들어 둠.
    채널을 늘리려면 (1) env._compute_feature_pools 의 pool dict 에 항목 추가 → (2) get_feat_names 에 이름 등록.
    quality 두 채널은 helper/입력에 의존하므로 wrapper 에서만 끼움.
    """
    BP = B * P
    J = env.num_jobs
    M = env.total_machines
    names = get_feat_names(env)

    # ── row: env pool + wafer_quality ──
    row_pool = dict(env.row_pool)
    row_pool['wafer_quality'] = (wafer_quality_t if wafer_quality_t is not None
                                  else torch.zeros((BP, J), dtype=torch.float32, device=device))
    row_feat = torch.stack([row_pool[n] for n in names['row']], dim=-1)

    # ── col: env pool 그대로 ──
    col_feat = torch.stack([env.col_pool[n] for n in names['col']], dim=-1)

    # ── edge: env pool + quality_zscore ──
    edge_pool = dict(env.edge_pool)
    if 'quality_zscore' in names['edge']:
        q_z_t = quality_helper.compute_machine_quality(env) if quality_helper is not None else None
        edge_pool['quality_zscore'] = (q_z_t if q_z_t is not None
                                        else torch.zeros(BP, J, M, dtype=torch.float32, device=device))
    edge_feat = torch.stack([edge_pool[n] for n in names['edge']], dim=-1)
    edge_feat.masked_fill_(~env.valid_edge_3d.unsqueeze(-1), 0.0)

    if lambdas_t is None:
        lambdas_t = torch.zeros(BP, dtype=torch.float32, device=device)

    state = ModelState(
        BATCH_IDX=batch_idx,
        row_feature=row_feat,
        col_feature=col_feat,
        edge_feature=edge_feat,
        edge_mask=env.edge_mask_ninf,
        finished=env.done,
        lambdas=lambdas_t,
    )
    return state, env.active_op

def model_action_to_env_edge(edge_selected: torch.Tensor,
                             active_op: torch.Tensor,
                             env_edge_lookup_t: torch.Tensor,
                             num_jobs: int) -> torch.Tensor:
    machine_idx = edge_selected // num_jobs
    job_idx = edge_selected % num_jobs
    bp_arange = torch.arange(edge_selected.size(0), device=edge_selected.device)
    op_for_j = active_op[bp_arange, job_idx]
    return env_edge_lookup_t[op_for_j, machine_idx]


# =====================================================
# Generic rollout body
# =====================================================
def _rollout_loop(env, model, env_edge_lookup_t, device, B, P, *,
                  greedy=False, with_grad=True, wafer_quality_t=None,
                  lambdas_t=None, quality_helper=None):
    BP = B * P
    log_probs = [] if with_grad else None
    total_reward = torch.zeros(BP, dtype=torch.float32, device=device)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, P).contiguous()

    was_training = model.training
    saved_eval_type = model.model_params.get('eval_type', 'softmax')
    if greedy:
        model.eval()
        model.model_params['eval_type'] = 'argmax'
    elif not was_training:
        model.model_params['eval_type'] = 'softmax'

    try:
        # 총 step 수 = env.num_ops (J·S) 로 고정 — done.all() 동기화 제거.
        T = env.num_ops
        for t in range(T):
            state, active_op_t = build_model_state(
                env, batch_idx, B, P, device,
                wafer_quality_t=wafer_quality_t,
                lambdas_t=lambdas_t,
                quality_helper=quality_helper,
            )
            edge_selected, flat_probs = model(state)

            if with_grad and not greedy:
                sel_prob = flat_probs.gather(1, edge_selected[:, None]).squeeze(1)
                sel_prob = torch.where(state.finished, torch.ones_like(sel_prob), sel_prob)
                log_probs.append(sel_prob.clamp(min=1e-30).log())

            env_edge = model_action_to_env_edge(edge_selected, active_op_t, env_edge_lookup_t, env.num_jobs)
            env_edge_safe = env_edge.clamp(min=0)

            # PyTorch 시뮬레이터는 Tensor를 직접 받음
            _, reward, _ = env.step(env_edge_safe, last=(t == T - 1))
            total_reward += reward
    finally:
        model.model_params['eval_type'] = saved_eval_type
        if was_training:
            model.train()

    G = total_reward
    log_prob_sum = torch.stack(log_probs, dim=0).sum(dim=0) if with_grad and log_probs else None
    return log_prob_sum, G


# =====================================================
# Self-Imitation Learning rollout
# =====================================================
def sil_pomo_rollout(env: HFSPGraphEnv, model: FFSPModel,
                     env_edge_lookup_t, device, seed: int, B: int, P: int,
                     lambdas: torch.Tensor,
                     quality_helper: QualityHelper,
                     m_best: float, m_worst: float,
                     q_best: float, q_worst: float,
                     recorder=None,
                     use_eft_slot: bool = True):
    # use_sjf_slot=True 면 P 샘플 중 마지막 1개를 sample_sjf_v2_action (EST-우선 휴리스틱)
    # 의 action 으로 대체. Phase-2 grad-replay 가 (B,P) reward 의 argmax 슬롯만 추적하므로,
    # 휴리스틱 trajectory 가 best 가 되면 모델이 자연스럽게 그 actions 을 imitate (BC).
    # 모델이 휴리스틱을 따라잡으면 best 슬롯이 다시 모델 쪽으로 이동 → smooth handoff.
    BP = B * P
    batch_idx_full = torch.arange(B, device=device)[:, None].expand(B, P).contiguous()
    batch_idx_best = torch.arange(B, device=device)[:, None]

    lam_bp = lambdas.to(device).float().view(B, 1).expand(B, P).reshape(BP).contiguous()

    # B 시나리오 × B 람다 1:1 매칭 — 각 batch row 가 unique (scenario, λ) 페어.
    # P 샘플은 같은 (scenario, λ) 안에서 trajectory 만 다름 → POMO baseline.
    pt_unique = env.sample_edge_proc_time(B, seed, identical_job=True)
    pt_bp = pt_unique.repeat_interleave(P, dim=0)

    wq_unique = quality_helper.sample_wafer_quality(B, env.num_jobs, seed, device)
    if wq_unique is not None:
        wafer_q_bp = wq_unique.repeat_interleave(P, dim=0)
    else:
        wafer_q_bp = None

    env.reset(seed=seed, batch_size=BP, edge_proc_time=pt_bp)
    states_seq = []
    history_edge_selected = torch.empty((env.num_ops, BP), dtype=torch.long, device=device)
    total_reward = torch.zeros(BP, dtype=torch.float32, device=device)

    was_training = model.training
    saved_eval_type = model.model_params.get('eval_type', 'softmax')
    model.eval()
    model.model_params['eval_type'] = 'softmax'

    # use_sjf_slot: BP 안에서 p == P-1 슬롯만 SJF 휴리스틱 action 으로 대체.
    # BP layout 은 (B, P) row-major → bp_idx % P == P-1 이 마지막 슬롯.
    sjf_slot_mask = None
    if use_eft_slot:
        sjf_slot_mask = (torch.arange(B * P, device=device) % P) == (P - 1)

    with torch.no_grad():
        # 총 step 수 = env.num_ops (J·S) 로 고정 — done.all() 동기화 제거.
        T = env.num_ops
        for t in range(T):
            state, active_op_t = build_model_state(
                env, batch_idx_full, B, P, device,
                wafer_quality_t=wafer_q_bp, lambdas_t=lam_bp, quality_helper=quality_helper,
            )
            edge_selected, _ = model(state)

            # SJF override — 마지막 P 슬롯의 (machine, job) edge_selected 를 env-edge 가 아닌
            # model action 공간(edge_selected = machine_idx*num_jobs + job_idx)으로 환산해 덮어씀.
            # sample_sjf_v2_action 은 env-edge idx 를 돌려주므로 역변환 필요:
            #   env_edge = env_edge_lookup[op_for_j, machine_idx] → (machine_idx, op→job) 추출.
            if sjf_slot_mask is not None:
                sjf_env_edge = sample_est_action(env, {'feasible_mask': env.feasible_mask})
                sjf_machine_idx = env.edge_machine[sjf_env_edge]
                sjf_op_idx = env.edge_op[sjf_env_edge]
                sjf_job_idx = sjf_op_idx // env.num_stages
                sjf_action = sjf_machine_idx * env.num_jobs + sjf_job_idx
                edge_selected = torch.where(sjf_slot_mask, sjf_action, edge_selected)

            states_seq.append(state)
            history_edge_selected[t] = edge_selected.detach()

            env_edge = model_action_to_env_edge(edge_selected, active_op_t, env_edge_lookup_t, env.num_jobs)
            _, reward, _ = env.step(env_edge.clamp(min=0), last=(t == T - 1))
            total_reward += reward
            
    model.model_params['eval_type'] = saved_eval_type
    if was_training:
        model.train()

    G_ms = total_reward
    G_q = quality_helper.compute_yield(env, wafer_q_bp, aggregate="mean")

    m_span = max(m_worst - m_best, 1e-8)
    q_span = max(q_best  - q_worst, 1e-8)
    m_hat = (m_worst - (-G_ms)) / m_span
    q_hat = (G_q - q_worst) / q_span
    r = (1.0 - lam_bp) * m_hat + lam_bp * q_hat
    r_bp = r.view(B, P)

    best_p_idx = r_bp.argmax(dim=1)
    best_flat_idx = torch.arange(B, device=device) * P + best_p_idx

    # Phase 2: pop 으로 BP-sized state 를 즉시 풀어 peak memory 절감.
    # slice (state_bp[...][best_flat_idx]) 가 새 B-sized 텐서를 만들고 나면 원본 참조 불필요.
    # 순서는 log_prob 합산에 영향 없음.
    actions_best_seq = history_edge_selected[:env.num_ops, best_flat_idx]   # (T, B)
    del history_edge_selected
    log_probs = []
    i = env.num_ops - 1
    while states_seq:
        state_bp = states_seq.pop()
        a_best = actions_best_seq[i]
        i -= 1
        state_b = ModelState(
            BATCH_IDX=batch_idx_best,
            row_feature=state_bp.row_feature[best_flat_idx],
            col_feature=state_bp.col_feature[best_flat_idx],
            edge_feature=state_bp.edge_feature[best_flat_idx],
            edge_mask=state_bp.edge_mask[best_flat_idx],
            finished=state_bp.finished[best_flat_idx],
            lambdas=state_bp.lambdas[best_flat_idx],
        )
        if recorder is not None:
            recorder.watch_step(state_b)

        del state_bp                                                 # 즉시 free
        # sample=False: action 은 a_best 로 fixed → Gumbel-Max 스킵.
        _, flat_probs = model(state_b, sample=False)
        sel_prob = flat_probs.gather(1, a_best[:, None]).squeeze(1)
        sel_prob = torch.where(state_b.finished, torch.ones_like(sel_prob), sel_prob)
        log_probs.append(sel_prob.clamp(min=1e-30).log())

    log_prob_sum_best = torch.stack(log_probs, dim=0).sum(dim=0)
    return log_prob_sum_best, G_ms, G_q, r_bp


# =====================================================
# Pareto evaluation
# =====================================================
def _hypervolume_2d(ms_at_lam: torch.Tensor, q_at_lam: torch.Tensor,
                    m_ref: float, q_ref: float):
    L, B = ms_at_lam.shape
    device = ms_at_lam.device
    order_q = torch.argsort(-q_at_lam, dim=0, stable=True)
    m_by_q  = torch.gather(ms_at_lam, 0, order_q)
    q_by_q  = torch.gather(q_at_lam,  0, order_q)
    order_m = torch.argsort(m_by_q, dim=0, stable=True)
    m_s = torch.gather(m_by_q, 0, order_m)
    q_s = torch.gather(q_by_q, 0, order_m)

    cmax, _ = torch.cummax(q_s, dim=0)
    # nd_mask[0] = True, nd_mask[i] = q_s[i] > max(q_s[0:i])
    nd_mask = torch.cat([
        torch.ones((1, B), dtype=torch.bool, device=device),
        q_s[1:] > cmax[:-1],
    ], dim=0)
    nd_count = nd_mask.sum(dim=0).to(torch.int32)

    m_nd = torch.where(nd_mask, m_s, torch.full_like(m_s, float('inf')))
    ref_row = torch.full((1, B), m_ref, dtype=m_s.dtype, device=device)
    m_with_ref = torch.cat([m_nd, ref_row], dim=0)
    # 역방향 cummin → next_m[i] = min(m_nd[i+1:] ∪ {m_ref})
    next_m = torch.cummin(m_with_ref.flip(0), dim=0).values.flip(0)[1:]

    valid = nd_mask & (m_s < m_ref) & (q_s > q_ref)
    contrib = torch.where(valid, (next_m - m_s) * (q_s - q_ref),
                          torch.zeros_like(m_s))
    hv = contrib.sum(dim=0).to(torch.float32)
    return hv, nd_count


def eval_pareto(eval_env: HFSPGraphEnv, model: FFSPModel,
                env_edge_lookup_t, device, *,
                batch_size: int, lambdas: torch.Tensor,
                quality_helper: QualityHelper, seed: int,
                m_ref: float, q_ref: float):

    B = int(batch_size)
    L = int(lambdas.numel())
    if B % L != 0:
        raise ValueError(f"eval batch_size ({B}) must be divisible by len(lambdas) ({L})")
    K = B // L

    pt_unique = eval_env.sample_edge_proc_time(K, seed, identical_job=True)
    pt_lb = pt_unique.repeat(L, 1)

    wq_unique = quality_helper.sample_wafer_quality(K, eval_env.num_jobs, seed, device)
    wq_lb = wq_unique.repeat(L, 1) if wq_unique is not None else None

    eval_env.reset(seed=seed, batch_size=B, edge_proc_time=pt_lb)
    lam_lb = lambdas.to(device=device, dtype=torch.float32).repeat_interleave(K)

    with torch.no_grad():
        _, G_ms = _rollout_loop(eval_env, model, env_edge_lookup_t, device,
                                B=B, P=1, greedy=True, with_grad=False,
                                wafer_quality_t=wq_lb, lambdas_t=lam_lb,
                                quality_helper=quality_helper)

    ms = (-G_ms).reshape(L, K)
    q_flat = quality_helper.compute_yield(eval_env, wq_lb, aggregate="mean")
    q = q_flat.reshape(L, K)

    hv, nd = _hypervolume_2d(ms, q, m_ref, q_ref)
    return ms, q, hv, nd


def plot_pareto(ms: torch.Tensor, q: torch.Tensor, lambdas: torch.Tensor,
                save_path: str, title: str = "",
                xlim: tuple = (100.0, 500.0),
                ylim: tuple = (0.50, 0.85),
                ref_point: tuple | None = None):
    # matplotlib boundary — 여기에서만 텐서를 numpy 로 내림. compute path 는 전부 torch.
    # xlim/ylim/ref_point 는 시각화 전용 하드코딩 값 — HV/reward 정규화 anchor 와 무관.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    ms_np  = ms.detach().cpu().numpy()      if torch.is_tensor(ms)      else ms
    q_np   = q.detach().cpu().numpy()       if torch.is_tensor(q)       else q
    lam_np = lambdas.detach().cpu().numpy() if torch.is_tensor(lambdas) else lambdas

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ms_np.ravel(), q_np.ravel(), s=8, alpha=0.15,
               c=np.repeat(lam_np, ms_np.shape[1]),
               cmap='viridis', vmin=0, vmax=1)

    m_mean, q_mean = ms_np.mean(axis=1), q_np.mean(axis=1)
    ax.plot(m_mean, q_mean, '-', color='gray', lw=1, alpha=0.6, zorder=4)
    sc = ax.scatter(m_mean, q_mean, c=lam_np, s=100, cmap='viridis',
                    edgecolor='black', zorder=5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='λ')

    if ref_point is not None:
        m_r, q_r = ref_point
        ax.scatter([m_r], [q_r], marker='X', color='red', s=60, zorder=4,
                   clip_on=False, label=f'ref ({m_r:.1f}, {q_r:.3f})')
        ax.legend(loc='lower left', fontsize=8)

    ax.set_xlabel('Makespan ↓'); ax.set_ylabel('Yield ↑')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(save_path, dpi=100); plt.close(fig)