"""
HFSP Graph-based RL Environment — CUDA-Optimized (No Triton)
============================================================
핵심 최적화:
1. Slot-fitting Python for-loop 완전 제거 → cummax + broadcast 로 병렬화
   - step()  1D: (N, J) 행렬 연산 1회로 대체
   - feature_pools() 3D: (B, J, M, J_iv) 4D broadcast 로 대체
2. 모든 hot-path 임시 텐서 사전 할당
3. 스칼라 상수 텐서 __init__ 에서 1회 생성, 재사용
4. in-place 연산 적극 활용
"""

from __future__ import annotations
import torch
import torch.nn.functional as F

from FFSProblemDef import get_random_problems, get_random_problems_identical_jobs


def derive_active_op(op_status: torch.Tensor, num_jobs: int, num_stages: int,
                     _const_last_stage=None):
    B = op_status.size(0)
    js = op_status.view(B, num_jobs, num_stages)
    not_done = js != 3
    has_active = not_done.any(dim=2)
    first_active_stage = not_done.to(torch.uint8).argmax(dim=2)
    if _const_last_stage is None:
        _const_last_stage = num_stages - 1
    active_stage = torch.where(has_active, first_active_stage, _const_last_stage)
    job_idx = torch.arange(num_jobs, device=op_status.device)
    active_op = job_idx.unsqueeze(0) * num_stages + active_stage
    is_done = ~has_active
    return active_op, is_done


class HFSPGraphEnv:
    def __init__(self, num_jobs, machine_cnt_list, device,
                 time_low=2, time_high=10, id_dim=6):

        self.device = torch.device(device)

        self.num_jobs = int(num_jobs)
        self.machine_cnt_list = list(machine_cnt_list)
        self.num_stages = len(self.machine_cnt_list)
        self.total_machines = int(sum(self.machine_cnt_list))
        self.time_low = int(time_low)
        self.time_high = int(time_high)
        self.id_dim = int(id_dim)

        J = self.num_jobs
        S = self.num_stages
        M = self.total_machines

        # ── 노드 메타 ──
        self.num_ops = J * S
        self.op_job = torch.arange(J, device=self.device).repeat_interleave(S)
        self.op_stage = torch.arange(S, device=self.device).repeat(J)

        # ── machine 메타 ──
        mcl = torch.tensor(self.machine_cnt_list, dtype=torch.long, device=self.device)
        self.machine_offset = torch.cat([mcl.new_zeros(1), mcl.cumsum(0)[:-1]])
        self.machine_stage = torch.arange(S, device=self.device).repeat_interleave(mcl)
        self.machine_local = (torch.arange(M, device=self.device)
                              - self.machine_offset[self.machine_stage])

        # ── 엣지 메타 ──
        pairs = (self.op_stage.unsqueeze(1) == self.machine_stage.unsqueeze(0)).nonzero(as_tuple=False)
        self.edge_op = pairs[:, 0].contiguous()
        self.edge_machine = pairs[:, 1].contiguous()
        self.num_edges = self.edge_op.numel()

        self.env_edge_lookup = torch.full(
            (self.num_ops, M), -1, dtype=torch.long, device=self.device)
        self.env_edge_lookup[self.edge_op, self.edge_machine] = torch.arange(
            self.num_edges, device=self.device)

        # predecessor
        arange_ops = torch.arange(self.num_ops, device=self.device)
        self.prev_op_in_job = torch.where(self.op_stage > 0, arange_ops - 1,
                                           torch.full_like(arange_ops, -1))
        self._has_prev = self.prev_op_in_job >= 0
        self._pred_safe = torch.where(self._has_prev, self.prev_op_in_job,
                                       torch.zeros_like(self.prev_op_in_job))

        # proc_time index mapping
        s_e = self.machine_stage[self.edge_machine]
        j_e = self.edge_op // S
        self._proc_src_idx = (J * self.machine_offset[s_e]
                              + j_e * mcl[s_e]
                              + self.machine_local[self.edge_machine])

        # ── 상수 캐시 ──
        self._CONST_ZERO_L = torch.tensor(0, dtype=torch.long, device=self.device)
        self._CONST_ZERO_F = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self._CONST_TRUE = torch.tensor(True, device=self.device)
        self._CONST_INF_F = torch.tensor(float('inf'), dtype=torch.float32, device=self.device)
        self._CONST_LAST_STAGE = torch.tensor(S - 1, dtype=torch.long, device=self.device)
        self._CONST_S = torch.tensor(S, dtype=torch.long, device=self.device)
        self._INT64_MAX = torch.iinfo(torch.long).max
        self._NEG_SENTINEL = torch.tensor(-(2**60), dtype=torch.long, device=self.device)

        # static feature 텐서
        self._stage_norm = self.machine_stage.float() / max(S - 1, 1)
        self._stage_oh_m = F.one_hot(self.machine_stage, num_classes=S).float()
        self._arange_J = torch.arange(J, device=self.device)
        self._arange_M = torch.arange(M, device=self.device)
        self._job_times_S = self._arange_J * S
        self._k_range_J = torch.arange(J, device=self.device)

        # expand 용 static index
        self._has_prev_u = self._has_prev.unsqueeze(0)
        self._pred_safe_u = self._pred_safe.unsqueeze(0)
        self._edge_op_u = self.edge_op.unsqueeze(0)

        # ── 동적 상태 ──
        self.batch_size = None
        self.edge_proc_time = None
        self.per_op_min_proc = None
        self.op_assigned = None
        self.op_machine = None
        self.op_proc_chosen = None
        self.op_start = None
        self.op_end = None
        self.machine_iv_start = None
        self.machine_iv_end = None
        self.machine_iv_count = None
        self.feasible_mask = None
        self.done = None
        self.row_pool = None
        self.col_pool = None
        self.edge_pool = None
        self.active_op = None
        self.valid_edge_3d = None
        self.edge_mask_ninf = None

        self._bufs_B = -1

    # =====================================================
    # 버퍼 사전 할당
    # =====================================================
    def _alloc_buffers(self, B):
        if self._bufs_B == B:
            return
        dev = self.device
        J = self.num_jobs
        M = self.total_machines
        S = self.num_stages

        self._buf_b_idx = torch.arange(B, device=dev)
        self._buf_prev_done = torch.zeros(B, dtype=torch.bool, device=dev)
        self._buf_k_idx = torch.arange(J, device=dev).unsqueeze(0)
        self._buf_stage_oh_bp = self._stage_oh_m.unsqueeze(0).expand(B, -1, -1).contiguous()
        self._buf_op_status = torch.zeros((B, self.num_ops), dtype=torch.long, device=dev)
        self._buf_ninf_mask = torch.zeros((B, M, J), dtype=torch.float32, device=dev)

        self._bufs_B = B

    # =====================================================
    # Slot Fitting — Fully Vectorized (No Python Loop)
    # =====================================================
    def _slot_fit_1d_vec(self, ivs_s, ivs_e, cnt, t_min, a_proc):
        """Vectorized 1D slot fitting for step().
        
        핵심: cursor_k = max(t_min, e_0, ..., e_{k-1}) = max(t_min, cummax(e)[k-1])
        → sequential dependency 완전 제거, 단일 cummax + broadcast 로 해결.
        
        Args: ivs_s, ivs_e: (N, J), cnt: (N,), t_min, a_proc: (N,)
        Returns: slot_start (N,)
        """
        N = t_min.shape[0]
        J = self.num_jobs

        k_range = self._buf_k_idx[:, :J]
        valid_k = k_range < cnt.unsqueeze(1)

        # invalid e → sentinel (cummax 에 영향 안 줌)
        e_safe = torch.where(valid_k, ivs_e, self._NEG_SENTINEL)
        prefix_max_e = e_safe.cummax(dim=1)[0]

        # cursor[k] = max(t_min, shifted_prefix[k])
        # shifted: [NEG_SENTINEL, prefix_max[0], ..., prefix_max[J-2]]
        shifted = torch.cat([
            self._NEG_SENTINEL.expand(N, 1),
            prefix_max_e[:, :-1]
        ], dim=1)
        cursor_all = torch.maximum(t_min.unsqueeze(1), shifted)

        # gap 계산 + 첫 번째 fit 탐색
        gap = ivs_s - cursor_all
        fits = valid_k & (gap >= a_proc.unsqueeze(1))

        has_fit = fits.any(dim=1)
        first_k = fits.to(torch.uint8).argmax(dim=1)
        found_start = cursor_all.gather(1, first_k.unsqueeze(1)).squeeze(1)

        # not found → cursor after all intervals
        last_valid = (cnt - 1).clamp(min=0)
        max_e = prefix_max_e.gather(1, last_valid.unsqueeze(1)).squeeze(1)
        cursor_at_end = torch.where(cnt > 0, torch.maximum(t_min, max_e), t_min)

        return torch.where(has_fit, found_start, cursor_at_end)

    def _slot_fit_3d_vec(self, t_min_jm, proc_jm, valid_edge_3d):
        """Vectorized 3D slot fitting for _compute_feature_pools().
        
        (B, J, M, J_iv) 4D broadcast: 모든 (b,j,m)×(k) 조합을 동시 계산.
        Python for-loop 완전 제거.
        
        GPU 메모리: B×J×M×J_iv × 8bytes (int64).
        B=1000, J=25, M=30 → ~150MB 추가. 대부분의 GPU 에서 문제 없음.
        """
        B = self.batch_size
        J = self.num_jobs
        M = self.total_machines

        k_range = self._k_range_J
        iv_cnt = self.machine_iv_count                                   # (B, M)
        valid_iv_2d = k_range.view(1, 1, -1) < iv_cnt.unsqueeze(-1)     # (B, M, J_iv)

        e_safe = torch.where(valid_iv_2d, self.machine_iv_end,
                             self._NEG_SENTINEL)
        prefix_max_e = e_safe.cummax(dim=2)[0]                           # (B, M, J_iv)

        shifted = torch.cat([
            self._NEG_SENTINEL.expand(B, M, 1),
            prefix_max_e[:, :, :-1]
        ], dim=2)                                                         # (B, M, J_iv)

        # 4D broadcast: cursor(b,j,m,k) = max(t_min(b,j,m), shifted(b,m,k))
        t_min_4d = t_min_jm.unsqueeze(-1).long()                         # (B, J, M, 1)
        shifted_4d = shifted.unsqueeze(1)                                 # (B, 1, M, J_iv)
        cursor_all = torch.maximum(t_min_4d, shifted_4d)                  # (B, J, M, J_iv)

        # gap & fits
        iv_s_4d = self.machine_iv_start.unsqueeze(1)                      # (B, 1, M, J_iv)
        gap = iv_s_4d - cursor_all
        valid_iv_4d = valid_iv_2d.unsqueeze(1)                            # (B, 1, M, J_iv)
        valid_edge_4d = valid_edge_3d.unsqueeze(-1)                       # (B, J, M, 1)
        proc_4d = proc_jm.unsqueeze(-1).long()                           # (B, J, M, 1)

        fits = valid_edge_4d & valid_iv_4d & (gap >= proc_4d)            # (B, J, M, J_iv)

        has_fit = fits.any(dim=-1)                                        # (B, J, M)
        first_k = fits.to(torch.uint8).argmax(dim=-1)                    # (B, J, M)
        found_start = cursor_all.gather(-1, first_k.unsqueeze(-1)).squeeze(-1).float()

        # not found → cursor at end
        last_valid = (iv_cnt - 1).clamp(min=0)
        max_e = prefix_max_e.gather(2, last_valid.unsqueeze(-1)).squeeze(-1)  # (B, M)
        t_min_long = t_min_jm.long()
        max_e_exp = max_e.unsqueeze(1).expand_as(t_min_long)
        has_iv = (iv_cnt > 0).unsqueeze(1)
        cursor_at_end = torch.where(has_iv, torch.maximum(t_min_long, max_e_exp),
                                    t_min_long).float()

        est_jm = torch.where(has_fit, found_start, cursor_at_end)
        est_jm = torch.where(valid_edge_3d, est_jm, self._CONST_ZERO_F)
        return est_jm

    # =====================================================
    # Problem sampling
    # =====================================================
    def sample_edge_proc_time(self, B, seed, identical_job=False):
        fn = get_random_problems_identical_jobs if identical_job else get_random_problems
        pts = fn(
            batch_size=B, stage_cnt=self.num_stages,
            machine_cnt_list=self.machine_cnt_list, job_cnt=self.num_jobs,
            process_time_params={'time_low': self.time_low, 'time_high': self.time_high},
            seed=seed, device=self.device,
        )
        concat = torch.cat([pt.reshape(B, -1) for pt in pts], dim=1)
        return concat[:, self._proc_src_idx]

    # =====================================================
    # Lifecycle
    # =====================================================
    def reset_pomo(self, seed=0, batch_size=1, pomo_size=1, identical_job=False):
        B = int(batch_size)
        P = pomo_size
        self.batch_size = B * P

        if B * P != self.batch_size:
            raise ValueError(f"B*P ({B}*{P}={B*P}) != batch_size ({self.batch_size})")

        pt_b = self.sample_edge_proc_time(B, seed, identical_job)
        pt_bp = pt_b.repeat_interleave(P, dim=0)
        return self.reset(seed=seed, batch_size=self.batch_size, edge_proc_time=pt_bp)

    def reset(self, seed=0, batch_size=1, edge_proc_time=None, identical_job=False):
        self.batch_size = int(batch_size)
        B = self.batch_size
        dev = self.device
        J = self.num_jobs
        M = self.total_machines
        NE = self.num_edges

        if edge_proc_time is not None:
            ept = edge_proc_time.to(dev).long()
            if ept.shape != (B, NE):
                raise ValueError(f"edge_proc_time shape {ept.shape} != ({B}, {NE})")
            self.edge_proc_time = ept
        else:
            self.edge_proc_time = self.sample_edge_proc_time(B, seed, identical_job)

        self.per_op_min_proc = torch.full((B, self.num_ops), float('inf'),
                                           dtype=torch.float32, device=dev)
        self.per_op_min_proc.scatter_reduce_(
            1,
            self._edge_op_u.expand(B, -1),
            self.edge_proc_time.float(),
            reduce='amin',
            include_self=False,
        )

        self.op_assigned = torch.zeros((B, self.num_ops), dtype=torch.bool, device=dev)
        self.op_machine = torch.full((B, self.num_ops), -1, dtype=torch.long, device=dev)
        self.op_proc_chosen = torch.full((B, self.num_ops), -1, dtype=torch.long, device=dev)
        self.op_start = torch.full((B, self.num_ops), -1, dtype=torch.long, device=dev)
        self.op_end = torch.full((B, self.num_ops), -1, dtype=torch.long, device=dev)

        self.machine_iv_start = torch.full((B, M, J), self._INT64_MAX, dtype=torch.long, device=dev)
        self.machine_iv_end = torch.full((B, M, J), self._INT64_MAX, dtype=torch.long, device=dev)
        self.machine_iv_count = torch.zeros((B, M), dtype=torch.long, device=dev)

        self.done = torch.zeros(B, dtype=torch.bool, device=dev)

        self._alloc_buffers(B)

        self._refresh_feasibility()
        self._compute_feature_pools()
        return self._get_obs()

    # =====================================================
    # Feasibility
    # =====================================================
    def _op_ready_array(self):
        B = self.batch_size
        pred_assigned = self.op_assigned.gather(1, self._pred_safe_u.expand(B, -1))
        pred_ok = torch.where(self._has_prev_u, pred_assigned, self._CONST_TRUE)
        return (~self.op_assigned) & pred_ok

    def _refresh_feasibility(self):
        self._op_ready_cache = self._op_ready_array()
        B = self.batch_size
        self.feasible_mask = self._op_ready_cache.gather(
            1, self._edge_op_u.expand(B, -1))

    # =====================================================
    # Feature pools
    # =====================================================
    def _compute_feature_pools(self):
        B = self.batch_size
        J = self.num_jobs
        M = self.total_machines
        S = self.num_stages
        dev = self.device

        # ── active op ──
        op_status = self._buf_op_status
        op_status.zero_()
        op_status[self._op_ready_cache] = 1
        op_status[self.op_assigned] = 3

        active_op, is_done = derive_active_op(
            op_status, J, S, _const_last_stage=self._CONST_LAST_STAGE)
        self.active_op = active_op

        # ── edge index 3D ──
        edge_idx_3d = self.env_edge_lookup[active_op]
        valid_edge_3d = edge_idx_3d >= 0
        safe_idx_3d = edge_idx_3d.clamp(min=0)

        proc_jm = self.edge_proc_time.gather(
            1, safe_idx_3d.view(B, -1)).view(B, J, M).float()

        # ── EST/EFT — vectorized ──
        pred_safe_a = self._pred_safe[active_op]
        has_prev_a = self._has_prev[active_op]
        pred_end_a = self.op_end.gather(1, pred_safe_a)
        pred_assigned_a = self.op_assigned.gather(1, pred_safe_a)
        pred_ready = has_prev_a & pred_assigned_a
        pred_end_safe_a = torch.where(pred_ready, pred_end_a, self._CONST_ZERO_L).float()
        t_min_jm = pred_end_safe_a.unsqueeze(-1).expand(-1, -1, M)

        est_jm = self._slot_fit_3d_vec(t_min_jm, proc_jm, valid_edge_3d)
        eft_jm = est_jm + proc_jm

        # ── min_est ──
        feas_for_min = valid_edge_3d & (~is_done).unsqueeze(-1)
        min_est_raw = torch.where(feas_for_min, est_jm, self._CONST_INF_F).amin(dim=(1, 2))
        min_est = torch.where(torch.isinf(min_est_raw), self._CONST_ZERO_F, min_est_raw)
        min_est_bj = min_est.unsqueeze(1)
        min_est_bjm = min_est.view(B, 1, 1)

        # ── row pool ──
        active_stage_internal = active_op - self._job_times_S.unsqueeze(0)
        active_stage_feat = torch.where(is_done, self._CONST_S, active_stage_internal).float()
        active_stage_oh = F.one_hot(active_stage_feat.long(), num_classes=S + 1).float()

        op_end_safe = torch.where(self.op_assigned, self.op_end.float(), self._CONST_ZERO_F)
        recent_end = op_end_safe.view(B, J, S).max(dim=2)[0]
        recent_end_rel = recent_end - min_est_bj

        op_end_rem = (self.op_end.float() - min_est_bj).clamp_(min=0.0)
        per_op_rem = torch.where(self.op_assigned, op_end_rem, self.per_op_min_proc)
        remaining_proc = per_op_rem.view(B, J, S).sum(dim=2)

        self.row_pool = {
            'active_stage':      active_stage_feat,
            'recent_end':        recent_end_rel,
            'remaining_proc':    remaining_proc,
            'active_stage_done': active_stage_oh[..., S],
            **{f'active_stage_s{s}': active_stage_oh[..., s] for s in range(S)},
        }

        # ── col pool ──
        cnt = self.machine_iv_count
        iv_count_f = cnt.float()
        last_idx = (cnt - 1).clamp(min=0, max=J - 1)
        b_arr = self._buf_b_idx.unsqueeze(1)
        m_arr = self._arange_M.unsqueeze(0)
        last_end = self.machine_iv_end[b_arr, m_arr, last_idx]
        ready_t = torch.where(cnt > 0, last_end, self._CONST_ZERO_L).float()

        idle = (ready_t <= min_est_bj).float()
        time_to_idle = (ready_t - min_est_bj).clamp_(min=0.0)

        k_range = self._k_range_J
        valid_iv = k_range.view(1, 1, -1) < iv_count_f.unsqueeze(-1)
        iv_s_safe = torch.where(valid_iv, self.machine_iv_start.float(), self._CONST_ZERO_F)
        iv_e_safe = torch.where(valid_iv, self.machine_iv_end.float(), self._CONST_ZERO_F)
        busy = (iv_e_safe - iv_s_safe).sum(dim=2)
        util = torch.where(cnt > 0, busy / ready_t.clamp(min=1e-9), self._CONST_ZERO_F)

        self.col_pool = {
            'stage_id':     self._stage_norm.unsqueeze(0).expand(B, -1),
            'idle':         idle,
            'time_to_idle': time_to_idle,
            'iv_count':     iv_count_f,
            'utilization':  util,
            **{f'stage_id_s{s}': self._buf_stage_oh_bp[..., s] for s in range(S)},
        }

        # ── edge pool ──
        self.edge_pool = {
            'proc_time': proc_jm,
            'est':       est_jm - min_est_bjm,
            'eft':       eft_jm - min_est_bjm,
        }
        self.valid_edge_3d = valid_edge_3d

        # ── edge mask ──
        feas_3d = valid_edge_3d & (~is_done).unsqueeze(-1)
        feas_mt = feas_3d.transpose(1, 2)
        feas_mt = feas_mt | self.done.view(B, 1, 1)
        ninf_mask = self._buf_ninf_mask
        ninf_mask.zero_()
        ninf_mask.masked_fill_(~feas_mt, float('-inf'))
        self.edge_mask_ninf = ninf_mask

    # =====================================================
    # Step
    # =====================================================
    def step(self, actions, last=False, assume_all_active=True, validate=False):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        else:
            actions = actions.to(self.device).long().view(-1)

        B = self.batch_size
        if actions.shape[0] != B:
            raise ValueError(f"actions length {actions.shape[0]} != batch_size {B}")

        prev_done = self._buf_prev_done
        prev_done.copy_(self.done)
        b_idx = self._buf_b_idx

        if validate:
            active_v = ~self.done
            chosen_feasible = self.feasible_mask[b_idx, actions]
            bad = active_v & (~chosen_feasible)
            if bad.any():
                idx = b_idx[bad].cpu().tolist()
                raise ValueError(f"Infeasible action for batch indices {idx}")

        if assume_all_active:
            ab = b_idx
            a_act = actions
        else:
            active = ~self.done
            ab = b_idx[active]
            a_act = actions[active]

        a_op = self.edge_op[a_act]
        a_m = self.edge_machine[a_act]
        a_proc = self.edge_proc_time[ab, a_act]

        has_prev = self._has_prev[a_op]
        prev_op = self._pred_safe[a_op]
        t_min = torch.where(has_prev, self.op_end[ab, prev_op], self._CONST_ZERO_L)

        ivs_s = self.machine_iv_start[ab, a_m]
        ivs_e = self.machine_iv_end[ab, a_m]
        cnt = self.machine_iv_count[ab, a_m]

        # ── Vectorized slot fitting — NO Python loop ──
        slot_start = self._slot_fit_1d_vec(ivs_s, ivs_e, cnt, t_min, a_proc)
        end_t = slot_start + a_proc

        self.op_assigned[ab, a_op] = True
        self.op_machine[ab, a_op] = a_m
        self.op_proc_chosen[ab, a_op] = a_proc
        self.op_start[ab, a_op] = slot_start
        self.op_end[ab, a_op] = end_t

        # ── Left-shift gap-filling ──
        k_idx = self._buf_k_idx
        pos = (ivs_s < slot_start.unsqueeze(1)).sum(dim=1)
        pos_b = pos.unsqueeze(1)
        is_after = k_idx > pos_b
        is_at = k_idx == pos_b

        J = self.num_jobs
        src = torch.where(is_after, k_idx - 1, k_idx).clamp_(0, J - 1)
        new_s = ivs_s.gather(1, src)
        new_e = ivs_e.gather(1, src)
        new_s = torch.where(is_at, slot_start.unsqueeze(1), new_s)
        new_e = torch.where(is_at, end_t.unsqueeze(1), new_e)

        self.machine_iv_start[ab.unsqueeze(1), a_m.unsqueeze(1), k_idx] = new_s
        self.machine_iv_end[ab.unsqueeze(1), a_m.unsqueeze(1), k_idx] = new_e
        self.machine_iv_count[ab, a_m] = cnt + 1

        self.done = self.done | self.op_assigned.all(dim=1)

        # ── Reward ──
        just_done = self.done & (~prev_done)
        ms_all = self.op_end.max(dim=1)[0]
        reward = torch.where(just_done, -ms_all, self._CONST_ZERO_L)

        if not last:
            self._refresh_feasibility()
            self._compute_feature_pools()
        return self._get_obs(), reward, self.done

    def makespan(self):
        return self.op_end.max(dim=1)[0]

    # =====================================================
    # Observation
    # =====================================================
    def _op_status(self):
        status = torch.zeros((self.batch_size, self.num_ops), dtype=torch.long, device=self.device)
        status[self._op_ready_cache] = 1
        status[self.op_assigned] = 3
        return status

    def _get_obs(self):
        return {
            'feasible_mask': self.feasible_mask,
            'done': self.done,
        }

    def feasible_edges_per_batch(self):
        return [torch.where(self.feasible_mask[b])[0] for b in range(self.batch_size)]

    # =====================================================
    # Utilities & Replay
    # =====================================================
    def render_schedule(self, batch_idx=0, save_path=None, title=None):
        import matplotlib.pyplot as plt
        b = batch_idx

        assigned = self.op_assigned[b].tolist()
        if not any(assigned):
            print(f"[batch {b}] 아직 dispatch된 작업이 없습니다.")
            return

        ms_val = float(self.op_end[b].max().item())
        cmap = plt.get_cmap('tab20', max(self.num_jobs, 1))
        fig, ax = plt.subplots(figsize=(max(8, ms_val / 3), max(4, 0.4 * self.total_machines + 2)))

        for op in range(self.num_ops):
            if not assigned[op]:
                continue
            j = int(self.op_job[op].item())
            m = int(self.op_machine[b, op].item())
            s = float(self.op_start[b, op].item())
            d = float(self.op_end[b, op].item() - s)

            ax.broken_barh([(s, d)], (m - 0.4, 0.8), facecolors=cmap(j),
                           edgecolor='black', linewidth=0.5)
            ax.text(s + d / 2.0, m, f"J{j}S{int(self.op_stage[op].item())}",
                    ha='center', va='center', fontsize=7, color='white', weight='bold')

        ax.set_yticks(range(self.total_machines))
        ylabels = [f"S{int(self.machine_stage[m].item())}·M{int(self.machine_local[m].item())}"
                   for m in range(self.total_machines)]
        ax.set_yticklabels(ylabels)
        ax.invert_yaxis()
        ax.set_xlabel('Time'); ax.set_ylabel('Machine')
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        ax.axvline(x=ms_val, color='red', linestyle='--', linewidth=1.0)
        ax.set_title(f"{title or f'HFSP Schedule (batch {b})'}  |  Makespan = {ms_val:.2f}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)


# =====================================================
# Random agent
# =====================================================
def sample_random_actions(feasible_mask):
    weights = feasible_mask.float()
    rand = torch.rand_like(weights)
    rand.mul_(weights)
    return rand.argmax(dim=1)


def run_random_agent(env: HFSPGraphEnv, seed=0, B=1, verbose=False):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed, batch_size=B, identical_job=True)
    total_reward = torch.zeros(B, dtype=torch.long, device=env.device)
    T = env.num_ops
    for steps in range(T):
        actions = sample_random_actions(obs['feasible_mask'])
        if verbose and steps < 5:
            n_feas = obs['feasible_mask'].sum(dim=1)
            print(f"  step={steps:3d}  |feasible|/batch={n_feas[:5].tolist()}...")
        obs, reward, done = env.step(actions, last=(steps == T - 1))
        total_reward += reward
    makespans = -total_reward
    return makespans, T


def sample_sjf_v2_action(env: HFSPGraphEnv, obs):
    B = env.batch_size
    J = env.num_jobs
    M = env.total_machines

    proc_jm = env.edge_pool['proc_time']
    est_jm = env.edge_pool['est']

    score_jm = est_jm + 1e-4 * proc_jm

    edge_idx_3d = env.env_edge_lookup[env.active_op]
    valid_edge_3d = edge_idx_3d >= 0
    safe_idx_3d = edge_idx_3d.clamp(min=0)

    is_feasible_jm = obs['feasible_mask'].gather(1, safe_idx_3d.view(B, -1)).view(B, J, M)
    is_feasible_jm = is_feasible_jm & valid_edge_3d

    masked_score_jm = torch.where(is_feasible_jm, score_jm, env._CONST_INF_F)

    best_idx_flat = masked_score_jm.view(B, -1).argmin(dim=1)
    actions = safe_idx_3d.view(B, -1).gather(1, best_idx_flat.unsqueeze(1)).squeeze(1)

    return actions


def run_sjf_agent(env: HFSPGraphEnv, seed=0, B=1, verbose=False):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed, batch_size=B, identical_job=True)
    total_reward = torch.zeros(B, dtype=torch.long, device=env.device)
    T = env.num_ops
    for steps in range(T):
        actions = sample_sjf_v2_action(env, obs)
        if verbose and steps < 5:
            n_feas = obs['feasible_mask'].sum(dim=1)
            print(f"  step={steps:3d}  |feasible|/batch={n_feas[:5].tolist()}...")
        obs, reward, done = env.step(actions, last=(steps == T - 1))
        total_reward += reward
    makespans = -total_reward
    return makespans, T


if __name__ == "__main__":
    import time
    BATCH_SIZE = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = HFSPGraphEnv(
        num_jobs=25,
        machine_cnt_list=[5, 3, 7, 3, 5, 7],
        time_low=3, time_high=22,
        device=device
    )

    print(f"#ops={env.num_ops}  #machines={env.total_machines}  "
          f"#edges={env.num_edges}  batch={BATCH_SIZE}")

    # Warm-up
    _, _ = run_sjf_agent(env, seed=0, B=2, verbose=False)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    makespans, steps = run_sjf_agent(env, seed=42, B=BATCH_SIZE, verbose=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"\nbatched run: steps={steps}  elapsed={dt:.3f}s  ({BATCH_SIZE/dt:.1f} ep/s)")
    print(f"makespan  mean={makespans.float().mean().item():.2f}  "
          f"min={makespans.min().item():.2f}  "
          f"max={makespans.max().item():.2f}  "
          f"std={makespans.float().std().item():.2f}")

    best_idx = int(makespans.argmin().item())
    env.render_schedule(
        batch_idx=best_idx,
        save_path="gantt_sjf_torch.png",
        title=f"SJF Agent (torch, batch {best_idx})",
    )
    print(f"[render] saved gantt_sjf_torch.png  (batch={best_idx}, "
          f"makespan={makespans[best_idx].item()})")