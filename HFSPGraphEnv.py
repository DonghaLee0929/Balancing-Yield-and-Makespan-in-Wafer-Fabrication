"""
HFSP Graph-based RL Environment — PyTorch Vectorized (Fast for Large Batches)
"""

from __future__ import annotations
import torch

# 기존 FFSProblemDef 구조 유지 (내부적으로 리스트 반환 시 PyTorch 텐서 기반으로 가정)
from FFSProblemDef import get_random_problems, get_random_problems_identical_jobs


def derive_active_op(op_status: torch.Tensor, num_jobs: int, num_stages: int):
    """각 (B, j) 의 current active op = status != done 인 가장 이른 stage.
    전부 done 이면 마지막 stage 를 쓰면서 is_done=True (done 표식으로 사용)."""
    B = op_status.size(0)
    js = op_status.view(B, num_jobs, num_stages)
    not_done = js != 3
    has_active = not_done.any(dim=2)
    first_active_stage = not_done.to(torch.uint8).argmax(dim=2)
    active_stage = torch.where(has_active, first_active_stage,
                                torch.tensor(num_stages - 1, device=op_status.device))
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

        # ── 노드 메타 (batch-shared) ────────────────────────────
        self.num_ops = self.num_jobs * self.num_stages
        
        op_job = torch.arange(self.num_jobs).repeat_interleave(self.num_stages)
        op_stage = torch.arange(self.num_stages).repeat(self.num_jobs)
        self.op_job = op_job.to(self.device)
        self.op_stage = op_stage.to(self.device)

        # ── machine 메타 + offset (batch-shared) ────────────────
        mcl = torch.tensor(self.machine_cnt_list, dtype=torch.long, device=self.device)
        self.machine_offset = torch.cat([mcl.new_zeros(1), mcl.cumsum(0)[:-1]])
        self.machine_stage = torch.arange(self.num_stages, device=self.device).repeat_interleave(mcl)
        self.machine_local = (torch.arange(self.total_machines, device=self.device)
                              - self.machine_offset[self.machine_stage])

        # ── 엣지 메타 (batch-shared) ────────────────────────────
        # nonzero 는 row-major 라 (op asc, m asc) 순서 보장 → 기존 이중 for-loop 와 동일 순서.
        pairs = (self.op_stage.unsqueeze(1) == self.machine_stage.unsqueeze(0)).nonzero(as_tuple=False)
        self.edge_op = pairs[:, 0].contiguous()
        self.edge_machine = pairs[:, 1].contiguous()
        self.num_edges = self.edge_op.numel()

        # Static (op, machine) → edge idx 룩업 (-1 = no edge). feature pool 과 외부 action mapping 공용.
        self.env_edge_lookup = torch.full(
            (self.num_ops, self.total_machines), -1, dtype=torch.long, device=self.device)
        self.env_edge_lookup[self.edge_op, self.edge_machine] = torch.arange(self.num_edges, device=self.device)

        # (J, M) → edge idx : job j 의 "machine m 의 stage 에 해당하는 operation" 의 엣지.
        # 모든 (j, m) 에서 항상 유효(>=0). proc_time 채널을 active op 한정이 아니라 모든 stage
        # (과거/현재/미래) 의 처리시간으로 노출하기 위한 정적 룩업.
        job_arange0 = torch.arange(self.num_jobs, device=self.device)
        op_at_m_stage = job_arange0.unsqueeze(1) * self.num_stages + self.machine_stage.unsqueeze(0)   # (J, M)
        m_arange0 = torch.arange(self.total_machines, device=self.device)
        self.full_edge_idx = self.env_edge_lookup[op_at_m_stage, m_arange0.unsqueeze(0)]               # (J, M)

        # job-wise predecessor (batch-shared) — stage 0 은 -1
        arange_ops = torch.arange(self.num_ops, device=self.device)
        self.prev_op_in_job = torch.where(self.op_stage > 0, arange_ops - 1,
                                           torch.full_like(arange_ops, -1))
        self._has_prev = self.prev_op_in_job >= 0
        self._pred_safe = torch.where(self._has_prev, self.prev_op_in_job, torch.zeros_like(self.prev_op_in_job))

        # sample_edge_proc_time 용 src 인덱스 precompute.
        # concat([stage_pt[s].reshape(B,-1) for s])[:, _proc_src_idx] == edge_proc_time 순서.
        # edge e: src = J·machine_offset[s(e)] + j(e)·cnt(s(e)) + m_local(e).
        s_e = self.machine_stage[self.edge_machine]
        j_e = self.edge_op // self.num_stages
        self._proc_src_idx = (self.num_jobs * self.machine_offset[s_e]
                              + j_e * mcl[s_e]
                              + self.machine_local[self.edge_machine])

        # ── 동적 상태 (reset에서 채움) — leading dim = B ──
        self.batch_size = None
        self.edge_proc_time = None       # (B, num_edges)
        self.per_op_min_proc = None      # (B, num_ops) — min over compatible machines, rollout-constant
        self.op_assigned = None          # (B, num_ops) bool
        self.op_machine = None           # (B, num_ops) long
        self.op_proc_chosen = None       # (B, num_ops) long
        self.op_start = None             # (B, num_ops) long
        self.op_end = None               # (B, num_ops) long
        
        self.machine_iv_start = None     # (B, total_machines, num_jobs)
        self.machine_iv_end = None       # (B, total_machines, num_jobs)
        self.machine_iv_count = None     # (B, total_machines) long
        
        self.feasible_mask = None        # (B, num_edges) bool — step 의 action validation 용
        self.done = None                 # (B,) bool

        # ── precomputed feature pools (quality 채널 제외) ──
        # wrapper 가 wafer_quality (row), quality_zscore (edge) 두 채널만 덧붙여 ModelState 조립.
        # row : active_stage, active_stage_done, active_stage_s{s}, recent_end, remaining_proc
        # col : stage_id, stage_id_s{s}, idle, time_to_idle, iv_count, utilization
        # edge: proc_time, est, eft   (est/eft 는 min_est 로 re-anchor 된 값)
        self.row_pool = None             # dict[str, (B, J, ...)]
        self.col_pool = None             # dict[str, (B, M, ...)]
        self.edge_pool = None            # dict[str, (B, J, M)]
        self.active_op = None            # (B, J) long
        self.valid_edge_3d = None        # (B, J, M) bool — wrapper 가 edge_feat masked_fill 에 사용
        self.edge_mask_ninf = None       # (B, M, J) float — 0/-inf attention mask

        # 성능 최적화를 위한 미리 할당된 상수/인덱스 텐서
        self._INT64_MAX = torch.iinfo(torch.long).max

    # =====================================================
    # Problem sampling
    # =====================================================
    def sample_edge_proc_time(self, B, seed, identical_job=False):
        # device=self.device 로 처음부터 GPU 메모리 안에서 난수 생성 → CPU→GPU 복사 제거.
        # stage list → edge order 매핑은 __init__ 의 _proc_src_idx 에 위임 → 단일 gather.
        fn = get_random_problems_identical_jobs if identical_job else get_random_problems
        pts = fn(
            batch_size=B, stage_cnt=self.num_stages,
            machine_cnt_list=self.machine_cnt_list, job_cnt=self.num_jobs,
            process_time_params={'time_low': self.time_low, 'time_high': self.time_high},
            seed=seed, device=self.device,
        )
        concat = torch.cat([pt.reshape(B, -1) for pt in pts], dim=1)            # (B, NE)
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
        
        if edge_proc_time is not None:
            ept = edge_proc_time.to(self.device).long()
            if ept.shape != (B, self.num_edges):
                raise ValueError(f"edge_proc_time shape {ept.shape} != ({B}, {self.num_edges})")
            self.edge_proc_time = ept
        else:
            self.edge_proc_time = self.sample_edge_proc_time(B, seed, identical_job)

        # 각 op의 호환 머신 중 최소 처리시간 — rollout 내내 불변이라 precompute.
        self.per_op_min_proc = torch.full((B, self.num_ops), float('inf'), device=self.device)
        self.per_op_min_proc.scatter_reduce_(
            1,
            self.edge_op.unsqueeze(0).expand(B, -1),
            self.edge_proc_time.float(),
            reduce='amin',
            include_self=False,
        )

        self.op_assigned = torch.zeros((B, self.num_ops), dtype=torch.bool, device=self.device)
        self.op_machine = torch.full((B, self.num_ops), -1, dtype=torch.long, device=self.device)
        self.op_proc_chosen = torch.full((B, self.num_ops), -1, dtype=torch.long, device=self.device)
        self.op_start = torch.full((B, self.num_ops), -1, dtype=torch.long, device=self.device)
        self.op_end = torch.full((B, self.num_ops), -1, dtype=torch.long, device=self.device)
        
        self.machine_iv_start = torch.full((B, self.total_machines, self.num_jobs), self._INT64_MAX, dtype=torch.long, device=self.device)
        self.machine_iv_end = torch.full((B, self.total_machines, self.num_jobs), self._INT64_MAX, dtype=torch.long, device=self.device)
        self.machine_iv_count = torch.zeros((B, self.total_machines), dtype=torch.long, device=self.device)

        self.done = torch.zeros(B, dtype=torch.bool, device=self.device)

        self._refresh_feasibility()
        self._compute_feature_pools()
        return self._get_obs()

    # =====================================================
    # Feasibility
    # =====================================================
    def _op_ready_array(self):
        pred_assigned = self.op_assigned.gather(1, self._pred_safe.unsqueeze(0).expand(self.batch_size, -1))
        pred_ok = torch.where(self._has_prev.unsqueeze(0), pred_assigned, torch.tensor(True, device=self.device))
        return (~self.op_assigned) & pred_ok

    def _refresh_feasibility(self):
        self._op_ready_cache = self._op_ready_array()
        # [:, self.edge_op] 와 동일
        self.feasible_mask = self._op_ready_cache.gather(1, self.edge_op.unsqueeze(0).expand(self.batch_size, -1))

    # =====================================================
    # Feature pools (quality 채널 제외) — wrapper 가 그대로 가져다 씀
    # =====================================================
    def _compute_feature_pools(self):
        """row/col/edge pool + valid_edge_3d + edge_mask_ninf 를 한 번에 빌드.

        Wrapper 의 build_model_state 는 여기서 만든 dict 를 가져와 wafer_quality / quality_zscore
        만 덧붙여 stack 한다. 모든 시간 채널은 min_est (= feasible edge 의 min EST) 로 re-anchor.

        Pool 이 실제로 쓰는 (B, J, M) 영역만 계산한다 — 전체 (B, NE) edge_est/edge_eft 는 만들지 않음.
        """
        B = self.batch_size
        J = self.num_jobs
        M = self.total_machines
        S = self.num_stages
        device = self.device

        # ── active op per job ──
        op_status = self._op_status()
        active_op, is_done = derive_active_op(op_status, J, S)
        self.active_op = active_op

        # ── (active op × machine) edge index — valid_edge_3d / min_est / edge_mask 공유 ──
        edge_idx_3d = self.env_edge_lookup[active_op]                                 # (B, J, M)
        valid_edge_3d = edge_idx_3d >= 0

        # proc_time : (j, m) 별 "머신 m 의 stage 의 op" 처리시간 — 모든 (j, m) 에서 항상 유효.
        # active stage 칸에선 active op 의 처리시간과 동일하므로 아래 slot fitting 에도 그대로 사용.
        # (무효 칸은 found=True 로 루프 본체를 건너뛰어 이 값이 est 에 안 쓰이고, eft 는 wrapper 에서
        #  -1 로 마스킹되므로 active-op 한정 proc 를 따로 gather 할 필요가 없음.)
        proc_jm_full = self.edge_proc_time[:, self.full_edge_idx].float()             # (B, J, M)

        # ── EST/EFT at active_op (slot fitting per (b, m), J 축은 broadcast) ──
        # 무효 칸(stage 불일치)은 found=True 로 시작해 루프 본체 무시 → 낭비 FLOP 최소화.
        pred_safe_a = self._pred_safe[active_op]                                      # (B, J)
        has_prev_a = self._has_prev[active_op]                                        # (B, J)
        pred_end_a = self.op_end.gather(1, pred_safe_a)                               # (B, J)
        pred_assigned_a = self.op_assigned.gather(1, pred_safe_a)                     # (B, J)
        pred_ready = has_prev_a & pred_assigned_a
        pred_end_safe_a = torch.where(pred_ready, pred_end_a,
                                       torch.zeros_like(pred_end_a)).float()          # (B, J)
        t_min_jm = pred_end_safe_a.unsqueeze(-1).expand(-1, -1, M).contiguous()       # (B, J, M)

        # Sync-free slot fitting: 동적 max_iv 슬라이스 대신 전체 J 슬롯을 항상 순회.
        # storage 자체가 (B, M, J) 로 패딩돼있어 추가 alloc 없고, k >= cnt 슬롯은 valid_k=False
        # 로 자동 무시됨. 매 step .item()/found.all() sync 제거.
        iv_cnt_b = self.machine_iv_count.unsqueeze(1)                                 # (B, 1, M)
        iv_s = self.machine_iv_start.float().unsqueeze(1)                             # (B, 1, M, J)
        iv_e = self.machine_iv_end.float().unsqueeze(1)

        cursor = t_min_jm.clone()
        slot_start = torch.zeros_like(cursor)
        found = ~valid_edge_3d                                                        # 무효 칸은 즉시 "찾았다" 처리

        for k in range(J):
            s_k = iv_s[..., k]                                                        # (B, 1, M) → broadcast (B, J, M)
            e_k = iv_e[..., k]
            valid_k = k < iv_cnt_b
            fits = (~found) & valid_k & (s_k - cursor >= proc_jm_full)
            slot_start = torch.where(fits, cursor, slot_start)
            found = found | fits
            advance = (~found) & valid_k
            cursor = torch.where(advance, torch.maximum(cursor, e_k), cursor)

        est_jm = torch.where(found, slot_start, cursor)                               # (B, J, M)
        eft_jm = est_jm + proc_jm_full

        # ── min_est = min EST over feasible (= valid_edge_3d & ~is_done) ──
        # active_op 는 항상 status=1 (ready) 이므로 feasibility 는 valid & ~is_done 와 동치.
        feas_for_min = valid_edge_3d & (~is_done).unsqueeze(-1)
        inf_t = torch.tensor(float('inf'), device=device)
        min_est = torch.where(feas_for_min, est_jm, inf_t).amin(dim=(1, 2))
        min_est = torch.where(torch.isinf(min_est), torch.zeros_like(min_est), min_est)

        # ── row pool ──
        # active_stage ∈ [0, S-1] 진행 중, S = 완료 표식 (one-hot 마지막 채널이 done).
        arange_J = torch.arange(J, device=device)
        active_stage_internal = active_op - arange_J.unsqueeze(0) * S
        active_stage_feat = torch.where(is_done,
                                         torch.tensor(S, device=device),
                                         active_stage_internal).float()
        active_stage_oh = torch.nn.functional.one_hot(active_stage_feat.long(),
                                                       num_classes=S + 1).float()

        op_end_safe = torch.where(self.op_assigned, self.op_end.float(), 0.0)
        recent_end = op_end_safe.view(B, J, S).max(dim=2)[0]
        recent_end_rel = recent_end - min_est.unsqueeze(1)

        # remaining_proc (job-level): Σ_{s} ( scheduled ? PT_remaining_on_assigned_machine : min_{m∈M_ijs} P_ijsm )
        # — assigned 쪽은 op_end - max(op_start, min_est) 로 시작 전 대기시간을 제외한 순수 처리시간만 카운트.
        #   (시작 전: full proc_time, 처리 중: op_end-min_est, 완료: 0).
        # — unassigned 는 호환 머신 중 최소 처리시간 (낙관적 lower bound).
        op_anchor = torch.maximum(self.op_start.float(), min_est.unsqueeze(1))
        op_end_rem = (self.op_end.float() - op_anchor).clamp(min=0.0)
        per_op_rem = torch.where(self.op_assigned, op_end_rem, self.per_op_min_proc)
        remaining_proc = per_op_rem.view(B, J, S).sum(dim=2)
        # 전 stage 완료(is_done) job 은 둘 결정도 없고 edge 도 전부 마스킹되므로 잔여 0 강제.
        # (op_end > min_est 인 늦게 끝난 job 이 frontier 기준 양수 잔여를 남기는 leak 차단.)
        remaining_proc = torch.where(is_done, torch.zeros_like(remaining_proc), remaining_proc)

        self.row_pool = {
            'active_stage':      active_stage_feat,
            'recent_end':        recent_end_rel,
            'remaining_proc':    remaining_proc,
            'active_stage_done': active_stage_oh[..., S],
            **{f'active_stage_s{s}': active_stage_oh[..., s] for s in range(S)},
        }

        # ── col pool ──
        # 평행 머신 대칭성 위해 머신별 ID 대신 stage 인덱스만 사용.
        stage_norm = self.machine_stage.float() / max(S - 1, 1)
        stage_oh_m = torch.nn.functional.one_hot(self.machine_stage, num_classes=S).float()
        stage_oh_bp = stage_oh_m.unsqueeze(0).expand(B, -1, -1)

        cnt = self.machine_iv_count
        iv_count_f = cnt.float()
        last_idx = torch.clamp(cnt - 1, 0, J - 1)
        b_arr = torch.arange(B, device=device).unsqueeze(1)
        m_arr = torch.arange(M, device=device).unsqueeze(0)
        last_end = self.machine_iv_end[b_arr, m_arr, last_idx]
        ready_t = torch.where(cnt > 0, last_end,
                               torch.tensor(0, dtype=torch.long, device=device)).float()

        idle = (ready_t <= min_est.unsqueeze(1)).float()
        time_to_idle = torch.clamp(ready_t - min_est.unsqueeze(1), min=0.0)

        k_range = torch.arange(J, device=device)
        valid_iv = k_range.view(1, 1, -1) < iv_count_f.unsqueeze(-1)
        iv_s_safe = torch.where(valid_iv, self.machine_iv_start.float(), 0.0)
        iv_e_safe = torch.where(valid_iv, self.machine_iv_end.float(), 0.0)
        busy = (iv_e_safe - iv_s_safe).sum(dim=2)
        util = torch.where(cnt > 0, busy / torch.clamp(ready_t, min=1e-9), torch.zeros_like(busy))

        self.col_pool = {
            'stage_id':     stage_norm.unsqueeze(0).expand(B, -1),
            'idle':         idle,
            'time_to_idle': time_to_idle,
            'iv_count':     iv_count_f,
            'utilization':  util,
            **{f'stage_id_s{s}': stage_oh_bp[..., s] for s in range(S)},
        }

        # ── edge pool ──
        # proc_time : (j, m) 별 "머신 m 의 stage 의 op" 처리시간 (proc_jm_full) — 모든 (j,m) 에서
        #             항상 유효(과거/현재/미래 stage 전부). wrapper 에서 마스킹하지 않음.
        # est/eft   : active op 기준 슬롯 fitting 값 (min_est 로 re-anchor). 실제 선택 가능한 엣지
        #             (= valid_edge_3d) 외 칸은 의미 없으므로 wrapper 가 -1 로 패딩.
        self.edge_pool = {
            'proc_time': proc_jm_full,
            'est':       est_jm - min_est.view(B, 1, 1),
            'eft':       eft_jm - min_est.view(B, 1, 1),
        }
        self.valid_edge_3d = valid_edge_3d

        # ── edge mask: (B, M, J), 0=feasible / -inf=blocked. done 행은 풀어 softmax NaN 방지. ──
        feas_3d = valid_edge_3d & (~is_done).unsqueeze(-1)
        feas_mt = feas_3d.transpose(1, 2)
        feas_mt = feas_mt | self.done.view(B, 1, 1)
        ninf_mask = torch.zeros_like(feas_mt, dtype=torch.float32)
        ninf_mask.masked_fill_(~feas_mt, float('-inf'))
        self.edge_mask_ninf = ninf_mask

    # =====================================================
    # Step (Pure PyTorch Vectorized) — sync-free hot path.
    # 호출자가 총 num_ops 번 step 을 호출한다는 전제 (HFSP 는 J·S 회 고정).
    # `while not done.all()` 대신 `for t in range(env.num_ops)` 로 굴려야 sync 가 사라짐.
    # =====================================================
    def step(self, actions, last=False, assume_all_active=True, validate=False):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        else:
            actions = actions.to(self.device).long().view(-1)

        if actions.shape[0] != self.batch_size:
            raise ValueError(f"actions length {actions.shape[0]} != batch_size {self.batch_size}")

        prev_done = self.done.clone()
        b_idx = torch.arange(self.batch_size, device=self.device)

        # Infeasibility 검증은 GPU→CPU sync 를 강제하므로 기본 off. 디버그용으로만 켜 사용.
        if validate:
            active_v = ~self.done
            chosen_feasible = self.feasible_mask[b_idx, actions]
            bad = active_v & (~chosen_feasible)
            if bad.any():
                idx = b_idx[bad].cpu().tolist()
                raise ValueError(f"Infeasible action for batch indices {idx}")

        if assume_all_active:
            # Fast path — fixed-step caller 패턴에서 boolean indexing sync 제거.
            ab = b_idx
            a_act = actions
        else:
            # Safe path — 일부 batch 가 미리 done 인 경우 (caller 가 done batch 도 호출 가능).
            active = ~self.done
            ab = b_idx[active]
            a_act = actions[active]

        a_op = self.edge_op[a_act]
        a_m = self.edge_machine[a_act]
        a_proc = self.edge_proc_time[ab, a_act]

        has_prev = self._has_prev[a_op]
        prev_op = self._pred_safe[a_op]
        t_min = torch.where(has_prev, self.op_end[ab, prev_op],
                            torch.tensor(0, dtype=torch.long, device=self.device))

        ivs_s = self.machine_iv_start[ab, a_m]
        ivs_e = self.machine_iv_end[ab, a_m]
        cnt = self.machine_iv_count[ab, a_m]

        N = ab.shape[0]
        cursor = t_min.clone()
        slot_start = torch.zeros(N, dtype=torch.long, device=self.device)
        found = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Slot fitting — 고정 J-step 루프. 기존 `cnt.max().item()` 과 `found.all() break` 두 sync 제거.
        # storage 가 (B, M, J) 로 패딩돼있어 k >= cnt 슬롯은 valid_k=False 로 자동 무시됨.
        for k in range(self.num_jobs):
            s_iv = ivs_s[:, k]
            e_iv = ivs_e[:, k]
            valid_k = k < cnt
            fits = (~found) & valid_k & (s_iv - cursor >= a_proc)
            slot_start = torch.where(fits, cursor, slot_start)
            found |= fits
            advance = (~found) & valid_k
            cursor = torch.where(advance, torch.maximum(cursor, e_iv), cursor)

        slot_start = torch.where(found, slot_start, cursor)
        end_t = slot_start + a_proc

        self.op_assigned[ab, a_op] = True
        self.op_machine[ab, a_op] = a_m
        self.op_proc_chosen[ab, a_op] = a_proc
        self.op_start[ab, a_op] = slot_start
        self.op_end[ab, a_op] = end_t

        # Left-shift gap-filling — 동적 limit=max_cnt+1 대신 full-J 윈도우.
        # cnt 이후 슬롯은 sentinel (_INT64_MAX) 라 pos 계산/gather 결과가 sentinel→sentinel 으로 자기복사 → no-op.
        k_idx = torch.arange(self.num_jobs, device=self.device).unsqueeze(0)
        pos = (ivs_s < slot_start.unsqueeze(1)).sum(dim=1)
        pos_b = pos.unsqueeze(1)
        is_after = k_idx > pos_b
        is_at = k_idx == pos_b

        src = torch.where(is_after, k_idx - 1, k_idx).clamp(0, self.num_jobs - 1)
        new_s = ivs_s.gather(1, src)
        new_e = ivs_e.gather(1, src)
        new_s = torch.where(is_at, slot_start.unsqueeze(1), new_s)
        new_e = torch.where(is_at, end_t.unsqueeze(1), new_e)

        self.machine_iv_start[ab.unsqueeze(1), a_m.unsqueeze(1), k_idx] = new_s
        self.machine_iv_end[ab.unsqueeze(1), a_m.unsqueeze(1), k_idx] = new_e
        self.machine_iv_count[ab, a_m] = cnt + 1

        self.done = self.done | self.op_assigned.all(dim=1)

        # Reward — `if just_done.any():` 분기와 boolean indexing 모두 제거. ms 를 무조건 계산하고
        # where 로 마스킹. 비-done batch 의 op_end.max() 는 어차피 reward 에 들어가지 않으므로 무해.
        just_done = self.done & (~prev_done)
        ms_all = self.op_end.max(dim=1)[0]
        reward = torch.where(just_done, -ms_all, torch.zeros_like(ms_all))

        # last step 에선 feasibility/feature_pool 재계산이 어차피 무쓸모 — caller 가 obs 를 안 씀.
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
        
        # CPU로 옮겨서 렌더링
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
            
            ax.broken_barh([(s, d)], (m - 0.4, 0.8), facecolors=cmap(j), edgecolor='black', linewidth=0.5)
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
# Random agent (Fully on PyTorch)
# =====================================================
def sample_random_actions(feasible_mask):
    weights = feasible_mask.float()
    rand = torch.rand_like(weights) * weights
    actions = rand.argmax(dim=1)
    return actions


def run_random_agent(env: HFSPGraphEnv, seed=0, B=1, verbose=False):
    # PyTorch 시드 고정은 전역/제너레이터로 관리 가능
    torch.manual_seed(seed)
    obs = env.reset(seed=seed, batch_size=B, identical_job=True)
    total_reward = torch.zeros(B, dtype=torch.long, device=env.device)

    # 총 step 수 = env.num_ops (J·S) 로 고정 — `while not env.done.all()` 의 매 step sync 제거.
    T = env.num_ops
    for steps in range(T):
        actions = sample_random_actions(obs['feasible_mask'])
        if verbose and steps < 5:
            n_feas = obs['feasible_mask'].sum(dim=1)
            print(f"  step={steps:3d}  |feasible|/batch={n_feas[:5].tolist()}...") # 너무 길면 생략

        obs, reward, done = env.step(actions, last=(steps == T - 1))
        total_reward += reward

    makespans = -total_reward
    return makespans, T

def sample_est_action(env: HFSPGraphEnv, obs):
    """
    EST (Time-driven 환경 완벽 에뮬레이션)
    1. 가장 일찍 시작할 수 있는(비어있는) 기계를 최우선으로 찾음 (est_jm)
    2. 만약 동시에 비어있는 기계가 여러 개라면, 처리 시간이 짧은 기계 선택 (proc_jm)
    """
    B = env.batch_size
    J = env.num_jobs
    M = env.total_machines
    
    # 1. EST(Earliest Start Time)와 Proc 특징 가져오기
    proc_jm = env.edge_pool['proc_time']  # (B, J, M)
    est_jm = env.edge_pool['est']         # (B, J, M)
    
    # 2. 핵심 수정 포인트: EST를 메인 점수로, 처리 시간을 미세 가중치로!
    # - EST가 0인(당장 비어있는) 기계들끼리는 proc_jm에 의해 짧은 작업이 승리함
    # - 특정 기계에 작업이 들어가 EST가 1이라도 커지면, 무조건 다른 빈 기계로 넘어감
    score_jm = est_jm + 1e-4 * proc_jm
    
    # 3. Feasible Mask를 3D 공간(B, J, M)으로 매핑
    edge_idx_3d = env.env_edge_lookup[env.active_op] 
    valid_edge_3d = edge_idx_3d >= 0
    safe_idx_3d = torch.where(valid_edge_3d, edge_idx_3d, torch.zeros_like(edge_idx_3d))
    
    is_feasible_jm = obs['feasible_mask'].gather(1, safe_idx_3d.view(B, -1)).view(B, J, M)
    is_feasible_jm = is_feasible_jm & valid_edge_3d
    
    # 4. Infeasible 엣지는 무한대(INF)로 마스킹하여 선택 방지
    INF = torch.tensor(float('inf'), device=env.device)
    masked_score_jm = torch.where(is_feasible_jm, score_jm, INF)
    
    # 5. (B, J, M) 공간에서 최소 점수를 가진 인덱스 추출 및 복원
    best_idx_flat = masked_score_jm.view(B, -1).argmin(dim=1) 
    actions = safe_idx_3d.view(B, -1).gather(1, best_idx_flat.unsqueeze(1)).squeeze(1)
    
    return actions

def run_est_agent(env: HFSPGraphEnv, seed=0, B=1, verbose=False):
    torch.manual_seed(seed)
    obs = env.reset(seed=seed, batch_size=B, identical_job=True)
    total_reward = torch.zeros(B, dtype=torch.long, device=env.device)

    # 총 step 수 = env.num_ops (J * S)
    T = env.num_ops
    for steps in range(T):
        # Random 대신 EST action 선택
        actions = sample_est_action(env, obs)
        
        if verbose and steps < 5:
            n_feas = obs['feasible_mask'].sum(dim=1)
            print(f"  step={steps:3d}  |feasible|/batch={n_feas[:5].tolist()}...") 

        obs, reward, done = env.step(actions, last=(steps == T - 1))
        total_reward += reward

    makespans = -total_reward
    return makespans, T


if __name__ == "__main__":
    import time
    BATCH_SIZE = 1000 # 대규모 배치 설정

    # GPU 강제 할당 테스트 (없으면 CPU로 동작)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = HFSPGraphEnv(
        num_jobs=25,
        machine_cnt_list=[5,3,7,3,5,7],
        time_low=3, time_high=22,
        device=device
    )
    
    print(f"#ops={env.num_ops}  #machines={env.total_machines}  "
          f"#edges={env.num_edges}  batch={BATCH_SIZE}")

    # Warm-up (최초 JIT 컴파일/CUDA 초기화 비용 제외용)
    _, _ = run_est_agent(env, seed=0, B=2, verbose=False)

    t0 = time.time()
    makespans, steps = run_est_agent(env, seed=42, B=BATCH_SIZE, verbose=True)
    dt = time.time() - t0
    
    print(f"\nbatched run: steps={steps}  elapsed={dt:.3f}s  ({BATCH_SIZE/dt:.1f} ep/s)")
    print(f"makespan  mean={makespans.float().mean().item():.2f}  "
          f"min={makespans.min().item():.2f}  "
          f"max={makespans.max().item():.2f}  "
          f"std={makespans.float().std().item():.2f}")

    # 가장 좋은(makespan 최소) 배치 원소를 골라 Gantt 저장
    best_idx = int(makespans.argmin().item())
    env.render_schedule(
        batch_idx=best_idx,
        save_path="gantt_est_torch.png",
        title=f"EST Agent (torch, batch {best_idx})",
    )
    print(f"[render] saved gantt_est_torch.png  (batch={best_idx}, "
          f"makespan={makespans[best_idx].item()})")