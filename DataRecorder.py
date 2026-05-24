"""WandB-backed training recorder for HFSP ASIL.

- scalar 메트릭은 mini-batch 마다 평균 누적 → epoch 마다 flush.
- feature saliency: sil_pomo_rollout 이 watch_step(mini_state) 호출 시,
  row/col/edge feature 를 leaf 텐서로 바꿔 모아두고, backward 후 collect_saliency()
  로 채널별 |∂loss/∂feat| 평균을 기록.
- wandb 미설치 / 미로그인 시 graceful no-op.
"""
from __future__ import annotations
from typing import Optional
import torch

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


class DataRecorder:
    def __init__(self,
                 project: str = 'hfsp-asil',
                 run_name: Optional[str] = None,
                 config: Optional[dict] = None,
                 feat_names: Optional[dict] = None,
                 enabled: bool = True):
        self.enabled = enabled and _WANDB_OK
        if enabled and not _WANDB_OK:
            print('[DataRecorder] wandb 미설치 — 로깅 비활성')
        self.feat_names = feat_names or {}
        self._buf: dict = {}       # one-shot (eval/, time/, lr 등)
        self._acc: dict = {}       # {key: (sum, count)} — flush 시 평균
        self._watched: list = []   # [(rf, cf, ef), ...]
        self._ent_sum = None       # policy entropy GPU 누적 (per-step sync 회피)
        self._ent_n = 0
        if self.enabled:
            wandb.init(project=project, name=run_name, config=config)

    # ── HFSPWrapper.sil_pomo_rollout 가 매 step Phase 2 에서 호출 ──
    def watch_step(self, mini_state):
        if not self.enabled:
            return
        rf = mini_state.row_feature.detach().requires_grad_(True)
        cf = mini_state.col_feature.detach().requires_grad_(True)
        ef = mini_state.edge_feature.detach().requires_grad_(True)
        lf = mini_state.lambdas.detach().requires_grad_(True)
        mini_state.row_feature, mini_state.col_feature, mini_state.edge_feature = rf, cf, ef
        mini_state.lambdas = lf
        self._watched.append((rf, cf, ef, lf))

    # ── train_ASIL 호출: per mini-batch ──
    def log_step(self, loss, G_ms, G_q, r_bp, lambdas_b, log_prob_sum_best, weight):
        """ms / q / r / weight / λ-q 진단을 한 번에 누적.

        Args:
            G_ms:        (BP,) env return = -makespan
            G_q:         (BP,) predicted yield (mean per instance over jobs)
            r_bp:        (B, P) scalarized reward r = (1-λ)·m̂ + λ·q̂
            lambdas_b:   (B,) instance-level λ
        """
        if not self.enabled:
            return
        B = weight.shape[0]
        P = G_ms.shape[0] // B
        with torch.no_grad():
            ms = (-G_ms).float()
            ms_bp = ms.view(B, P)
            q = G_q.float()
            q_bp = q.view(B, P)

            r_std_inst = r_bp.std(dim=1)
            # λ vs (q, ms) Pearson — λ-conditioning 진단.
            # collapse → 둘 다 0. 정상 작동 → λ↑ 시 yield↑(+), makespan↑(+) 양쪽 양의 상관.
            # 한쪽만 양수면 비대칭 학습 (한 objective 만 λ 에 반응).
            q_inst  = q_bp.mean(dim=1)
            ms_inst = ms_bp.mean(dim=1)
            lam = lambdas_b.float()
            lam_c = lam - lam.mean()
            q_c   = q_inst  - q_inst.mean()
            ms_c  = ms_inst - ms_inst.mean()
            zero = torch.zeros((), device=lam.device)
            denom_q  = lam_c.norm() * q_c.norm()
            denom_ms = lam_c.norm() * ms_c.norm()
            corr_lam_q  = (lam_c * q_c).sum()  / denom_q  if denom_q  > 1e-8 else zero
            corr_lam_ms = (lam_c * ms_c).sum() / denom_ms if denom_ms > 1e-8 else zero

            # baseline degenerate 비율: 같은 (시나리오, λ) 의 P 샘플이 동일 reward → weight nan/0
            zero_frac = (r_std_inst < 1e-6).float().mean()

            vals = {
                'loss/mb_loss': loss.item(),
                # makespan
                'ms/mean': ms.mean().item(),
                'ms/best_per_inst': ms_bp.min(dim=1).values.mean().item(),
                'ms/std_per_inst': ms_bp.std(dim=1).mean().item(),
                'ms/range_per_inst': (ms_bp.max(dim=1).values - ms_bp.min(dim=1).values).mean().item(),
                # yield
                'q/mean': q.mean().item(),
                'q/best_per_inst': q_bp.max(dim=1).values.mean().item(),
                'q/std_per_inst': q_bp.std(dim=1).mean().item(),
                'q/range_per_inst': (q_bp.max(dim=1).values - q_bp.min(dim=1).values).mean().item(),
                # scalarized reward
                'r/mean': r_bp.mean().item(),
                'r/std_per_inst': r_std_inst.mean().item(),
                # baseline / weight
                'weight/mean': weight.mean().item(),
                'weight/max': weight.max().item(),
                'weight/std': weight.std().item() if B > 1 else 0.0,
                'weight/zero_frac': zero_frac.item(),
                # λ-objective correlation (mode collapse diagnostic)
                'r_corr/lambda_vs_q': corr_lam_q.item(),
                'r_corr/lambda_vs_ms': corr_lam_ms.item(),
                'log_prob/mean_best': log_prob_sum_best.mean().item(),
                'log_prob/min_best': log_prob_sum_best.min().item(),
            }
        for k, v in vals.items():
            self._add(k, v)

    def log_anchors(self, m_min, m_max, q_min, q_max):
        """Scalarization normalization anchors — 학습 중 m̂/q̂ dynamic range 추적.

        range 가 좁아지면 (1-λ)·m̂ + λ·q̂ 의 λ-sensitivity 가 떨어져 mode collapse
        직격 원인이 됨. 매 iter 갱신되므로 step 단위 누적 후 epoch 평균.
        """
        if not self.enabled:
            return
        self._add('norm/m_min', float(m_min))
        self._add('norm/m_max', float(m_max))
        self._add('norm/m_range', float(m_max - m_min))
        self._add('norm/q_min', float(q_min))
        self._add('norm/q_max', float(q_max))
        self._add('norm/q_range', float(q_max - q_min))

    def collect_saliency(self):
        """매 mini-batch backward 직후 호출. row/col/edge/λ 채널별 |∂loss/∂feat| 평균.

        λ saliency 가 0 에 가까우면 모델이 conditioning 을 무시 → mode collapse 의 근본
        원인 후보. row/col/edge 평균 saliency 와 동일 척도로 비교 가능.
        """
        watched, self._watched = self._watched, []
        if not self.enabled or not watched:
            return
        row_sum = col_sum = edge_sum = None
        lam_sum_t = None
        n = 0
        for rf, cf, ef, lf in watched:
            if rf.grad is None:
                continue
            r = rf.grad.detach().abs().mean(dim=(0, 1))         # (row_feat_dim,)
            c = cf.grad.detach().abs().mean(dim=(0, 1))         # (col_feat_dim,)
            e = ef.grad.detach().abs().mean(dim=(0, 1, 2))      # (edge_feat_dim,)
            row_sum = r if row_sum is None else row_sum + r
            col_sum = c if col_sum is None else col_sum + c
            edge_sum = e if edge_sum is None else edge_sum + e
            if lf.grad is not None:
                l = lf.grad.detach().abs().mean()               # scalar
                lam_sum_t = l if lam_sum_t is None else lam_sum_t + l
            n += 1
        if n == 0:
            return
        row_avg = (row_sum / n).tolist()
        col_avg = (col_sum / n).tolist()
        edge_avg = (edge_sum / n).tolist()
        row_names = self.feat_names.get('row', [f'row_{i}' for i in range(len(row_avg))])
        col_names = self.feat_names.get('col', [f'col_{i}' for i in range(len(col_avg))])
        edge_names = self.feat_names.get('edge', [f'edge_{i}' for i in range(len(edge_avg))])
        for name, v in zip(row_names, row_avg):
            self._add(f'saliency/{name}', v)
        for name, v in zip(col_names, col_avg):
            self._add(f'saliency/{name}', v)
        for name, v in zip(edge_names, edge_avg):
            self._add(f'saliency/{name}', v)
        if lam_sum_t is not None:
            self._add('saliency/lambda', (lam_sum_t / n).item())

    def log_gradients(self, model):
        """clip 직전 호출 — 누적 grad 의 global L2 norm.

        파라미터별 .item() 으로 N 번 sync 하던 것을 _foreach_norm 으로 묶어
        sync 1 회 (stack→norm→item) 로 축소.
        """
        if not self.enabled:
            return
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        per_param_norms = torch._foreach_norm(grads)             # list of 0-dim tensors
        total = torch.linalg.vector_norm(torch.stack(per_param_norms)).item()
        self._add('grad/global_norm', total)

    def accumulate_entropy(self, probs):
        """Phase-1 sampling 의 step별 정책 엔트로피를 GPU 에 누적 (sync 없음).

        probs: (BP, A) joint (machine×job) softmax. 엔트로피가 떨어지면 P 샘플이 같은
        trajectory 로 수렴 → POMO baseline std→0 (weight/zero_frac 의 상류 원인).
        collect_entropy 가 micro-rollout 마다 1회 sync 로 평균을 기록.
        """
        if not self.enabled:
            return
        with torch.no_grad():
            p = probs.float()
            ent = -(p.clamp_min(1e-12).log() * p).sum(dim=-1).mean()
        self._ent_sum = ent if self._ent_sum is None else self._ent_sum + ent
        self._ent_n += 1

    def collect_entropy(self):
        """micro-rollout 마다 호출 — 누적된 step 엔트로피 평균을 1회 sync 로 기록."""
        if not self.enabled or self._ent_n == 0:
            return
        self._add('policy/entropy', (self._ent_sum / self._ent_n).item())
        self._ent_sum, self._ent_n = None, 0

    def log_moe(self, model):
        """CCO MoE 라우팅 헬스 — epoch 당 1회 (clip 직전, log_gradients 옆).

        CCOBlock 이 train-mode forward(=Phase-2 grad-replay)마다 게이트 질량·엔트로피를
        누적 → 여기서 pop. load_max↑ 또는 gate_entropy↓ → 라우팅이 소수 expert 로
        쏠림(expert collapse). id_expert_frac↑ → FF expert 들이 학습에 기여 안 함.
        """
        if not self.enabled:
            return
        from FFSPModel_SUB import CCOBlock
        for m in model.modules():
            if not isinstance(m, CCOBlock):
                continue
            stats = m.pop_routing_stats()
            if stats is None:
                return
            load = stats['load']
            for j, v in enumerate(load):
                tag = 'id' if j == len(load) - 1 else f'e{j}'
                self._add(f'moe/load_{tag}', v)
            self._add('moe/load_max', max(load))
            self._add('moe/gate_entropy', stats['entropy'])
            self._add('moe/id_expert_frac', load[-1])
            return  # 단일 CCO 블록 가정

    def log_pareto(self, hv_mean, hv_std, ms_at_lam0, q_at_lam1,
                   nd_mean, scatter_path,
                   hv_norm_mean, hv_norm_std,
                   d_ms=None, d_q=None):
        """λ-sweep 평가 결과: HV (메인), normalized HV, endpoint 성능, ND 다양성, scatter.

        hv_norm = 단위 박스(makespan [100, hv_ref_m] × yield [0, hv_ref_q])로 정규화한 HV
                  → run 간 비교 가능한 [0,1] 스케일.
        d_ms = ms@0 - ms@1 (proper λ-conditioning 이면 음수, collapse → 0).
        d_q  = q@1  - q@0  (proper λ-conditioning 이면 양수, collapse → 0).
        """
        if not self.enabled:
            return
        self._buf.update({
            'eval/hv_mean': float(hv_mean),
            'eval/hv_std': float(hv_std),
            'eval/hv_norm_mean': float(hv_norm_mean),  # 정규화 HV (단위 박스 기준)
            'eval/hv_norm_std': float(hv_norm_std),
            'eval/ms_at_lam0': float(ms_at_lam0),  # 순수 makespan 정책 평균
            'eval/q_at_lam1': float(q_at_lam1),    # 순수 yield 정책 평균
            'eval/nd_count': float(nd_mean),
            'eval/pareto': wandb.Image(scatter_path),
        })
        if d_ms is not None:
            self._buf['eval/d_ms'] = float(d_ms)
        if d_q is not None:
            self._buf['eval/d_q'] = float(d_q)

    def log_epoch(self, epoch_time, lr):
        if not self.enabled:
            return
        self._buf['time/epoch_sec'] = float(epoch_time)
        self._buf['lr'] = float(lr)

    def flush(self, step):
        if not self.enabled:
            return
        out = dict(self._buf)
        for k, (s, n) in self._acc.items():
            out[k] = s / n
        if out:
            wandb.log(out, step=step)
        self._buf, self._acc = {}, {}

    def finish(self):
        if self.enabled:
            wandb.finish()

    def _add(self, key, value):
        s, n = self._acc.get(key, (0.0, 0))
        self._acc[key] = (s + value, n + 1)
