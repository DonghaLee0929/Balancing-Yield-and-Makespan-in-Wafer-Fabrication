
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']
        # mix1 입력 채널 = dot_product_score (1) + edge_feature 채널 수
        # cost_mat 채널 수는 __init__ 시점에 고정되어야 하므로 외부에서 주입받는다.
        self.edge_feature_dim = int(model_params.get('edge_feature_dim', 1))
        mix1_in = 1 + self.edge_feature_dim

        mix1_weight = torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, mix1_in, ms_hidden_dim))
        mix1_bias = torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 1+edge_feature_dim, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)                 — 단일 채널 (legacy)
        #              or (batch, row_cnt, col_cnt, edge_feat_dim)  — 멀티 채널

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        # cost_mat 을 항상 4D (batch, row_cnt, col_cnt, edge_feat_dim) 로 정규화.
        if cost_mat.dim() == 3:
            cost_mat = cost_mat.unsqueeze(-1)
        edge_feat_dim = cost_mat.size(-1)

        cost_mat_score = cost_mat[:, None, :, :, :].expand(
            batch_size, head_num, row_cnt, col_cnt, edge_feat_dim)
        # shape: (batch, head_num, row_cnt, col_cnt, edge_feat_dim)

        dot_product_score_5d = dot_product_score.unsqueeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        all_scores = torch.cat([dot_product_score_5d, cost_mat_score], dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 1+edge_feat_dim)

        all_scores_transposed = all_scores.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, col_cnt, 1+edge_feat_dim)

        ms1 = torch.matmul(all_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1,2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat
    
# Mixed Multi-Head Cross-Attention
class Mixed_MultiHeadCrossAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        # cost_mat 채널 수는 __init__ 시점에 고정되어야 하므로 외부에서 주입받는다.
        self.edge_feature_dim = int(model_params.get('edge_feature_dim', 1))

        self.W_qe = nn.Linear(self.edge_feature_dim, head_num * model_params['qkv_dim'], bias=False)
        self.W_ke = nn.Linear(self.edge_feature_dim, head_num * model_params['qkv_dim'], bias=False)
        self.W_ve = nn.Linear(self.edge_feature_dim, head_num * model_params['qkv_dim'], bias=False)

    def forward(self, q, k, v, cost_mat, mask=None):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)                 — 단일 채널 (legacy)
        #              or (batch, row_cnt, col_cnt, edge_feat_dim)  — 멀티 채널
        # mask.shape:     (batch, row_cnt, col_cnt) 또는 broadcastable (..., row_cnt, col_cnt)
        #                  — ninf style: 가능 0 / 불가능 -inf (decoder 의 ninf_mask 와 동일 컨벤션)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        # cost_mat 을 항상 4D (batch, row_cnt, col_cnt, edge_feat_dim) 로 정규화.
        if cost_mat.dim() == 3:
            cost_mat = cost_mat.unsqueeze(-1)

        # 1. Edge 투영 및 차원 재배치
        q_edge = self.W_qe(cost_mat).view(batch_size, row_cnt, col_cnt, head_num, qkv_dim).permute(0, 3, 1, 2, 4)
        k_edge = self.W_ke(cost_mat).view(batch_size, row_cnt, col_cnt, head_num, qkv_dim).permute(0, 3, 1, 2, 4)
        v_edge = self.W_ve(cost_mat).view(batch_size, row_cnt, col_cnt, head_num, qkv_dim).permute(0, 3, 1, 2, 4)

        # 2. Additive Mixing & Score 계산
        q_total = q.unsqueeze(3) + q_edge
        k_total = k.unsqueeze(2) + k_edge

        score = (q_total * k_total).sum(dim=-1)
        mixed_scores = score / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        # 3. Stage feasibility masking (수식 5 의 m ∈ M_ij) — additive ninf style.
        if mask is not None:
            # (batch, 1, row_cnt, col_cnt) 로 정규화하여 head 축에 broadcast.
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Dead-row safety: row 전체가 -inf 면 softmax → NaN. 그 row 만 0 으로 풀어 uniform 으로 둠.
            # 어차피 caller 가 해당 row 결과를 액션에 쓰지 않음 (decoder 의 dead_row 처리와 동일 패턴).
            dead_row = (mask < 0).all(dim=-1, keepdim=True)
            safe_mask = mask.masked_fill(dead_row, 0.0)
            mixed_scores = mixed_scores + safe_mask

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        v_total = v.unsqueeze(2) + v_edge
        out = (weights.unsqueeze(-1) * v_total).sum(dim=3)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat


class CCOBlock(nn.Module):
    """
    Conditional Computation block (POCCO, NeurIPS 2025) — 논문 Eq.(2) 충실 구현.
        CCO(h_c) = IN( Σ_j G(h_c)_j · E_j(h_c) + h_c ),  G(h_c) = Softmax(TopK(h_c · W_G))
    gate 와 m 개 FF expert 모두 입력 h_c 를 그대로 사용하고, ID expert E_{m+1}(h_c)=h_c
    (parameter-free identity) 한 개를 더한다. 정규화(IN)는 residual 합 뒤에 마지막에 적용.
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.num_ff_experts = model_params.get('cco_num_ff_experts', 4)
        self.top_k = model_params.get('cco_top_k', 2)
        num_experts = self.num_ff_experts + 1  # +1: parameter-free ID expert

        # m 개 FF expert: 모든 expert 를 stack 해 einsum 으로 한 번에 계산 (각자 독립 파라미터)
        self.W1 = nn.Parameter(torch.empty(self.num_ff_experts, embedding_dim, ff_hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_ff_experts, ff_hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_ff_experts, ff_hidden_dim, embedding_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_ff_experts, embedding_dim))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

        # router W_G ∈ R^{d×(m+1)} (paper Eq.2 / Appendix C.3): 입력은 h_c (embedding_dim).
        self.gate = nn.Linear(embedding_dim, num_experts, bias=False)
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

        # ── 라우팅 헬스 누적 버퍼 (train-mode forward 에서만 갱신) ──
        # expert collapse / 게이트 경직 감시용. DataRecorder.log_moe 가 epoch 마다 pop.
        # persistent=False → state_dict(체크포인트)에 포함되지 않음.
        self.register_buffer('_route_mass', torch.zeros(num_experts), persistent=False)
        self.register_buffer('_route_ent_sum', torch.zeros(()), persistent=False)
        self.register_buffer('_route_count', torch.zeros(()), persistent=False)

    def forward(self, x):
        # x = h_c.shape: (batch, problem, embedding)  — MHA 출력 context vector.
        # gate 와 expert 모두 h_c 를 입력으로 사용 (paper Eq.2). 머신별 context 가 독립 라우팅됨.
        # num_ff_experts 가 작아 sparse gather 대신 모든 expert 를 dense 계산 후 top_k gate 만 가중.
        gate_logits = self.gate(x)                             # (batch, problem, num_experts) = h_c · W_G
        topk_logits, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        # autocast(bf16) 하에서 softmax 는 fp32 로 승격되므로 gate_logits(bf16) 와 dtype 이 어긋남 →
        # scatter dtype 일치를 위해 activation dtype 으로 되돌림.
        topk_gates = F.softmax(topk_logits, dim=-1).to(gate_logits.dtype)
        # 선택된 top_k 슬롯에만 softmax gate, 나머지 expert 엔 0 → Softmax(TopK(·)) 와 동일.
        full_gates = torch.zeros_like(gate_logits).scatter(-1, topk_idx, topk_gates)

        if self.training:
            # 라우팅 통계 누적 (no-grad, sync 없음). model.eval() 인 Phase-1 sampling/eval 은
            # 제외되고, model.train() 인 Phase-2 grad-replay(실제 학습 대상) 만 잡힌다.
            with torch.no_grad():
                g = full_gates.detach().float()                      # (batch, problem, num_experts)
                self._route_mass += g.sum(dim=(0, 1))                # expert 별 게이트 질량 누적
                ent = -(g.clamp_min(1e-12).log() * g).sum(dim=-1)    # (batch, problem) per-token 엔트로피
                self._route_ent_sum += ent.sum()
                self._route_count += g.shape[0] * g.shape[1]

        # FF experts E_1..E_m 를 h_c 에 적용. b1/b2 는 trailing dim 으로 broadcast.
        ff_hidden = F.relu(torch.einsum('bpe,nef->bpnf', x, self.W1) + self.b1)
        ff_all = torch.einsum('bpnf,nfe->bpne', ff_hidden, self.W2) + self.b2
        # ID expert E_{m+1}(h_c) = h_c (parameter-free identity) 를 마지막 슬롯에 붙임.
        expert_out = torch.cat([ff_all, x.unsqueeze(2)], dim=2)  # (batch, problem, num_experts, embedding)

        combined = (expert_out * full_gates.unsqueeze(-1)).sum(dim=2)  # Σ_j G_j E_j(h_c)
        # residual + h_c 뒤에 InstanceNorm — paper Eq.2 의 IN(... + h_c) (post-norm).
        out = combined + x
        return self.norm(out.transpose(1, 2)).transpose(1, 2)

    def pop_routing_stats(self):
        """누적 라우팅 통계 반환 + 리셋. train-mode forward 가 없던 epoch 이면 None.
        load[j] = expert j 로 간 평균 게이트 질량 (Σ_j load = 1), 마지막 슬롯이 ID expert.
        load_max↑ 또는 entropy↓ → 소수 expert 로 쏠림(collapse) 신호."""
        if float(self._route_count) == 0.0:
            return None
        n = self._route_count.clamp_min(1.0)
        load = (self._route_mass / n).tolist()
        entropy = float(self._route_ent_sum / n)
        self._route_mass.zero_()
        self._route_ent_sum.zero_()
        self._route_count.zero_()
        return {'load': load, 'entropy': entropy}


class InstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        # 기존과 동일한 InstanceNorm1d 사용
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, x):
        # shape: (batch, problem, embedding) -> (batch, embedding, problem) -> (batch, problem, embedding)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)
