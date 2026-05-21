
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
    Conditional Computation block (POCCO, NeurIPS 2025).
    각 입력 토큰을 sparse Top-k gate 를 통해 {FF experts, ID expert} 로 라우팅한 뒤
    residual + InstanceNorm 을 적용한다.
        CCO(h) = IN( sum_j G(h)_j · E_j(h) + h ),  G = Softmax(TopK(h · W_G))
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.num_ff_experts = model_params.get('cco_num_ff_experts', 4)
        self.top_k = model_params.get('cco_top_k', 2)
        num_experts = self.num_ff_experts + 1  # +1: parameter-free ID expert

        # FF experts: 모든 expert 를 stack 해 einsum 으로 한 번에 계산
        self.W1 = nn.Parameter(torch.empty(self.num_ff_experts, embedding_dim, ff_hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_ff_experts, ff_hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_ff_experts, ff_hidden_dim, embedding_dim))
        self.b2 = nn.Parameter(torch.zeros(self.num_ff_experts, embedding_dim))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

        self.gate = nn.Linear(embedding_dim, num_experts, bias=False)
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, x, gate_input):
        # x.shape:          (batch, problem, embedding)
        # gate_input.shape: (batch, embedding) — per-subproblem 라우팅. 한 배치 요소 안의
        #                   모든 token 이 동일 expert 조합을 공유 → sparse dispatch 로
        #                   선택된 top_k expert weight 만 gather 해서 FF 연산.
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)

        topk_logits, topk_idx = torch.topk(self.gate(gate_input), self.top_k, dim=-1)
        topk_gates = F.softmax(topk_logits, dim=-1)
        # shape: (batch, top_k)

        # ID expert (index = num_ff_experts) 는 파라미터 없음 → clamp 후 해당 슬롯의
        # FF 출력을 x_norm 으로 덮어쓴다.
        is_id = (topk_idx == self.num_ff_experts)
        ff_idx = topk_idx.clamp(max=self.num_ff_experts - 1)

        W1_sel = self.W1[ff_idx]  # (batch, top_k, embedding, ff_hidden)
        b1_sel = self.b1[ff_idx]  # (batch, top_k, ff_hidden)
        W2_sel = self.W2[ff_idx]  # (batch, top_k, ff_hidden, embedding)
        b2_sel = self.b2[ff_idx]  # (batch, top_k, embedding)

        ff_hidden = F.relu(torch.einsum('bpe,bkef->bpkf', x_norm, W1_sel) + b1_sel.unsqueeze(1))
        ff_sel = torch.einsum('bpkf,bkfe->bpke', ff_hidden, W2_sel) + b2_sel.unsqueeze(1)
        # shape: (batch, problem, top_k, embedding)

        ff_sel = torch.where(is_id[:, None, :, None], x_norm.unsqueeze(2), ff_sel)

        combined = (ff_sel * topk_gates[:, None, :, None]).sum(dim=2)
        # shape: (batch, problem, embedding)

        return combined + x
    

class InstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        # 기존과 동일한 InstanceNorm1d 사용
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, x):
        # shape: (batch, problem, embedding) -> (batch, embedding, problem) -> (batch, problem, embedding)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)
