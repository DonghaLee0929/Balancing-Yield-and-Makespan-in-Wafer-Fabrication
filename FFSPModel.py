
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

from FFSPModel_SUB import AddAndInstanceNormalization, FeedForward, MixedScore_MultiHeadAttention, CCOBlock


class FFSPModel(nn.Module):

    def __init__(self, row_feature_dim, col_feature_dim, edge_feature_dim=1, **model_params):
        super().__init__()
        # MixedScore_MultiHeadAttention 의 mix1 입력 채널을 결정하기 위해 model_params 로 전파.
        model_params['edge_feature_dim'] = int(edge_feature_dim)
        self.model_params = model_params

        embedding_dim = self.model_params['embedding_dim']
        self.Wr = nn.Linear(row_feature_dim, embedding_dim, bias=True)
        self.Wc = nn.Linear(col_feature_dim, embedding_dim, bias=True)
        # λ (preference scalar) → embedding_dim. POCCO eq.(8) FiLM 신호로 encoder/decoder 양쪽 사용.
        self.W_lam = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.encoder = FFSP_Encoder(**model_params)
        self.decoder = FFSP_Decoder(**model_params)

        self.encoded_col = None
        # shape: (batch x pomo, machine_cnt, embedding)
        self.encoded_row = None
        # shape: (batch x pomo, job_cnt, embedding)

    def forward(self, state, sample=True):
        # state.BATCH_IDX.shape    : (batch, pomo)
        # state.row_feature.shape  : (BP, job_cnt, row_feat_dim)
        # state.col_feature.shape  : (BP, machine_cnt, col_feat_dim)
        # state.edge_feature.shape : (BP, job_cnt, machine_cnt, edge_feat_dim)
        # state.edge_mask.shape    : (BP, machine_cnt, job_cnt)
        # state.finished.shape     : (BP,) bool
        # sample=False → Gumbel-Max/argmax 스킵, edge_selected=None 반환 (SIL replay 용).
        # Returns:
        #   edge_selected : (BP,) long — softmax sample 또는 argmax index ∈ [0, machine_cnt * job_cnt). sample=False 면 None.
        #   flat_probs    : (BP, machine_cnt * job_cnt) — joint distribution. caller 가
        #                   gather(action) + finished masking 책임 (SIL replay 포함).
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        BP = batch_size * pomo_size

        lam_emb = self.W_lam(state.lambdas.view(-1, 1))  # (BP, embedding_dim) — initial λ embedding
        row_emb = self.Wr(state.row_feature)
        col_emb = self.Wc(state.col_feature)
        # encoder 가 λ 도 layer 마다 업데이트 (POCCO eq 9, 12) → encoded_lam 반환
        # state.edge_mask 는 (BP, machine_cnt, job_cnt) ninf 스타일 → encoder 의 (J, M) 방향에 맞춰 transpose.
        edge_mask_jm = state.edge_mask.transpose(1, 2)
        self.encoded_row, self.encoded_col, encoded_lam = self.encoder(
            row_emb, col_emb, state.edge_feature, edge_mask=edge_mask_jm, lam_emb=lam_emb)
        # encoded_row.shape: (BP, job_cnt, embedding)
        # encoded_col.shape: (BP, machine_cnt, embedding)
        # encoded_lam.shape: (BP, embedding)  — h^L_λ, 모든 instance feature 가 attend 된 동적 신호

        # decoder: MHA K/V 에 λ 한 슬롯 추가 (POCCO eq.13) + CCO gate 입력으로 encoded λ 전달
        # (paper Fig.1: per-subproblem 라우팅). set_kv 가 lam_emb 도 저장해 forward 에서 사용.
        self.decoder.set_kv(self.encoded_row, lam_emb=encoded_lam)
        all_edge_probs = self.decoder(self.encoded_col, ninf_mask=state.edge_mask)
        # shape: (BP, machine_cnt, job_cnt) — 전 (machine, job) 그리드 joint 분포

        # (machine × job_cnt) 을 단일 edge 차원으로 평탄화
        flat_probs = all_edge_probs.reshape(BP, -1)
        # shape: (BP, edge_cnt)  where edge_cnt = machine_cnt * job_cnt

        # SIL replay 처럼 action 이 미리 정해진 호출은 sample=False 로 Gumbel-Max/argmax 스킵.
        if not sample:
            return None, flat_probs

        if self.training or self.model_params['eval_type'] == 'softmax':
            # [Gumbel-Max Trick]
            # 1. 0인 확률은 -inf에 가깝게 만들고 나머지는 Log 확률로 변환
            logits = torch.log(flat_probs.clamp(min=1e-30))
            
            # 2. Gumbel 노이즈 생성 (flat_probs와 동일한 디바이스/형태)
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-30))
            
            # 3. Logits + Noise의 argmax 추출 (수학적으로 multinomial과 동일)
            edge_selected = (logits + gumbel_noise).argmax(dim=1)
        else:
            edge_selected = flat_probs.argmax(dim=1)

        # edge_selected[bp] ∈ [0, machine_cnt * job_cnt)
        #   machine_idx = edge_selected // job_cnt
        #   job_idx     = edge_selected %  job_cnt
        return edge_selected, flat_probs


########################################
# ENCODER
########################################
class FFSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, cost_mat, edge_mask=None, lam_emb=None):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        # edge_mask.shape: (batch, row_cnt, col_cnt) ninf 스타일 — stage feasibility. None 이면 비활성.
        # lam_emb.shape: (batch, embedding)  — POCCO eq.(8-12). None 이면 비활성.
        # λ 는 layer 사이로 thread 되며 매 layer 의 EncodingBlock MHA 로 업데이트 (eq 9, 12).

        for layer in self.layers:
            row_emb, col_emb, lam_emb = layer(row_emb, col_emb, cost_mat, edge_mask=edge_mask, lam_emb=lam_emb)

        return row_emb, col_emb, lam_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat, edge_mask=None, lam_emb=None):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        # edge_mask.shape: (batch, row_cnt, col_cnt) ninf — cost_mat 과 같은 (row, col) 방향.
        # lam_emb.shape: (batch, embedding) — None 이면 FiLM/λ-pseudo-node 모두 bypass.
        # row/col block 둘 다 자기 관점에서 λ update 를 뱉음 → 평균해 layer-output λ 로 사용
        # (paper 의 단일 노드셋 eq 9 를 bipartite 로 적응한 형태).
        # col block 은 cost_mat 을 transpose 하므로 mask 도 동일 방향으로 transpose.
        col_mask = edge_mask.transpose(1, 2) if edge_mask is not None else None
        row_emb_out, lam_from_row = self.row_encoding_block(row_emb, col_emb, cost_mat, mask=edge_mask, lam_emb=lam_emb)
        col_emb_out, lam_from_col = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2), mask=col_mask, lam_emb=lam_emb)

        lam_out = 0.5 * (lam_from_row + lam_from_col) if lam_emb is not None else None
        return row_emb_out, col_emb_out, lam_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # POCCO eq.(8) FiLM conditioner: γ = 1 + W_γ(λ),  β = W_β(λ).
        # small-normal init (std=0.01): init 시점부터 λ 가 미세하게 출력에 흘러가
        # gradient self-amplification 경로가 깨어있도록. 
        self.W_gamma = nn.Linear(embedding_dim, embedding_dim)
        self.W_beta  = nn.Linear(embedding_dim, embedding_dim)
        nn.init.normal_(self.W_gamma.weight, std=0.01); nn.init.zeros_(self.W_gamma.bias)
        nn.init.normal_(self.W_beta.weight,  std=0.01); nn.init.zeros_(self.W_beta.bias)

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params) # previous
        # self.mixed_score_MHA = Mixed_MultiHeadCrossAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat, mask=None, lam_emb=None):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        # mask.shape: (batch, row_cnt, col_cnt) ninf — stage feasibility. None 이면 비활성.
        # lam_emb.shape: (batch, embedding)  — None 이면 FiLM/λ-pseudo-node 모두 bypass.
        head_num = self.model_params['head_num']

        # POCCO eq.(8-12) 충실 구현:
        #   - nodes 는 FiLM-conditioned 로 Q/K/V 사용,
        #   - λ pseudo-node 는 uncond 으로 Q/K/V 양쪽에 한 자리 추가,
        #   - residual base 는 원본 (nodes: h^{l-1}_i, λ: h^{l-1}_λ).
        # 단일 MHA 호출로 (nodes + λ) 동시 업데이트 → 끝에서 분리해 반환.
        if lam_emb is not None:
            gamma = 1.0 + self.W_gamma(lam_emb).unsqueeze(1)   # (batch, 1, embedding)
            beta  = self.W_beta(lam_emb).unsqueeze(1)
            row_cond = gamma * row_emb + beta
            col_cond = gamma * col_emb + beta
            lam_token = lam_emb.unsqueeze(1)                    # (batch, 1, embedding)

            q_in   = torch.cat([row_cond, lam_token], dim=1)    # (batch, row_cnt+1, embedding)
            kv_in  = torch.cat([col_cond, lam_token], dim=1)    # (batch, col_cnt+1, embedding)
            # cost_mat: 3D (B,J,M) 또는 4D (B,J,M,edge_feat). 4D 로 정규화해서 J/M 만 우/하단 1 씩 zero pad.
            cost_4d = cost_mat if cost_mat.dim() == 4 else cost_mat.unsqueeze(-1)
            cost_in = F.pad(cost_4d, (0, 0, 0, 1, 0, 1), value=0.0)  # (B, J+1, M+1, edge_feat)
            # mask 도 동일하게 우/하단 1 씩 0 (lam slot 은 항상 feasible) 으로 패딩.
            # mask_in = F.pad(mask, (0, 1, 0, 1), value=0.0) if mask is not None else None
            res_in  = torch.cat([row_emb, lam_token], dim=1)    # 원본 (uncond) residual base
        else:
            q_in, kv_in, cost_in = row_emb, col_emb, cost_mat
            # mask_in = mask
            res_in = row_emb

        q = reshape_by_heads(self.Wq(q_in), head_num=head_num)
        # q shape: (batch, head_num, row_cnt(+1), qkv_dim)
        k = reshape_by_heads(self.Wk(kv_in), head_num=head_num)
        v = reshape_by_heads(self.Wv(kv_in), head_num=head_num)
        # kv shape: (batch, head_num, col_cnt(+1), qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_in) # previous
        # out_concat = self.mixed_score_MHA(q, k, v, cost_in, mask=mask_in) # previous
        # shape: (batch, row_cnt(+1), head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt(+1), embedding)

        out1 = self.add_n_normalization_1(res_in, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        if lam_emb is not None:
            return out3[:, :-1, :], out3[:, -1, :]
        return out3, None
        # shape: (batch, row_cnt, embedding), (batch, embedding) or None


########################################
# Decoder
########################################

class FFSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        # POCCO CCO block — λ_emb 으로 게이팅하는 per-subproblem expert routing
        self.cco_block = CCOBlock(**self.model_params)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention

    def set_kv(self, encoded_jobs, lam_emb=None):
        # encoded_jobs.shape: (batch, job_cnt, embedding)
        # lam_emb.shape: (batch, embedding)  — POCCO eq.(13): MHA K/V 에 λ 한 슬롯 추가.
        # compatibility / single_head_key 는 nodes 만 (action 공간에 'select λ' 없음).
        head_num = self.model_params['head_num']

        kv_in = (torch.cat([encoded_jobs, lam_emb.unsqueeze(1)], dim=1)
                 if lam_emb is not None else encoded_jobs)
        self.k = reshape_by_heads(self.Wk(kv_in), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(kv_in), head_num=head_num)
        # shape: (batch, head_num, job_cnt(+1), qkv_dim)
        self.single_head_key = encoded_jobs.transpose(1, 2)
        # shape: (batch, embedding, job_cnt)  — λ 제외
        self.lam_emb = lam_emb  # CCO gating 입력으로 forward 에서 사용

    def forward(self, encoded_machines, ninf_mask):
        # encoded_machines.shape: (batch, machine_cnt, embedding)  ← 모든 기계를 쿼리로
        # ninf_mask.shape: (batch, machine_cnt, job_cnt)
        # CCO 는 mh_atten_out (= h_c) 자체로 per-(batch, machine) 라우팅 — POCCO 원형.
        # encoder 가 이미 FiLM 으로 λ-aware 라 late concat 없음.

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        q = reshape_by_heads(self.Wq_3(encoded_machines), head_num=head_num)
        # shape: (batch, head_num, machine_cnt, qkv_dim)

        # HFSP 마스크 의미가 (machine, job) feasibility 라 row 전체가 -inf 인 머신이 생긴다
        # (해당 머신의 stage 에 active job 이 없을 때). 그대로 softmax 면 NaN.
        # → dead-row 만 zero 로 풀어 NaN 회피. 그 머신은 아래 edge softmax 에서 어차피
        #   모든 job 이 -inf → 확률 0 이라 attention 결과 자체는 액션에 영향 없음.
        #   살아있는 머신은 정상적으로 자기 stage feasible job 만 attend → logit 날카로움 유지.
        # λ pseudo-node 슬롯이 K/V 끝에 있으면 항상 valid (0) 로 한 칸 패딩.
        dead_row = (ninf_mask < 0).all(dim=-1, keepdim=True)
        safe_mask = ninf_mask.masked_fill(dead_row, 0.0)
        if self.k.size(2) != ninf_mask.size(-1):
            safe_mask = F.pad(safe_mask, (0, 1), value=0.0)
        out_concat = self._multi_head_attention_for_decoder(q, self.k, self.v, rank3_ninf_mask=safe_mask)
        # shape: (batch, machine_cnt, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, machine_cnt, embedding)

        # POCCO CCO routing — paper Fig.1 / eq.2 의 per-subproblem 라우팅.
        # gate_input = encoded λ 벡터 → 같은 (batch, λ) 의 모든 머신 토큰이 동일 expert 조합으로
        # 처리됨 → λ 다를수록 다른 computation path → policy specialization.
        mh_atten_out = self.cco_block(mh_atten_out, gate_input=self.lam_emb)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, machine_cnt, job_cnt)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        # shape: (batch, machine_cnt, job_cnt)

        # (machine, job) 전체 그리드에 대해 joint softmax
        # → 전 (machine, job) edge 위에서 확률이 1로 합쳐지는 단일 분포
        batch_size, machine_cnt, job_cnt = score_masked.shape
        probs = F.softmax(score_masked.reshape(batch_size, machine_cnt * job_cnt), dim=1)
        probs = probs.reshape(batch_size, machine_cnt, job_cnt)
        # shape: (batch, machine_cnt, job_cnt)

        return probs

    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        # q shape: (batch, head_num, n, qkv_dim)   : n can be either 1 or PROBLEM_SIZE
        # k,v shape: (batch, head_num, job_cnt, qkv_dim)
        # rank2_ninf_mask.shape: (batch, job_cnt)
        # rank3_ninf_mask.shape: (batch, n, job_cnt)

        batch_size = q.size(0)
        n = q.size(2)
        job_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, n, job_cnt)

        score_scaled = score / sqrt_qkv_dim

        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_size, head_num, n, job_cnt)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, n, job_cnt)

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, job_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, n, head_num * qkv_dim)
        # shape: (batch, n, head_num*qkv_dim)

        return out_concat


########################################
# NN SUB FUNCTIONS
########################################


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed
