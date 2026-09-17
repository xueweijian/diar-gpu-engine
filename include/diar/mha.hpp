#pragma once

// M2 Stage 2 relative-position MHA (conformer self-attn, non-cached /
// offline path only). Reference FP32 implementation; inference semantics
// (dropouts are eval identities).
//
// Upstream pins (NOT assumptions), in priority order:
//   1. NeMo torch RelPositionMultiHeadAttention.forward (non-SDPA branch) —
//      THE math authority for Stage 2, because the .npz teacher values come
//      from NeMo fp32 (nemo/collections/asr/parts/submodules/
//      multi_head_attention.py @ main):
//        q,k,v = forward_qkv (separate linears, then view(B,T,H,Dk)
//          transpose(1,2) -> (B,H,T,Dk))
//        p = linear_pos(pos_emb, bias=False).view(Bp,-1,H,Dk).transpose(1,2)
//        q_u = (q + pos_bias_u).transpose(1,2)   # (B,H,T,Dk)
//        q_v = (q + pos_bias_v).transpose(1,2)
//        matrix_ac = q_u @ k^T                   # (B,H,Tq,Tkv)
//        matrix_bd = q_v @ p^T -> rel_shift -> truncate to [:,:,:,:Tk]
//        scores = (ac + bd) / s_d_k, s_d_k = sqrt(d_k)  (plain sqrt, the
//          transformer_modules.py d_k^0.25 pair is the OTHER MHA family)
//        softmax(-1) over keys, context = probs @ v, merge heads (B,T,H*Dk),
//        linear_out. mask=None (offline full-attention); cache=None.
//   2. C++ ggml port rel_pos_attention.cpp (unfused CPU path) — same math in
//      ggml layout, incl. the portable pad+view rel-shift; qu = qh+bu,
//      qv = qh+bv with bu/bv = reshape(pos_bias_{u,v}, d_k,1,n_head,1). The
//      Stage 1 nn::rel_shift_forward contract S[k,j]=BD[k-j+T-1,j] was proven
//      against this chain.
//   3. NeMo torch rel_shift: pad(x,(1,0)) on last dim -> view(B,H,T2+1,T1) ->
//      drop row 0 -> view(B,H,T1,T2) with T2 = 2*T1-1 (self-attn square
//      case). B=1 here.
//
// Layout contract: activations row-major frame-major [T,C]; ALL Linear
// weights PyTorch [out,in] (QKV separate + pos/proj/output — the fused
// [n_feat,3*n_feat] stacking is a ggml-runtime load detail, invisible to
// this reference). pos_bias_u/v are [H, Dk] row-major (NeMo (h,d_k)).
// pos_emb is [P, C] with P = 2*T-1 (NeMo RelPositionalEncoding.forward
// window for cache_len=0); p_len is asserted.
// Heads split CONTIGUOUS (head h = cols [h*Dk,(h+1)*Dk)) — consistent with
// layers.hpp; the convention-vs-NeMo-weights truth is owned by the Kaggle
// teacher-forced gates, local oracles pin wiring only.
//
// B=1, unmasked, non-cached. T <= 0 is a no-op. Inputs never modified.

namespace diar {

// All weights PyTorch [out,in]; biases row vectors (pos_lin has NO bias).
struct RelPosMhaWeights {
    const float* q_w = nullptr;    // [C,C] + b[C]
    const float* q_b = nullptr;
    const float* k_w = nullptr;    // [C,C] + b[C]
    const float* k_b = nullptr;
    const float* v_w = nullptr;    // [C,C] + b[C]
    const float* v_b = nullptr;
    const float* pos_w = nullptr;  // linear_pos [C,C], no bias
    const float* bu = nullptr;     // pos_bias_u [H,Dk]
    const float* bv = nullptr;     // pos_bias_v [H,Dk]
    const float* out_w = nullptr;  // linear_out [C,C] + b[C]
    const float* out_b = nullptr;
};

// x [T,C] (+ pos_emb [2T-1,C]) -> y [T,C].
void relpos_mha_forward(const float* x, const float* pos_emb, const RelPosMhaWeights& w, float* y,
    int t, int c, int n_heads);

}  // namespace diar
