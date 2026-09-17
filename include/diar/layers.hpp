#pragma once

// M2 Stage 2 layer compositions out of the Stage 1 tensor core (nn.hpp).
// Reference FP32 implementations; inference semantics only (all dropouts are
// eval-mode identities and never appear).
//
// Upstream pins (NeMo-Speech.cpp a5b6953 C++ port + NeMo torch, NOT assumptions):
//   transformer block (post-LN, no pos emb, ReLU FF):
//     <- transformer_encoder.cpp TransformerBlock::build_graph (order: MHA ->
//        +residual -> LN1 -> dense_in -> relu -> dense_out -> +residual -> LN2;
//        scale 1/sqrt(d_k) single-sided == NeMo q/k d_k^0.25 pair)
//     <- NeMo TransformerEncoderBlock.forward_postln + MultiHeadAttention +
//        PositionWiseFF(hidden_act="relu"); LN eps 1e-5 explicit in __init__.
//   head (relu -> lin -> relu -> lin -> sigmoid):
//     <- sortformer_model.cpp SortformerGraph::build_graph §4
//     <- NeMo SortformerModules.forward_speaker_sigmoids/logits.
//   ConformerFF (macaron half-step body, WITHOUT the 0.5 scale):
//     <- fastconformer.cpp ConformerFF::build_graph (linear1 -> silu ->
//        linear2; BF16 casts are backend artifacts, skipped in FP32 ref)
//     <- NeMo ConformerFeedForward.forward (Linear -> Swish==SiLU -> Linear).
//     The 0.5 residual scale lives at the ConformerLayer call site
//     (ggml_scale(ff, 0.5f) / NeMo fc_factor), NOT inside this function.
//
// Layout contract (same as nn.hpp): activations row-major frame-major [T, C];
// Linear weights PyTorch [out, in]. Head split is CONTIGUOUS: head h owns
// cols [h*d_k, (h+1)*d_k) — matches ggml reshape (d_k,n_head,T) of (H,T) and
// torch view(B,T,H,D). (Convention truth vs NeMo weights is owned by the
// Stage 2 Kaggle teacher-forced gates; local oracles pin wiring only.)
//
// All functions allocate their own temporaries (reference speed is
// irrelevant). T <= 0 is a no-op. Input buffers are never modified.

namespace diar {

// Transformer block weights, all PyTorch layout. H=hidden (192), I=inner (768).
struct TransformerBlockWeights {
    const float* q_w = nullptr;  // [H,H] + b[H]
    const float* q_b = nullptr;
    const float* k_w = nullptr;
    const float* k_b = nullptr;
    const float* v_w = nullptr;
    const float* v_b = nullptr;
    const float* o_w = nullptr;  // out_projection [H,H] + b[H]
    const float* o_b = nullptr;
    const float* ln1_g = nullptr;  // [H] gamma/beta
    const float* ln1_b = nullptr;
    const float* ln2_g = nullptr;
    const float* ln2_b = nullptr;
    const float* f1_w = nullptr;  // dense_in [I,H] + b[I]
    const float* f1_b = nullptr;
    const float* f2_w = nullptr;  // dense_out [H,I] + b[H]
    const float* f2_b = nullptr;
};

// x [T,H] -> y [T,H]. Plain (non-rel-pos, unmasked) multi-head attention.
void transformer_block_forward(const float* x, const TransformerBlockWeights& w, float* y, int t,
    int h = 192, int inner = 768, int n_heads = 8);

// x [T,H] -> y [T,S] speaker probabilities (sigmoid).
void diar_head_forward(const float* x, const float* hidden_w, const float* hidden_b,
    const float* spks_w, const float* spks_b, float* y, int t, int h = 192, int n_spk = 4);

// ConformerFF body: Linear(d->dff) -> SiLU -> Linear(dff->d). No 0.5 scale.
void conformer_ff_forward(const float* x, const float* w1, const float* b1, const float* w2,
    const float* b2, float* y, int t, int d_model, int d_ff);

}  // namespace diar
