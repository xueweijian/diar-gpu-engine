#pragma once

// M2 Stage 2 full Conformer layer (non-cached / offline path only).
// Reference FP32 implementation; inference semantics.
//
// Upstream pins (NOT assumptions):
//   NeMo torch ConformerLayer.forward, non-cached (conformer_modules.py):
//     residual=x; x=norm_ff1(x); x=ff1(x); residual+=dropout(x)*fc_factor(0.5)
//     x=norm_self_att(residual); x=self_attn(q=k=v=x, pos_emb)   [rel_pos]
//     residual+=dropout(x); x=norm_conv(residual); x=conv(x); residual+=dropout(x)
//     x=norm_ff2(residual); x=ff2(x); residual+=dropout(x)*0.5; x=norm_out(residual)
//   C++ ggml port ConformerLayer::build_graph non-cached (fastconformer.cpp):
//     same order incl. ggml_scale(ff,0.5) at both FF sites, norm_out final.
//     pos_emb input + offline mask threading skipped here (offline
//     full-attention teacher gates pass full-length inputs, no padding).
//   fc_factor=0.5: NeMo ConformerLayer.__init__ (self.fc_factor = 0.5).
//   All five norms are LayerNorm(d_model) [torch nn default eps 1e-5 —
//     matches kLayerNormEps]; conv norm inside conv module is BatchNorm
//     (see conv.hpp diar config).
//
// Layout contract: [T,C] frame-major everywhere; weights PyTorch [out,in].
// pos_emb [2T-1,C]. T <= 0 no-op. Inputs never modified.

#include "diar/conv.hpp"
#include "diar/mha.hpp"

namespace diar {

struct ConformerLayerWeights {
    // norms: 5x (gamma[C], beta[C]) in order ff1/self_att/conv/ff2/out.
    const float* n_ff1_g = nullptr;
    const float* n_ff1_b = nullptr;
    const float* n_sa_g = nullptr;
    const float* n_sa_b = nullptr;
    const float* n_conv_g = nullptr;
    const float* n_conv_b = nullptr;
    const float* n_ff2_g = nullptr;
    const float* n_ff2_b = nullptr;
    const float* n_out_g = nullptr;
    const float* n_out_b = nullptr;
    // ff1 / ff2 ConformerFF bodies.
    const float* ff1_w1 = nullptr;  // [F,D]+b[F]
    const float* ff1_b1 = nullptr;
    const float* ff1_w2 = nullptr;  // [D,F]+b[D]
    const float* ff1_b2 = nullptr;
    const float* ff2_w1 = nullptr;
    const float* ff2_b1 = nullptr;
    const float* ff2_w2 = nullptr;
    const float* ff2_b2 = nullptr;
    // rel-pos self-attn.
    RelPosMhaWeights attn{};
    // conv module.
    ConformerConvWeights conv{};
};

// x [T,D] (+ pos_emb [2T-1,D]) -> y [T,D].
void conformer_layer_forward(const float* x, const float* pos_emb,
    const ConformerLayerWeights& w, float* y, int t, int d_model, int d_ff, int n_heads,
    int conv_kernel = 9);

}  // namespace diar
