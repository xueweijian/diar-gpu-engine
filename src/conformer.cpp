// M2 Stage 2 full Conformer layer — see include/diar/conformer.hpp.
#include "diar/conformer.hpp"

#include <cstddef>
#include <vector>

#include "diar/conv.hpp"
#include "diar/layers.hpp"
#include "diar/mha.hpp"
#include "diar/nn.hpp"

namespace diar {

void conformer_layer_forward(const float* x, const float* pos_emb,
    const ConformerLayerWeights& w, float* y, int t, int d_model, int d_ff, int n_heads,
    int conv_kernel) {
    if (t <= 0) return;
    const std::size_t T = static_cast<std::size_t>(t);
    const std::size_t D = static_cast<std::size_t>(d_model);

    std::vector<float> residual(T * D);
    for (std::size_t i = 0; i < T * D; ++i) residual[i] = x[i];

    // FF1 (macaron half-step).
    std::vector<float> n1(T * D), f1(T * D);
    nn::layernorm_forward(residual.data(), w.n_ff1_g, w.n_ff1_b, n1.data(), T, D);
    conformer_ff_forward(
        n1.data(), w.ff1_w1, w.ff1_b1, w.ff1_w2, w.ff1_b2, f1.data(), t, d_model, d_ff);
    for (std::size_t i = 0; i < T * D; ++i) residual[i] += 0.5F * f1[i];

    // rel-pos self-attn.
    std::vector<float> na(T * D), attn(T * D);
    nn::layernorm_forward(residual.data(), w.n_sa_g, w.n_sa_b, na.data(), T, D);
    relpos_mha_forward(na.data(), pos_emb, w.attn, attn.data(), t, d_model, n_heads);
    for (std::size_t i = 0; i < T * D; ++i) residual[i] += attn[i];

    // Conv module.
    std::vector<float> nc(T * D), cv(T * D);
    nn::layernorm_forward(residual.data(), w.n_conv_g, w.n_conv_b, nc.data(), T, D);
    conformer_conv_forward(nc.data(), w.conv, cv.data(), t, d_model, conv_kernel);
    for (std::size_t i = 0; i < T * D; ++i) residual[i] += cv[i];

    // FF2 (macaron half-step).
    std::vector<float> n2(T * D), f2(T * D);
    nn::layernorm_forward(residual.data(), w.n_ff2_g, w.n_ff2_b, n2.data(), T, D);
    conformer_ff_forward(
        n2.data(), w.ff2_w1, w.ff2_b1, w.ff2_w2, w.ff2_b2, f2.data(), t, d_model, d_ff);
    for (std::size_t i = 0; i < T * D; ++i) residual[i] += 0.5F * f2[i];

    // Final norm.
    nn::layernorm_forward(residual.data(), w.n_out_g, w.n_out_b, y, T, D);
}

}  // namespace diar
