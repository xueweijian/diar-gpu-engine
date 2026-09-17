// M2 Stage 2 ConformerConv — see include/diar/conv.hpp for pins.
#include "diar/conv.hpp"

#include <cstddef>
#include <vector>

#include "diar/nn.hpp"

namespace diar {

void conformer_conv_forward(const float* x, const ConformerConvWeights& w, float* y, int t,
    int d_model, int kernel, ConvNormKind norm, float eps) {
    if (t <= 0) return;
    const std::size_t T = static_cast<std::size_t>(t);
    const std::size_t D = static_cast<std::size_t>(d_model);

    // pointwise_conv1 as Linear(D -> 2D), then GLU on channel halves.
    std::vector<float> e(T * 2 * D);
    nn::linear_forward(x, w.pw1_w, w.pw1_b, e.data(), T, D, 2 * D);
    std::vector<float> g(T * D);
    nn::glu_forward(e.data(), g.data(), T, D);

    // Depthwise k, stride 1, symmetric zero pad (torch CausalConv1D with
    // full symmetric context == plain symmetric pad in offline mode).
    const int pad = (kernel - 1) / 2;
    std::vector<float> d(T * D);
    nn::conv1d_depthwise_forward(
        g.data(), w.dw_w, w.dw_b, d.data(), t, d_model, kernel, /*stride=*/1, pad);

    // Norm over channels (value-identical pre/post transpose), then SiLU,
    // then pointwise_conv2 as Linear(D -> D).
    std::vector<float> n(T * D);
    if (norm == ConvNormKind::BatchNorm) {
        nn::batchnorm1d_infer_forward(
            d.data(), w.bn_w, w.bn_b, w.bn_mean, w.bn_var, n.data(), T, D, eps);
    } else {
        nn::layernorm_forward(d.data(), w.bn_w, w.bn_b, n.data(), T, D, eps);
    }
    std::vector<float> a(T * D);
    nn::silu_forward(n.data(), a.data(), T * D);
    nn::linear_forward(a.data(), w.pw2_w, w.pw2_b, y, T, D, D);
}

}  // namespace diar
