// M2 Stage 2 layer compositions — see include/diar/layers.hpp for pins.
#include "diar/layers.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "diar/nn.hpp"

namespace diar {

void transformer_block_forward(const float* x, const TransformerBlockWeights& w, float* y, int t,
    int h, int inner, int n_heads) {
    if (t <= 0) return;
    const std::size_t T = static_cast<std::size_t>(t);
    const std::size_t H = static_cast<std::size_t>(h);
    const std::size_t I = static_cast<std::size_t>(inner);
    const std::size_t NH = static_cast<std::size_t>(n_heads);
    const std::size_t DK = H / NH;

    std::vector<float> q(T * H), k(T * H), v(T * H);
    nn::linear_forward(x, w.q_w, w.q_b, q.data(), T, H, H);
    nn::linear_forward(x, w.k_w, w.k_b, k.data(), T, H, H);
    nn::linear_forward(x, w.v_w, w.v_b, v.data(), T, H, H);

    const float scale = 1.0F / std::sqrt(static_cast<float>(DK));
    std::vector<float> attn(T * H, 0.0F);
    std::vector<float> scores(T * T), probs(T * T);
    std::vector<float> ctx(T * DK);
    for (std::size_t hd = 0; hd < NH; ++hd) {
        for (std::size_t i = 0; i < T; ++i) {
            for (std::size_t j = 0; j < T; ++j) {
                float acc = 0.0F;
                for (std::size_t e = 0; e < DK; ++e)
                    acc += q[i * H + hd * DK + e] * k[j * H + hd * DK + e];
                scores[i * T + j] = acc * scale;
            }
        }
        nn::softmax_last_dim(scores.data(), probs.data(), T, T);
        for (std::size_t i = 0; i < T; ++i) {
            for (std::size_t e = 0; e < DK; ++e) {
                float acc = 0.0F;
                for (std::size_t j = 0; j < T; ++j) acc += probs[i * T + j] * v[j * H + hd * DK + e];
                ctx[i * DK + e] = acc;
            }
        }
        for (std::size_t i = 0; i < T; ++i)
            for (std::size_t e = 0; e < DK; ++e) attn[i * H + hd * DK + e] = ctx[i * DK + e];
    }

    std::vector<float> attn_out(T * H);
    nn::linear_forward(attn.data(), w.o_w, w.o_b, attn_out.data(), T, H, H);

    std::vector<float> h1(T * H);
    for (std::size_t i = 0; i < T * H; ++i) h1[i] = attn_out[i] + x[i];
    std::vector<float> h1_ln(T * H);
    nn::layernorm_forward(h1.data(), w.ln1_g, w.ln1_b, h1_ln.data(), T, H);

    std::vector<float> ff_mid(T * I);
    nn::linear_forward(h1_ln.data(), w.f1_w, w.f1_b, ff_mid.data(), T, H, I);
    nn::relu_inplace(ff_mid.data(), T * I);
    std::vector<float> ff_out(T * H);
    nn::linear_forward(ff_mid.data(), w.f2_w, w.f2_b, ff_out.data(), T, I, H);

    std::vector<float> h2(T * H);
    for (std::size_t i = 0; i < T * H; ++i) h2[i] = ff_out[i] + h1_ln[i];
    nn::layernorm_forward(h2.data(), w.ln2_g, w.ln2_b, y, T, H);
}

void diar_head_forward(const float* x, const float* hidden_w, const float* hidden_b,
    const float* spks_w, const float* spks_b, float* y, int t, int h, int n_spk) {
    if (t <= 0) return;
    const std::size_t T = static_cast<std::size_t>(t);
    const std::size_t H = static_cast<std::size_t>(h);
    const std::size_t S = static_cast<std::size_t>(n_spk);
    std::vector<float> a(T * H);
    for (std::size_t i = 0; i < T * H; ++i) a[i] = x[i];
    nn::relu_inplace(a.data(), T * H);
    std::vector<float> h1(T * H);
    nn::linear_forward(a.data(), hidden_w, hidden_b, h1.data(), T, H, H);
    nn::relu_inplace(h1.data(), T * H);
    std::vector<float> logits(T * S);
    nn::linear_forward(h1.data(), spks_w, spks_b, logits.data(), T, H, S);
    nn::sigmoid_forward(logits.data(), y, T * S);
}

void conformer_ff_forward(const float* x, const float* w1, const float* b1, const float* w2,
    const float* b2, float* y, int t, int d_model, int d_ff) {
    if (t <= 0) return;
    const std::size_t T = static_cast<std::size_t>(t);
    const std::size_t D = static_cast<std::size_t>(d_model);
    const std::size_t F = static_cast<std::size_t>(d_ff);
    std::vector<float> mid(T * F);
    nn::linear_forward(x, w1, b1, mid.data(), T, D, F);
    std::vector<float> act(T * F);
    nn::silu_forward(mid.data(), act.data(), T * F);
    nn::linear_forward(act.data(), w2, b2, y, T, F, D);
}

}  // namespace diar
