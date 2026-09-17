// M2 Stage 2 rel-pos MHA — see include/diar/mha.hpp for pins.
#include "diar/mha.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "diar/nn.hpp"

namespace diar {

void relpos_mha_forward(const float* x, const float* pos_emb, const RelPosMhaWeights& w, float* y,
    int t, int c, int n_heads) {
    if (t <= 0) return;
    const std::size_t T = static_cast<std::size_t>(t);
    const std::size_t C = static_cast<std::size_t>(c);
    const std::size_t NH = static_cast<std::size_t>(n_heads);
    const std::size_t DK = C / NH;
    const std::size_t P = 2 * T - 1;

    std::vector<float> q(T * C), k(T * C), v(T * C), p(P * C);
    nn::linear_forward(x, w.q_w, w.q_b, q.data(), T, C, C);
    nn::linear_forward(x, w.k_w, w.k_b, k.data(), T, C, C);
    nn::linear_forward(x, w.v_w, w.v_b, v.data(), T, C, C);
    nn::linear_forward(pos_emb, w.pos_w, nullptr, p.data(), P, C, C);

    const float s_d_k = std::sqrt(static_cast<float>(DK));
    // scores[i][j], i=query, j=key.
    std::vector<float> scores(T * T), probs(T * T);
    std::vector<float> bd(P * T);  // q_v @ p^T, pre-shift [P rows, T cols]
    std::vector<float> ctx(T * DK);
    std::vector<float> merged(T * C);
    // Orientation: nn::rel_shift_forward stores ggml key-major [Tkv,Tq]
    // (S_ggml[k,j] = BD[k-j+T-1,j], Stage 1). Torch needs query-major
    // S[q,k] = BD[k-q+T-1,q], the TRANSPOSE — hence shifted[j*T+i] below
    // (identity-weight trace: ggml [[2,5],[0,6]] vs torch [[2,0],[5,6]]).
    for (std::size_t hd = 0; hd < NH; ++hd) {
        for (std::size_t i = 0; i < T; ++i) {
            for (std::size_t j = 0; j < T; ++j) {
                float acc = 0.0F;
                for (std::size_t e = 0; e < DK; ++e)
                    acc += (q[i * C + hd * DK + e] + w.bu[hd * DK + e]) *
                           k[j * C + hd * DK + e];
                scores[i * T + j] = acc;
            }
        }
        for (std::size_t r = 0; r < P; ++r) {
            for (std::size_t i = 0; i < T; ++i) {
                float acc = 0.0F;
                for (std::size_t e = 0; e < DK; ++e)
                    acc += (q[i * C + hd * DK + e] + w.bv[hd * DK + e]) * p[r * C + hd * DK + e];
                bd[r * T + i] = acc;
            }
        }
        std::vector<float> shifted(T * T);
        nn::rel_shift_forward(bd.data(), shifted.data(), T);
        for (std::size_t i = 0; i < T; ++i)
            for (std::size_t j = 0; j < T; ++j)
                scores[i * T + j] = (scores[i * T + j] + shifted[j * T + i]) / s_d_k;
        nn::softmax_last_dim(scores.data(), probs.data(), T, T);
        for (std::size_t i = 0; i < T; ++i) {
            for (std::size_t e = 0; e < DK; ++e) {
                float acc = 0.0F;
                for (std::size_t j = 0; j < T; ++j) acc += probs[i * T + j] * v[j * C + hd * DK + e];
                ctx[i * DK + e] = acc;
            }
        }
        for (std::size_t i = 0; i < T; ++i)
            for (std::size_t e = 0; e < DK; ++e) merged[i * C + hd * DK + e] = ctx[i * DK + e];
    }
    nn::linear_forward(merged.data(), w.out_w, w.out_b, y, T, C, C);
}

}  // namespace diar
