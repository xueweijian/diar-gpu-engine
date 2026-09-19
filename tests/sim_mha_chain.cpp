// Local sim: mirror the CUDA MHA chain's buffer orientations & indexing in
// plain C++ and compare against diar::relpos_mha_forward (the pinned ref).
// Purpose: catch indexing/orientation bugs BEFORE burning a Kaggle cycle.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "diar/mha.hpp"
#include "diar/nn.hpp"

using std::vector;

static double max_abs(const vector<float>& a, const vector<float>& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

// What the CUDA chain computes, with buffers laid out exactly as cuBLAS
// writes them (orientation claims from the layout derivation; this sim is
// the executable version of those claims).
static void sim_mha(const float* x, const float* pos, int t, int c, int nh,
                    const vector<float>& q_w, const vector<float>& q_b,
                    const vector<float>& k_w, const vector<float>& k_b,
                    const vector<float>& v_w, const vector<float>& v_b,
                    const vector<float>& pos_w, const vector<float>& bu,
                    const vector<float>& bv, const vector<float>& out_w,
                    const vector<float>& out_b, vector<float>* y) {
    const int dk = c / nh;
    const int p = 2 * t - 1;
    const std::size_t tc = (std::size_t)t * c, pc = (std::size_t)p * c;
    vector<float> dq(tc), dkb(tc), dv(tc), dp(pc), qu(tc), qv(tc);
    diar::nn::linear_forward(x, q_w.data(), q_b.data(), dq.data(), t, c, c);
    diar::nn::linear_forward(x, k_w.data(), k_b.data(), dkb.data(), t, c, c);
    diar::nn::linear_forward(x, v_w.data(), v_b.data(), dv.data(), t, c, c);
    diar::nn::linear_forward(pos, pos_w.data(), nullptr, dp.data(), p, c, c);
    for (std::size_t i = 0; i < tc; ++i) {
        const int col = (int)(i % c);
        qu[i] = dq[i] + bu[col];
        qv[i] = dq[i] + bv[col];
    }
    // ac[h][i][j] = qu_i . k_j ; bd[h][n][m] = qv_n . p_m   (row-major)
    vector<float> ac((std::size_t)nh * t * t);
    vector<float> bd((std::size_t)nh * t * p);
    for (int h = 0; h < nh; ++h)
        for (int i = 0; i < t; ++i)
            for (int j = 0; j < t; ++j) {
                float acc = 0;
                for (int e = 0; e < dk; ++e)
                    acc += qu[(std::size_t)i * c + h * dk + e] *
                           dkb[(std::size_t)j * c + h * dk + e];
                ac[((std::size_t)h * t + i) * t + j] = acc;
            }
    for (int h = 0; h < nh; ++h)
        for (int n = 0; n < t; ++n)
            for (int m = 0; m < p; ++m) {
                float acc = 0;
                for (int e = 0; e < dk; ++e)
                    acc += qv[(std::size_t)n * c + h * dk + e] *
                           dp[(std::size_t)m * c + h * dk + e];
                bd[((std::size_t)h * t + n) * p + m] = acc;
            }
    // softmax kernel: s = (ac[i][j] + bd[i][j-i+t-1]) / sqrt(dk); row softmax
    const float inv_sdk = 1.0f / std::sqrt((float)dk);
    vector<float> probs((std::size_t)nh * t * t);
    for (int h = 0; h < nh; ++h)
        for (int i = 0; i < t; ++i) {
            const std::size_t base = ((std::size_t)h * t + i) * t;
            float mx = -3.0e38f;
            for (int j = 0; j < t; ++j) {
                const float s = (ac[base + j] +
                                 bd[((std::size_t)h * t + i) * p +
                                    (j - i + t - 1)]) * inv_sdk;
                probs[base + j] = s;
                mx = std::max(mx, s);
            }
            float sum = 0;
            for (int j = 0; j < t; ++j) {
                probs[base + j] = std::exp(probs[base + j] - mx);
                sum += probs[base + j];
            }
            const float is = 1.0f / sum;
            for (int j = 0; j < t; ++j) probs[base + j] *= is;
        }
    // ctx GEMM + output: otmp[n][h*dk+m] = sum_r probs[n][r] * v[r][h*dk+m]
    vector<float> otmp(tc);
    for (int h = 0; h < nh; ++h)
        for (int n = 0; n < t; ++n)
            for (int m = 0; m < dk; ++m) {
                float acc = 0;
                for (int r = 0; r < t; ++r)
                    acc += probs[((std::size_t)h * t + n) * t + r] *
                           dv[(std::size_t)r * c + h * dk + m];
                otmp[(std::size_t)n * c + h * dk + m] = acc;
            }
    diar::nn::linear_forward(otmp.data(), out_w.data(), out_b.data(),
                             y->data(), t, c, c);
}

int main() {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    auto rv = [&](int n) {
        vector<float> v(n);
        for (auto& e : v) e = nd(rng);
        return v;
    };
    struct Case { int t, c, nh; };
    const Case cases[] = {{3, 4, 2}, {5, 8, 4}, {12, 8, 2}, {20, 16, 4}};
    bool ok = true;
    for (const auto& cs : cases) {
        const int t = cs.t, c = cs.c, nh = cs.nh, dk = c / nh;
        auto x = rv(t * c);
        auto pos = rv((2 * t - 1) * c);
        vector<float> q_w = rv(c * c), q_b = rv(c), k_w = rv(c * c),
                      k_b = rv(c), v_w = rv(c * c), v_b = rv(c),
                      pos_w = rv(c * c), bu = rv(c), bv = rv(c),
                      out_w = rv(c * c), out_b = rv(c);
        diar::RelPosMhaWeights w;
        w.q_w = q_w.data(); w.q_b = q_b.data();
        w.k_w = k_w.data(); w.k_b = k_b.data();
        w.v_w = v_w.data(); w.v_b = v_b.data();
        w.pos_w = pos_w.data(); w.bu = bu.data(); w.bv = bv.data();
        w.out_w = out_w.data(); w.out_b = out_b.data();
        vector<float> y_ref(t * c);
        diar::relpos_mha_forward(x.data(), pos.data(), w, y_ref.data(), t, c,
                                 nh);
        vector<float> y_sim(t * c);
        sim_mha(x.data(), pos.data(), t, c, nh, q_w, q_b, k_w, k_b, v_w, v_b,
                pos_w, bu, bv, out_w, out_b, &y_sim);
        const double d = max_abs(y_ref, y_sim);
        std::printf("[sim] t=%d c=%d nh=%d dk=%d max_abs=%.3e %s\n", t, c, nh,
                    dk, d, d < 1e-5 ? "GREEN" : "RED");
        if (d >= 1e-5) ok = false;
        (void)dk;
    }
    std::printf("[sim] %s\n", ok ? "ALL GREEN - orientation contract holds"
                                 : "RED - fix kernel indexing");
    return ok ? 0 : 1;
}
