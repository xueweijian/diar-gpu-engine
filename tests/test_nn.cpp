// M2 Stage 1 op-level tests.
//
// Anti-self-certification rule (M2-NEURAL-CORE-PLAN Stage 1): every check
// below uses an INDEPENDENT oracle, never the op under test:
//   - linear/matmul: hand-computed small cases + commutativity-free identity
//   - layernorm: zero-mean/unit-variance property + closed-form small case
//     computed in double (oracle lives in this file, float loop in src/)
//   - softmax: row-sum-to-one property + closed-form e-based check
//   - activations/GLU: closed-form values from exp/sigmoid definitions
//   - convs: naive full-tensor reference (different loop order, zero-insert
//     then correlate) vs the op's skip-OOB formulation
//   - rel-shift: brute-force portrait of the ggml op chain
//     (zero-pad/concat/reshape/view-strip simulated with std::vector
//     indexing, no call to rel_shift_forward), T=2..6
//   - xscale: sqrt(512) closed form
// Mutation coverage (each must fail if the source regresses):
//   LN unbiased-variance (/n-1) breaks zero-mean/unit-var: pinned by property
//   LN eps outside sqrt: breaks the closed-form small case
//   GLU halves swapped: breaks the asymmetric gate case
//   conv causal pad instead of symmetric: breaks the edge-frame hand case
//   rel-shift off-by-one (BD[k-j+T,j]): breaks all T>=2 brute-force cases
// Wiring pins that must move WITH the source if upstream ever changes:
//   kLayerNormEps / kBatchNormEps == 1e-5 (nn.cpp LayerNorm::build_graph /
//   BatchNorm1d::set_data @ a5b6953). A drifted eps fails the closed-form
//   case only if the test recomputes it from the same constant -- so the
//   eps literal 1e-5 is ALSO spelled out raw below (eps_literal_mismatch).
#include "diar/nn.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
    expect(std::fabs(actual - expected) <= tolerance,
        message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

std::uint64_t lcg_next(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
}

double lcg_uniform(std::uint64_t& state) {
    return static_cast<double>(lcg_next(state) % 20001) / 10000.0 - 1.0;
}

// ---- linear ----

void test_linear_hand_case() {
    // x (2x3), W (2x3, out-major), b (2,): hand-computed.
    const std::vector<float> x = {1, 2, 3, 4, 5, 6};
    const std::vector<float> w = {1, 0, -1, 0, 1, 2};  // row0: x0-x2, row1: x1+2*x2
    const std::vector<float> b = {0.5F, -1.0F};
    std::vector<float> y(4, 0.0F);
    diar::nn::linear_forward(x.data(), w.data(), b.data(), y.data(), 2, 3, 2);
    // row0: [1-3+0.5, 2+6-1] = [-1.5, 7]; row1: [4-6+0.5, 5+12-1] = [-1.5, 16]
    expect_near(y[0], -1.5, 1e-6, "linear row0 out0");
    expect_near(y[1], 7.0, 1e-6, "linear row0 out1");
    expect_near(y[2], -1.5, 1e-6, "linear row1 out0");
    expect_near(y[3], 16.0, 1e-6, "linear row1 out1");
}

void test_linear_no_bias_identity() {
    // W = identity (4x4), no bias: output must equal input exactly.
    std::vector<float> w(16, 0.0F);
    for (int i = 0; i < 4; ++i) w[i * 4 + i] = 1.0F;
    const std::vector<float> x = {3, -1, 0.5F, 2, 7, 8, -9, 0.25F, 1, 1, 1, 1};
    std::vector<float> y(12, 0.0F);
    diar::nn::linear_forward(x.data(), w.data(), nullptr, y.data(), 3, 4, 4);
    for (std::size_t i = 0; i < 12; ++i) expect(y[i] == x[i], "linear identity passthrough");
}

void test_linear_random_vs_double_oracle() {
    std::uint64_t s = 12345;
    const std::size_t t = 5, in = 7, out = 6;
    std::vector<float> x(t * in), w(out * in), b(out), y(t * out, 0.0F);
    for (auto& v : x) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : w) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : b) v = static_cast<float>(lcg_uniform(s));
    diar::nn::linear_forward(x.data(), w.data(), b.data(), y.data(), t, in, out);
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t o = 0; o < out; ++o) {
            double acc = 0.0;
            for (std::size_t i = 0; i < in; ++i)
                acc += static_cast<double>(x[r * in + i]) * w[o * in + i];
            acc += b[o];
            expect_near(y[r * out + o], acc, 1e-4, "linear double oracle");
        }
    }
}

// ---- layernorm ----

void test_layernorm_eps_literal() {
    // The eps literal must stay 1e-5 (upstream ggml_norm(...,1e-5) @ a5b6953).
    // Spelled raw so a source-constant drift fails here.
    expect(diar::nn::kLayerNormEps == 1e-5F, "layernorm eps literal 1e-5");
    expect(diar::nn::kBatchNormEps == 1e-5F, "batchnorm eps literal 1e-5");
}

void test_layernorm_closed_form() {
    // Row [1,2,3,4], gamma=1, beta=0: mean=2.5, biased var=1.25,
    // inv = 1/sqrt(1.25+1e-5).
    const std::vector<float> x = {1, 2, 3, 4};
    std::vector<float> y(4, 0.0F);
    diar::nn::layernorm_forward(x.data(), nullptr, nullptr, y.data(), 1, 4);
    const double inv = 1.0 / std::sqrt(1.25 + 1e-5);
    expect_near(y[0], -1.5 * inv, 1e-6, "ln closed form 0");
    expect_near(y[1], -0.5 * inv, 1e-6, "ln closed form 1");
    expect_near(y[2], 0.5 * inv, 1e-6, "ln closed form 2");
    expect_near(y[3], 1.5 * inv, 1e-6, "ln closed form 3");
}

void test_layernorm_properties() {
    // Random rows: output has zero mean and unit biased variance; affine applies.
    // NOTE: a biased-vs-unbiased (/n vs /n-1) mixup is invisible to this
    // property test (both normalize to unit variance under their own
    // denominator), so the /n choice is pinned INSTEAD by
    // test_layernorm_biased_vs_unbiased below with an analytic gap far
    // above float noise. Do not "simplify" that test away.
    std::uint64_t s = 999;
    const std::size_t t = 4, c = 32;
    std::vector<float> x(t * c), g(c), b(c), y(t * c, 0.0F);
    for (auto& v : x) v = static_cast<float>(lcg_uniform(s)) * 10.0F;
    for (auto& v : g) v = static_cast<float>(lcg_uniform(s)) + 1.5F;
    for (auto& v : b) v = static_cast<float>(lcg_uniform(s));
    diar::nn::layernorm_forward(x.data(), g.data(), b.data(), y.data(), t, c);
    for (std::size_t r = 0; r < t; ++r) {
        // Strip affine: z = (y-b)/g, then check zero mean / unit var.
        double sum = 0.0, sq = 0.0;
        for (std::size_t i = 0; i < c; ++i) {
            const double z = (y[r * c + i] - b[i]) / g[i];
            sum += z;
            sq += z * z;
        }
        expect_near(sum / c, 0.0, 1e-4, "ln zero mean");
        expect_near(sq / c, 1.0, 1e-3, "ln unit biased variance");
    }
}

void test_layernorm_biased_vs_unbiased() {
    // ggml_norm divides by n (biased); torch.nn.LayerNorm does the same.
    // An unbiased (/n-1) build normalizes to biased-variance (n-1)/n instead
    // of 1 -- analytic gap, not float noise. c=2, x=[0, 2]: biased var=1,
    // unbiased var=2, so z_biased = [-1, 1] exactly (eps=0 removes the guard
    // from the comparison).
    const std::vector<float> x = {0.0F, 2.0F};
    std::vector<float> y(2, 0.0F);
    diar::nn::layernorm_forward(x.data(), nullptr, nullptr, y.data(), 1, 2, 0.0F);
    expect_near(y[0], -1.0, 1e-6, "ln biased z0");
    expect_near(y[1], 1.0, 1e-6, "ln biased z1");
    // Under /n-1 the same input gives [-0.7071, 0.7071]; assert we are NOT there.
    expect(std::fabs(y[0] + 1.0) < 0.01, "ln must be biased (/n), not unbiased");
}

// ---- batchnorm infer ----

void test_batchnorm_infer_closed_form() {
    // x=[3,5], mean=[1,2], var=[4,9], w=[2,0.5], b=[1,-1], eps=1e-5:
    // y0 = 2*2/sqrt(4+eps)+1, y1 = 0.5*3/sqrt(9+eps)-1.
    const std::vector<float> x = {3, 5};
    const std::vector<float> w = {2, 0.5F}, b = {1, -1}, m = {1, 2}, v = {4, 9};
    std::vector<float> y(2, 0.0F);
    diar::nn::batchnorm1d_infer_forward(
        x.data(), w.data(), b.data(), m.data(), v.data(), y.data(), 1, 2);
    expect_near(y[0], 2.0 * 2.0 / std::sqrt(4.0 + 1e-5) + 1.0, 1e-6, "bn closed form 0");
    expect_near(y[1], 0.5 * 3.0 / std::sqrt(9.0 + 1e-5) - 1.0, 1e-6, "bn closed form 1");
}

void test_batchnorm_random_vs_double_oracle() {
    std::uint64_t s = 4242;
    const std::size_t t = 6, c = 9;
    std::vector<float> x(t * c), w(c), b(c), m(c), v(c), y(t * c, 0.0F);
    for (auto& vv : x) vv = static_cast<float>(lcg_uniform(s)) * 5.0F;
    for (auto& vv : w) vv = static_cast<float>(lcg_uniform(s)) + 1.0F;
    for (auto& vv : b) vv = static_cast<float>(lcg_uniform(s));
    for (auto& vv : m) vv = static_cast<float>(lcg_uniform(s));
    for (auto& vv : v) vv = std::fabs(static_cast<float>(lcg_uniform(s))) + 0.1F;
    diar::nn::batchnorm1d_infer_forward(
        x.data(), w.data(), b.data(), m.data(), v.data(), y.data(), t, c);
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t i = 0; i < c; ++i) {
            const double e = (static_cast<double>(x[r * c + i]) - m[i]) /
                    std::sqrt(static_cast<double>(v[i]) + 1e-5) * w[i] +
                b[i];
            expect_near(y[r * c + i], e, 1e-5, "bn double oracle");
        }
    }
}

// ---- softmax ----

void test_softmax_sums_to_one_and_closed_form() {
    const std::vector<float> x = {1, 2, 3, 1000, 1000, -1000};
    std::vector<float> y(6, 0.0F);
    diar::nn::softmax_last_dim(x.data(), y.data(), 2, 3);
    expect_near(y[0] + y[1] + y[2], 1.0, 1e-6, "softmax row0 sums to one");
    expect_near(y[3] + y[4] + y[5], 1.0, 1e-6, "softmax row1 sums to one");
    const double e0 = 1.0, e1 = std::exp(1.0), e2 = std::exp(2.0);
    const double s = e0 + e1 + e2;
    expect_near(y[0], e0 / s, 1e-6, "softmax closed form 0");
    expect_near(y[2], e2 / s, 1e-6, "softmax closed form 2");
    expect_near(y[3], 0.5, 1e-6, "softmax large-equal split");
    expect(y[5] == 0.0F, "softmax -1000 underflows to exact zero");
    for (float v : y) expect(std::isfinite(v), "softmax finite (max-subtracted)");
}

void test_softmax_masked_neginf_row() {
    // One valid logit among -1e30 pads: prob ~1 on the valid slot.
    const float neg = -1e30F;
    const std::vector<float> x = {neg, 2.0F, neg, neg};
    std::vector<float> y(4, 0.0F);
    diar::nn::softmax_last_dim(x.data(), y.data(), 1, 4);
    expect_near(y[1], 1.0, 1e-6, "softmax masked row picks valid slot");
    expect_near(y[0] + y[2] + y[3], 0.0, 1e-6, "softmax masked slots zero");
}

// ---- activations ----

void test_relu() {
    std::vector<float> x = {-2, -0.0F, 0.0F, 3.5F, -1e-9F};
    diar::nn::relu_inplace(x.data(), x.size());
    expect(x[0] == 0.0F && x[1] == 0.0F && x[2] == 0.0F, "relu kills negatives/zeros");
    expect(x[3] == 3.5F, "relu keeps positives");
    expect(x[4] == 0.0F, "relu kills tiny negatives");
}

void test_silu_closed_form() {
    // silu(0)=0, silu(1)=1/(1+e^-1), silu(-2)=-2*sigmoid(-2).
    const std::vector<float> x = {0.0F, 1.0F, -2.0F, 10.0F};
    std::vector<float> y(4, 0.0F);
    diar::nn::silu_forward(x.data(), y.data(), 4);
    expect(y[0] == 0.0F, "silu(0)=0");
    expect_near(y[1], 1.0 / (1.0 + std::exp(-1.0)), 1e-6, "silu(1)");
    expect_near(y[2], -2.0 / (1.0 + std::exp(2.0)), 1e-6, "silu(-2)");
    expect_near(y[3], 10.0 / (1.0 + std::exp(-10.0)), 1e-4, "silu(10)~10");
}

void test_sigmoid_closed_form() {
    const std::vector<float> x = {0.0F, 100.0F, -100.0F};
    std::vector<float> y(3, 0.0F);
    diar::nn::sigmoid_forward(x.data(), y.data(), 3);
    expect(y[0] == 0.5F, "sigmoid(0)=0.5");
    expect(y[1] == 1.0F, "sigmoid(100) saturates to 1");
    expect(y[2] == 0.0F, "sigmoid(-100) saturates to 0");
}

// ---- GLU ----

void test_glu_asymmetric_gate() {
    // Halves swapped MUST fail: a=[10,20], b=[100,100] (gate~1) vs the
    // swapped reading (a=[100,100], b=[10,20]) give different outputs.
    const std::vector<float> x = {10.0F, 20.0F, 100.0F, 100.0F};  // t=1, c=2
    std::vector<float> y(2, 0.0F);
    diar::nn::glu_forward(x.data(), y.data(), 1, 2);
    expect_near(y[0], 10.0, 1e-3, "glu first-half gate~1 keeps a0");
    expect_near(y[1], 20.0, 1e-3, "glu first-half gate~1 keeps a1");
    const std::vector<float> xs = {0.0F, 0.0F, 5.0F, -5.0F};
    std::vector<float> ys(2, 0.0F);
    diar::nn::glu_forward(xs.data(), ys.data(), 1, 2);
    expect(ys[0] == 0.0F && ys[1] == 0.0F, "glu zero first half stays zero");
}

void test_glu_closed_form_random() {
    std::uint64_t s = 777;
    const std::size_t t = 3, c = 5;
    std::vector<float> x(t * 2 * c), y(t * c, 0.0F);
    for (auto& v : x) v = static_cast<float>(lcg_uniform(s)) * 3.0F;
    diar::nn::glu_forward(x.data(), y.data(), t, c);
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t i = 0; i < c; ++i) {
            const double a = x[r * 2 * c + i];
            const double bb = x[r * 2 * c + c + i];
            expect_near(y[r * c + i], a / (1.0 + std::exp(-bb)), 1e-5, "glu double oracle");
        }
    }
}

// ---- conv1d (naive zero-insert oracle, different loop order) ----

void naive_conv1d(const std::vector<float>& x, const std::vector<float>& w,
    const std::vector<float>& b, std::vector<float>& y, int t_in, int c_in, int c_out, int k,
    int s, int p) {
    // Zero-insert padding explicitly, then correlate (vs op's skip-OOB).
    const int tp = t_in + 2 * p;
    std::vector<float> xp(static_cast<std::size_t>(tp) * c_in, 0.0F);
    for (int t = 0; t < t_in; ++t)
        for (int i = 0; i < c_in; ++i) xp[(t + p) * c_in + i] = x[t * c_in + i];
    const int t_out = (t_in + 2 * p - k) / s + 1;
    for (int t = 0; t < t_out; ++t) {
        for (int o = 0; o < c_out; ++o) {
            double acc = 0.0;
            for (int i = 0; i < c_in; ++i)
                for (int kk = 0; kk < k; ++kk)
                    acc += static_cast<double>(xp[(t * s + kk) * c_in + i]) *
                        w[(o * c_in + i) * k + kk];
            y[t * c_out + o] =
                static_cast<float>(acc + (b.empty() ? 0.0 : b[o]));
        }
    }
}

void test_conv1d_hand_symmetric_pad() {
    // x=[1,2,3,4,5] (C=1), w=[1,0,-1] (k=3), symmetric p=1, s=1:
    // edges use zeros -> y = [-2,-2,-2,-2,4].
    const std::vector<float> x = {1, 2, 3, 4, 5};
    const std::vector<float> w = {1, 0, -1};
    std::vector<float> y(5, 0.0F);
    diar::nn::conv1d_forward(x.data(), w.data(), nullptr, y.data(), 5, 1, 1, 3, 1, 1);
    const std::vector<double> e = {-2, -2, -2, -2, 4};
    for (int i = 0; i < 5; ++i) expect_near(y[i], e[i], 1e-6, "conv1d symmetric edge");
}

void test_conv1d_random_vs_naive() {
    std::uint64_t s = 31337;
    const int t_in = 11, c_in = 3, c_out = 4, k = 3, st = 2, p = 1;
    const int t_out = (t_in + 2 * p - k) / st + 1;
    std::vector<float> x(t_in * c_in), w(c_out * c_in * k), b(c_out), y(t_out * c_out, 0),
        ref(t_out * c_out, 0);
    for (auto& v : x) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : w) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : b) v = static_cast<float>(lcg_uniform(s));
    diar::nn::conv1d_forward(x.data(), w.data(), b.data(), y.data(), t_in, c_in, c_out, k, st, p);
    naive_conv1d(x, w, b, ref, t_in, c_in, c_out, k, st, p);
    for (std::size_t i = 0; i < y.size(); ++i) expect_near(y[i], ref[i], 1e-4, "conv1d naive");
}

void test_conv1d_depthwise() {
    // Depthwise k=9 p=4 on constant-1 input with unit weights: interior == 9,
    // first frame == 5 (4 real taps + 5 zeros), last frame == 5.
    const int T = 12, C = 2;
    std::vector<float> x(T * C, 1.0F), w(C * 9, 1.0F), y(T * C, 0.0F);
    diar::nn::conv1d_depthwise_forward(x.data(), w.data(), nullptr, y.data(), T, C, 9, 1, 4);
    for (int c = 0; c < C; ++c) {
        expect_near(y[c], 5.0, 1e-5, "dw conv first frame partial");
        expect_near(y[5 * C + c], 9.0, 1e-5, "dw conv interior full");
        expect_near(y[11 * C + c], 5.0, 1e-5, "dw conv last frame partial");
    }
    // Cross-channel isolation: channel 1 ramp must not leak into channel 0.
    std::vector<float> xr(T * C, 0.0F);
    for (int t = 0; t < T; ++t) xr[t * C + 1] = static_cast<float>(t);
    std::vector<float> yr(T * C, 0.0F);
    diar::nn::conv1d_depthwise_forward(xr.data(), w.data(), nullptr, yr.data(), T, C, 3, 1, 1);
    for (int t = 0; t < T; ++t) {
        double e = 0.0;
        for (int kk = 0; kk < 3; ++kk) {
            const int ti = t - 1 + kk;
            if (ti >= 0 && ti < T) e += ti;
        }
        expect_near(yr[t * C + 1], e, 1e-5, "dw conv ramp");
        expect(yr[t * C] == 0.0F, "dw conv no cross-channel leak");
    }
}

// ---- conv2d ----

void test_conv2d_hand_case() {
    // 1ch 3x3 input 1..9, 2x2 all-ones kernel, s=1, p=0 -> 2x2 sums.
    const std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<float> w = {1, 1, 1, 1};
    std::vector<float> y(4, 0.0F);
    diar::nn::conv2d_forward(x.data(), w.data(), nullptr, y.data(), 1, 3, 3, 1, 2, 2, 1, 1, 0, 0);
    expect_near(y[0], 12.0, 1e-5, "conv2d hand 0");
    expect_near(y[1], 16.0, 1e-5, "conv2d hand 1");
    expect_near(y[2], 24.0, 1e-5, "conv2d hand 2");
    expect_near(y[3], 28.0, 1e-5, "conv2d hand 3");
}

void test_conv2d_subsampling_geometry() {
    // Upstream SubSampling: k=3 s=2 p=1 per stage; 128 wide -> 64 -> 32 -> 16
    // (ceil-div) and T=160 mel -> 80 -> 40 -> 20 (matches chunk000 pre_encode
    // (20,512) with mel_window (160,128)).
    auto ceil_div = [](int l) { return (l + 1) / 2; };
    expect(ceil_div(128) == 64 && ceil_div(64) == 32 && ceil_div(32) == 16, "subsampling F 128->16");
    expect(ceil_div(160) == 80 && ceil_div(80) == 40 && ceil_div(40) == 20, "subsampling T 160->20");
    // Strided shape through the op itself: 1ch 4x4, k=3 s=2 p=1 -> 2x2.
    std::vector<float> x(16, 1.0F), w(9, 1.0F), y(4, 0.0F);
    diar::nn::conv2d_forward(x.data(), w.data(), nullptr, y.data(), 1, 4, 4, 1, 3, 3, 2, 2, 1, 1);
    // Corners see 4 real taps (2x2 overlap), edges 6, center 9.
    expect_near(y[0], 4.0, 1e-5, "subsample corner");
    expect_near(y[3], 9.0, 1e-5, "subsample center");
}

// ---- rel-shift ----

void test_rel_shift_brute_force() {
    // Independent portrait of the ggml op chain (zero-row pad, concat on
    // dim0, metadata-only reshapes, strip offset by one row), handwritten
    // with std::vector indexing -- no call to rel_shift_forward inside.
    for (std::size_t T = 2; T <= 6; ++T) {
        const std::size_t N0 = 2 * T - 1;
        std::vector<float> bd(N0 * T);
        for (std::size_t j = 0; j < T; ++j)
            for (std::size_t i = 0; i < N0; ++i)
                bd[i * T + j] = static_cast<float>(i * 100 + j);
        // ggml linear layout: element (i0,i1) of [N0,T] sits at l=i0+N0*i1
        // (i0 fastest). The zero row is one element per column (interleaved),
        // NOT a contiguous block: M1(i0',i1) = 0 iff i0'==0 else BD(i0'-1,i1).
        std::vector<float> m1(2 * T * T, 0.0F);
        for (std::size_t i1 = 0; i1 < T; ++i1) {
            m1[2 * T * i1] = 0.0F;  // zero row
            for (std::size_t i0 = 0; i0 < N0; ++i0)
                m1[(i0 + 1) + 2 * T * i1] = bd[i0 * T + i1];
        }
        // Reshape [T,2T] is metadata-only (same buffer); the strip view drops
        // N's first column (offset = one N row = T buffer elements); reshape
        // [2T-1,T] metadata-only; keep [T,T]. Net: S[k,j] reads M1-buffer
        // linear T + k + (2T-1)*j (proven separately, T=2..5).
        std::vector<float> got(T * T);
        for (std::size_t k = 0; k < T; ++k)
            for (std::size_t j = 0; j < T; ++j) got[k * T + j] = m1[T + k + (2 * T - 1) * j];
        std::vector<float> out(T * T, 0.0F);
        diar::nn::rel_shift_forward(bd.data(), out.data(), T);
        for (std::size_t i = 0; i < T * T; ++i)
            expect(out[i] == got[i], "rel-shift matches brute-force chain T=" + std::to_string(T));
        // Spot-check the closed form directly: S[k,j] = BD[k-j+T-1,j].
        for (std::size_t k = 0; k < T; ++k)
            for (std::size_t j = 0; j < T; ++j)
                expect(out[k * T + j] == bd[(k - j + T - 1) * T + j], "rel-shift closed form");
    }
}

// ---- xscale ----

void test_xscale_sqrt512() {
    const float s = std::sqrt(512.0F);
    const std::vector<float> x = {1.0F, -2.0F, 0.0F};
    std::vector<float> y(3, 0.0F);
    diar::nn::xscale_forward(x.data(), y.data(), 3, s);
    expect_near(y[0], std::sqrt(512.0), 1e-5, "xscale unit");
    expect_near(y[1], -2.0 * std::sqrt(512.0), 1e-4, "xscale negative");
    expect(y[2] == 0.0F, "xscale zero stays zero");
}

}  // namespace

int main() {
    test_linear_hand_case();
    test_linear_no_bias_identity();
    test_linear_random_vs_double_oracle();
    test_layernorm_eps_literal();
    test_layernorm_closed_form();
    test_layernorm_properties();
    test_layernorm_biased_vs_unbiased();
    test_batchnorm_infer_closed_form();
    test_batchnorm_random_vs_double_oracle();
    test_softmax_sums_to_one_and_closed_form();
    test_softmax_masked_neginf_row();
    test_relu();
    test_silu_closed_form();
    test_sigmoid_closed_form();
    test_glu_asymmetric_gate();
    test_glu_closed_form_random();
    test_conv1d_hand_symmetric_pad();
    test_conv1d_random_vs_naive();
    test_conv1d_depthwise();
    test_conv2d_hand_case();
    test_conv2d_subsampling_geometry();
    test_rel_shift_brute_force();
    test_xscale_sqrt512();
    std::cout << "PASS: nn tensor-core op tests\n";
    return 0;
}
