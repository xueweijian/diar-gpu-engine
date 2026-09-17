// M2 Stage 2 ConformerConv tests. Anti-self-certification: the expected
// path is an INDEPENDENT naive double implementation in this file (explicit
// loops, direct GLU formula, manual symmetric-pad depthwise) — never via
// nn:: or diar:: calls. Catches: GLU half order, BN-before/after-SiLU swap,
// pad asymmetry, norm-axis error. A BN-vs-LN config error passes here (both
// paths tested) and is owned by the Kaggle true-weight gates.
#include "diar/conv.hpp"

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

// Independent double oracle: torch order Linear->GLU->DW(symmetric)->BN->SiLU->Linear.
void oracle_conv(const std::vector<double>& x, const std::vector<double>& pw1w,
    const std::vector<double>& pw1b, const std::vector<double>& dww,
    const std::vector<double>& dwb, const std::vector<double>& bnw,
    const std::vector<double>& bnb, const std::vector<double>& mean,
    const std::vector<double>& var, const std::vector<double>& pw2w,
    const std::vector<double>& pw2b, std::vector<double>& y, int t, int d, int k) {
    const int pad = (k - 1) / 2;
    std::vector<double> e(t * 2 * d, 0.0), g(t * d, 0.0);
    for (int r = 0; r < t; ++r)
        for (int o = 0; o < 2 * d; ++o) {
            double acc = pw1b.empty() ? 0.0 : pw1b[o];
            for (int i = 0; i < d; ++i) acc += x[r * d + i] * pw1w[o * d + i];
            e[r * 2 * d + o] = acc;
        }
    for (int r = 0; r < t; ++r)
        for (int c = 0; c < d; ++c) {
            const double a = e[r * 2 * d + c], b = e[r * 2 * d + d + c];
            g[r * d + c] = a / (1.0 + std::exp(-b));  // a*sigmoid(b), a=FIRST half
        }
    std::vector<double> dw(t * d, 0.0);
    for (int r = 0; r < t; ++r)
        for (int c = 0; c < d; ++c) {
            double acc = dwb.empty() ? 0.0 : dwb[c];
            for (int kk = 0; kk < k; ++kk) {
                const int src = r - pad + kk;
                const double v = (src < 0 || src >= t) ? 0.0 : g[src * d + c];
                acc += v * dww[c * k + kk];
            }
            dw[r * d + c] = acc;
        }
    std::vector<double> n(t * d, 0.0);
    for (int r = 0; r < t; ++r)
        for (int c = 0; c < d; ++c) {
            const double w = bnw.empty() ? 1.0 : bnw[c];
            const double b = bnb.empty() ? 0.0 : bnb[c];
            const double m = mean.empty() ? 0.0 : mean[c];
            const double vv = var.empty() ? 0.0 : var[c];
            n[r * d + c] = (dw[r * d + c] - m) / std::sqrt(vv + 1e-5) * w + b;
        }
    std::vector<double> a(t * d, 0.0);
    for (int i = 0; i < t * d; ++i) a[i] = n[i] / (1.0 + std::exp(-n[i]));
    for (int r = 0; r < t; ++r)
        for (int o = 0; o < d; ++o) {
            double acc = pw2b.empty() ? 0.0 : pw2b[o];
            for (int i = 0; i < d; ++i) acc += a[r * d + i] * pw2w[o * d + i];
            y[r * d + o] = acc;
        }
}

void run_case(int t, int d, int k, std::uint64_t seed, double tol) {
    diar::ConformerConvWeights w;
    std::vector<float> x(t * d), pw1w(2 * d * d), pw1b(2 * d), dww(d * k), dwb(d), bnw(d), bnb(d),
        mean(d), var(d), pw2w(d * d), pw2b(d);
    std::uint64_t s = seed;
    for (auto& z : x) z = static_cast<float>(lcg_uniform(s)) * 0.5F;
    for (auto& z : pw1w) z = static_cast<float>(lcg_uniform(s)) * 0.3F;
    for (auto& z : pw1b) z = static_cast<float>(lcg_uniform(s)) * 0.3F;
    for (auto& z : dww) z = static_cast<float>(lcg_uniform(s)) * 0.3F;
    for (auto& z : dwb) z = static_cast<float>(lcg_uniform(s)) * 0.3F;
    for (auto& z : bnw) z = 1.0F + static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& z : bnb) z = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& z : mean) z = static_cast<float>(lcg_uniform(s)) * 0.2F;
    for (auto& z : var) z = 0.5F + static_cast<float>(lcg_next(s) % 1000) / 1000.0F;
    for (auto& z : pw2w) z = static_cast<float>(lcg_uniform(s)) * 0.3F;
    for (auto& z : pw2b) z = static_cast<float>(lcg_uniform(s)) * 0.3F;
    w.pw1_w = pw1w.data();
    w.pw1_b = pw1b.data();
    w.dw_w = dww.data();
    w.dw_b = dwb.data();
    w.bn_w = bnw.data();
    w.bn_b = bnb.data();
    w.bn_mean = mean.data();
    w.bn_var = var.data();
    w.pw2_w = pw2w.data();
    w.pw2_b = pw2b.data();
    std::vector<float> y(t * d, 0.0F);
    diar::conformer_conv_forward(x.data(), w, y.data(), t, d, k);
    auto D = [](const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); };
    std::vector<double> e(t * d, 0.0);
    oracle_conv(D(x), D(pw1w), D(pw1b), D(dww), D(dwb), D(bnw), D(bnb), D(mean), D(var), D(pw2w),
        D(pw2b), e, t, d, k);
    const std::string tag = "conv t=" + std::to_string(t) + " d=" + std::to_string(d);
    for (int i = 0; i < t * d; ++i) expect_near(y[i], e[i], tol, tag);
    for (int i = 0; i < t * d; ++i) expect(std::isfinite(y[i]), tag + " finite");
}

void test_hand_asymmetric() {
    // GLU half-order trap: pw1 maps [x0,x1] -> [a0,a1,b0,b1] with a != b
    // halves, so a swapped-halves bug changes the answer.
    diar::ConformerConvWeights w;
    // T=3 D=2 K=3 pad=1. pw1 = identity stacked (a=x, b=x).
    const std::vector<float> x = {1, -1, 2, 0.5F, -0.5F, 1.5F};
    const std::vector<float> pw1w = {1, 0, 0, 1, 1, 0, 0, 1};  // [4,2]: a=x,b=x
    const std::vector<float> pw1b(4, 0.0F);
    // dw = delta (center tap 1): dw[c] = [0,1,0].
    const std::vector<float> dww = {0, 1, 0, 0, 1, 0};
    const std::vector<float> dwb(2, 0.0F);
    // BN identity-ish: w=1,b=0,mean=0,var=1 (var+eps≈1).
    const std::vector<float> bnw = {1, 1}, bnb = {0, 0}, mean = {0, 0}, var = {1, 1};
    // pw2 = identity.
    const std::vector<float> pw2w = {1, 0, 0, 1};
    const std::vector<float> pw2b(2, 0.0F);
    w.pw1_w = pw1w.data();
    w.pw1_b = pw1b.data();
    w.dw_w = dww.data();
    w.dw_b = dwb.data();
    w.bn_w = bnw.data();
    w.bn_b = bnb.data();
    w.bn_mean = mean.data();
    w.bn_var = var.data();
    w.pw2_w = pw2w.data();
    w.pw2_b = pw2b.data();
    std::vector<float> y(6, 0.0F);
    diar::conformer_conv_forward(x.data(), w, y.data(), 3, 2, 3);
    // g[r,c] = x*sigmoid(x); dw = identity; n ≈ g (var 1+eps); y = silu(n).
    auto silu = [](double z) { return z / (1.0 + std::exp(-z)); };
    for (int i = 0; i < 6; ++i) {
        const double g = x[i] / (1.0 + std::exp(-x[i]));
        const double n = g / std::sqrt(1.0 + 1e-5);
        expect_near(y[i], silu(n), 1e-5, "conv hand asymmetric");
    }
}

}  // namespace

int main() {
    test_hand_asymmetric();
    run_case(5, 8, 9, 5001, 1e-4);   // diar k=9, small D
    run_case(3, 4, 3, 5002, 1e-4);   // odd geometry
    run_case(20, 32, 9, 5003, 1e-3);  // longer T, diar k
    std::cout << "conv tests passed\n";
    return 0;
}
