// M2 Stage 2 layer tests: wiring oracles for head / transformer block /
// ConformerFF. Anti-self-certification: every expected value is computed by
// an INDEPENDENT naive double-precision implementation in this file (never
// by diar::nn or diar:: layer functions), plus hand-computed closed forms.
//
// What these tests own: op ORDER, residual placement, scale placement, LN
// placement (pre vs post), activation choice. What they explicitly do NOT
// own: the head-split convention vs NeMo weights and absolute numerical
// agreement vs NeMo — those belong to the Kaggle teacher-forced gates
// (kaggle/m2_stage2), which run true weights. A convention bug (interleaved
// vs contiguous split) passes here and dies loudly there by design.
#include "diar/layers.hpp"

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

// ---- independent double oracles (naive loops, no diar:: calls) ----

void dbl_linear(const std::vector<double>& x, const std::vector<double>& w,
    const std::vector<double>& b, std::vector<double>& y, int t, int in, int out) {
    for (int r = 0; r < t; ++r)
        for (int o = 0; o < out; ++o) {
            double acc = b.empty() ? 0.0 : b[o];
            for (int i = 0; i < in; ++i) acc += x[r * in + i] * w[o * in + i];
            y[r * out + o] = acc;
        }
}

void dbl_layernorm(const std::vector<double>& x, const std::vector<double>& g,
    const std::vector<double>& b, std::vector<double>& y, int t, int c) {
    for (int r = 0; r < t; ++r) {
        double mean = 0.0;
        for (int i = 0; i < c; ++i) mean += x[r * c + i];
        mean /= c;
        double var = 0.0;
        for (int i = 0; i < c; ++i) {
            const double d = x[r * c + i] - mean;
            var += d * d;
        }
        var /= c;
        const double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int i = 0; i < c; ++i)
            y[r * c + i] = (x[r * c + i] - mean) * inv * (g.empty() ? 1.0 : g[i]) +
                           (b.empty() ? 0.0 : b[i]);
    }
}

// ---- head ----

void test_head_hand_case() {
    // T=1 H=4 S=2. Wiring trap: W has negative/off-diagonal entries so that
    // relu-before-linear != linear-before-relu (identity/permutation weights
    // would commute and hide an order swap).
    const std::vector<float> x = {-1.0F, 0.5F, 2.0F, -0.25F};
    const std::vector<float> hw = {
        1, -1, 0, 0,  // r0
        0, 1, 2, 0,  // r1
        0, 0, 1, -1,  // r2
        2, 0, 0, 1,  // r3
    };
    const std::vector<float> hb = {0.25F, -0.5F, 0.0F, 0.5F};
    const std::vector<float> sw = {
        1, 0, -1, 2,  // s0
        0, 1, 1, -1,  // s1
    };
    const std::vector<float> sb = {0.1F, -0.2F};
    std::vector<float> y(2, 0.0F);
    diar::diar_head_forward(x.data(), hw.data(), hb.data(), sw.data(), sb.data(), y.data(), 1, 4, 2);
    // relu(x) = [0,.5,2,0]; lin -> [-.25, 4, 2, .5]; relu -> [0,4,2,.5];
    // logits = [-0.9, 5.3].
    const double e0 = 1.0 / (1.0 + std::exp(0.9));
    const double e1 = 1.0 / (1.0 + std::exp(-5.3));
    expect_near(y[0], e0, 1e-6, "head hand s0");
    expect_near(y[1], e1, 1e-6, "head hand s1");
}

void test_head_random_vs_double_oracle() {
    std::uint64_t s = 777;
    const int t = 3, h = 6, nspk = 2;
    std::vector<float> x(t * h), hw(h * h), hb(h), sw(nspk * h), sb(nspk);
    for (auto& v : x) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : hw) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : hb) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : sw) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : sb) v = static_cast<float>(lcg_uniform(s));
    std::vector<float> y(t * nspk, 0.0F);
    diar::diar_head_forward(
        x.data(), hw.data(), hb.data(), sw.data(), sb.data(), y.data(), t, h, nspk);
    std::vector<double> dx(x.begin(), x.end()), dhw(hw.begin(), hw.end()),
        dhb(hb.begin(), hb.end()), dsw(sw.begin(), sw.end()), dsb(sb.begin(), sb.end());
    std::vector<double> a(t * h), h1(t * h), logits(t * nspk);
    for (int i = 0; i < t * h; ++i) a[i] = dx[i] > 0 ? dx[i] : 0.0;
    dbl_linear(a, dhw, dhb, h1, t, h, h);
    for (auto& v : h1) v = v > 0 ? v : 0.0;
    dbl_linear(h1, dsw, dsb, logits, t, h, nspk);
    for (int i = 0; i < t * nspk; ++i) {
        const double e = 1.0 / (1.0 + std::exp(-logits[i]));
        expect_near(y[i], e, 1e-5, "head double oracle");
    }
}

// ---- conformer FF ----

void test_conformer_ff_hand_case() {
    // T=1 d=2 dff=3. lin1 -> silu -> lin2, double oracle.
    const std::vector<float> x = {1.0F, -2.0F};
    const std::vector<float> w1 = {1, 0, 0, 1, -1, 2};  // rows: [1,0],[0,1],[-1,2]
    const std::vector<float> b1 = {0.5F, -0.5F, 0.0F};
    const std::vector<float> w2 = {1, 1, 0, 0, 0, 1};  // rows: [1,1,0],[0,0,1]
    const std::vector<float> b2 = {0.25F, -0.25F};
    std::vector<float> y(2, 0.0F);
    diar::conformer_ff_forward(x.data(), w1.data(), b1.data(), w2.data(), b2.data(), y.data(), 1, 2, 3);
    // lin1(x) = [1.5, -2.5, -5]; silu(z) = z/(1+e^-z).
    auto silu = [](double z) { return z / (1.0 + std::exp(-z)); };
    const double m0 = silu(1.5), m1 = silu(-2.5), m2 = silu(-5.0);
    expect_near(y[0], m0 + m1 + 0.25, 1e-6, "cff hand 0");
    expect_near(y[1], m2 - 0.25, 1e-6, "cff hand 1");
}

void test_conformer_ff_random_vs_double_oracle() {
    std::uint64_t s = 4242;
    const int t = 4, d = 5, f = 11;
    std::vector<float> x(t * d), w1(f * d), b1(f), w2(d * f), b2(d);
    for (auto& v : x) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : w1) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : b1) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : w2) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : b2) v = static_cast<float>(lcg_uniform(s));
    std::vector<float> y(t * d, 0.0F);
    diar::conformer_ff_forward(
        x.data(), w1.data(), b1.data(), w2.data(), b2.data(), y.data(), t, d, f);
    std::vector<double> dx(x.begin(), x.end()), dw1(w1.begin(), w1.end()),
        db1(b1.begin(), b1.end()), dw2(w2.begin(), w2.end()), db2(b2.begin(), b2.end());
    std::vector<double> mid(t * f), act(t * f), out(t * d);
    dbl_linear(dx, dw1, db1, mid, t, d, f);
    for (int i = 0; i < t * f; ++i) act[i] = mid[i] / (1.0 + std::exp(-mid[i]));
    dbl_linear(act, dw2, db2, out, t, f, d);
    for (int i = 0; i < t * d; ++i) expect_near(y[i], out[i], 1e-4, "cff double oracle");
}

// ---- transformer block ----

void test_transformer_block_tiny_vs_double_oracle() {
    // T=2 H=4 heads=2 inner=8. Full naive double oracle: catches residual /
    // scale / LN-placement / activation wiring errors. Head-split CONVENTION
    // vs NeMo is owned by the Kaggle true-weight gates, not here.
    std::uint64_t s = 90210;
    const int t = 2, h = 4, inner = 8, nh = 2, dk = 2;
    diar::TransformerBlockWeights w;
    std::vector<float> x(t * h), qw(h * h), qb(h), kw(h * h), kb(h), vw(h * h), vb(h), ow(h * h),
        ob(h), g1(h), b1(h), f1(inner * h), fb1(inner), f2(h * inner), fb2(h), g2(h), b2(h);
    for (auto& v : x) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : qw) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : qb) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : kw) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : kb) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : vw) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : vb) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : ow) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : ob) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : g1) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : b1) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : f1) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : fb1) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : f2) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : fb2) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : g2) v = static_cast<float>(lcg_uniform(s));
    for (auto& v : b2) v = static_cast<float>(lcg_uniform(s));
    w.q_w = qw.data();
    w.q_b = qb.data();
    w.k_w = kw.data();
    w.k_b = kb.data();
    w.v_w = vw.data();
    w.v_b = vb.data();
    w.o_w = ow.data();
    w.o_b = ob.data();
    w.ln1_g = g1.data();
    w.ln1_b = b1.data();
    w.ln2_g = g2.data();
    w.ln2_b = b2.data();
    w.f1_w = f1.data();
    w.f1_b = fb1.data();
    w.f2_w = f2.data();
    w.f2_b = fb2.data();
    std::vector<float> y(t * h, 0.0F);
    diar::transformer_block_forward(x.data(), w, y.data(), t, h, inner, nh);

    const std::vector<double> dx(x.begin(), x.end());
    auto D = [](const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); };
    const std::vector<double> dqw = D(qw), dqb = D(qb), dkw = D(kw), dkb = D(kb), dvw = D(vw),
                              dvb = D(vb), dow = D(ow), dob = D(ob), dg1 = D(g1), db1 = D(b1),
                              df1 = D(f1), dfb1 = D(fb1), df2 = D(f2), dfb2 = D(fb2), dg2 = D(g2),
                              db2 = D(b2), empty;
    std::vector<double> q(t * h), k(t * h), v(t * h);
    dbl_linear(dx, dqw, dqb, q, t, h, h);
    dbl_linear(dx, dkw, dkb, k, t, h, h);
    dbl_linear(dx, dvw, dvb, v, t, h, h);
    const double scale = 1.0 / std::sqrt(static_cast<double>(dk));
    std::vector<double> attn(t * h, 0.0);
    for (int hd = 0; hd < nh; ++hd) {
        std::vector<double> sc(t * t), pr(t * t);
        for (int i = 0; i < t; ++i) {
            double mx = -1e300;
            for (int j = 0; j < t; ++j) {
                double acc = 0.0;
                for (int e = 0; e < dk; ++e) acc += q[i * h + hd * dk + e] * k[j * h + hd * dk + e];
                sc[i * t + j] = acc * scale;
                mx = std::max(mx, sc[i * t + j]);
            }
            double sum = 0.0;
            for (int j = 0; j < t; ++j) {
                pr[i * t + j] = std::exp(sc[i * t + j] - mx);
                sum += pr[i * t + j];
            }
            for (int j = 0; j < t; ++j) pr[i * t + j] /= sum;
        }
        for (int i = 0; i < t; ++i)
            for (int e = 0; e < dk; ++e) {
                double acc = 0.0;
                for (int j = 0; j < t; ++j) acc += pr[i * t + j] * v[j * h + hd * dk + e];
                attn[i * h + hd * dk + e] = acc;
            }
    }
    std::vector<double> ao(t * h), h1(t * h), h1ln(t * h), fm(t * inner), fo(t * h), h2(t * h),
        out(t * h);
    dbl_linear(attn, dow, dob, ao, t, h, h);
    for (int i = 0; i < t * h; ++i) h1[i] = ao[i] + dx[i];
    dbl_layernorm(h1, dg1, db1, h1ln, t, h);
    dbl_linear(h1ln, df1, dfb1, fm, t, h, inner);
    for (auto& z : fm) z = z > 0 ? z : 0.0;
    dbl_linear(fm, df2, dfb2, fo, t, inner, h);
    for (int i = 0; i < t * h; ++i) h2[i] = fo[i] + h1ln[i];
    dbl_layernorm(h2, dg2, db2, out, t, h);
    for (int i = 0; i < t * h; ++i) expect_near(y[i], out[i], 1e-4, "tblk double oracle");
}

void test_transformer_block_fullsize_smoke() {
    // Real geometry (T=60, H=192, inner=768, 8 heads): no-crash + finite.
    // Catches indexing bugs that tiny shapes miss. No oracle (oracle is the
    // Kaggle true-weight gate).
    std::uint64_t s = 31337;
    const int t = 60, h = 192, inner = 768, nh = 8;
    diar::TransformerBlockWeights w;
    std::vector<float> x(t * h), qw(h * h), qb(h), kw(h * h), kb(h), vw(h * h), vb(h), ow(h * h),
        ob(h), g1(h), b1(h), f1(inner * h), fb1(inner), f2(h * inner), fb2(h), g2(h), b2(h);
    for (auto& v : x) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : qw) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : qb) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : kw) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : kb) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : vw) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : vb) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : ow) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : ob) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : g1) v = 1.0F;
    for (auto& v : b1) v = 0.0F;
    for (auto& v : f1) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : fb1) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : f2) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : fb2) v = static_cast<float>(lcg_uniform(s)) * 0.1F;
    for (auto& v : g2) v = 1.0F;
    for (auto& v : b2) v = 0.0F;
    w.q_w = qw.data();
    w.q_b = qb.data();
    w.k_w = kw.data();
    w.k_b = kb.data();
    w.v_w = vw.data();
    w.v_b = vb.data();
    w.o_w = ow.data();
    w.o_b = ob.data();
    w.ln1_g = g1.data();
    w.ln1_b = b1.data();
    w.ln2_g = g2.data();
    w.ln2_b = b2.data();
    w.f1_w = f1.data();
    w.f1_b = fb1.data();
    w.f2_w = f2.data();
    w.f2_b = fb2.data();
    std::vector<float> y(t * h, 0.0F);
    diar::transformer_block_forward(x.data(), w, y.data(), t, h, inner, nh);
    for (int i = 0; i < t * h; ++i) expect(std::isfinite(y[i]), "tblk fullsize finite");
}

}  // namespace

int main() {
    test_head_hand_case();
    test_head_random_vs_double_oracle();
    test_conformer_ff_hand_case();
    test_conformer_ff_random_vs_double_oracle();
    test_transformer_block_tiny_vs_double_oracle();
    test_transformer_block_fullsize_smoke();
    std::cout << "layers tests passed\n";
    return 0;
}
