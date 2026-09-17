// M2 Stage 2 full-Conformer-layer test. Anti-self-certification: expected
// path is an INDEPENDENT naive double implementation in this file (never via
// nn::/diar:: calls). Catches: FF half-scale missing/at wrong site, norm
// order swaps, residual drops, final-norm missing. Small geometry (the
// module-level suites already cover each submodule at diar sizes).
#include "diar/conformer.hpp"

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

void dbl_linear(const std::vector<double>& x, const std::vector<double>& w,
    const std::vector<double>& b, std::vector<double>& y, int t, int in, int out) {
    for (int r = 0; r < t; ++r)
        for (int o = 0; o < out; ++o) {
            double acc = b.empty() ? 0.0 : b[o];
            for (int i = 0; i < in; ++i) acc += x[r * in + i] * w[o * in + i];
            y[r * out + o] = acc;
        }
}

void dbl_ln(const std::vector<double>& x, const std::vector<double>& g,
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
            y[r * c + i] =
                (x[r * c + i] - mean) * inv * (g.empty() ? 1.0 : g[i]) + (b.empty() ? 0.0 : b[i]);
    }
}

void dbl_ff(const std::vector<double>& x, const std::vector<double>& w1,
    const std::vector<double>& b1, const std::vector<double>& w2, const std::vector<double>& b2,
    std::vector<double>& y, int t, int d, int f) {
    std::vector<double> mid(t * f), act(t * f);
    dbl_linear(x, w1, b1, mid, t, d, f);
    for (int i = 0; i < t * f; ++i) act[i] = mid[i] / (1.0 + std::exp(-mid[i]));
    dbl_linear(act, w2, b2, y, t, f, d);
}

void dbl_conv(const std::vector<double>& x, const std::vector<double>& pw1w,
    const std::vector<double>& pw1b, const std::vector<double>& dww,
    const std::vector<double>& dwb, const std::vector<double>& bnw,
    const std::vector<double>& bnb, const std::vector<double>& mean,
    const std::vector<double>& var, const std::vector<double>& pw2w,
    const std::vector<double>& pw2b, std::vector<double>& y, int t, int d, int k) {
    const int pad = (k - 1) / 2;
    std::vector<double> e(t * 2 * d, 0.0), g(t * d, 0.0);
    dbl_linear(x, pw1w, pw1b, e, t, d, 2 * d);
    for (int r = 0; r < t; ++r)
        for (int c = 0; c < d; ++c) {
            const double a = e[r * 2 * d + c], b = e[r * 2 * d + d + c];
            g[r * d + c] = a / (1.0 + std::exp(-b));
        }
    std::vector<double> dw(t * d, 0.0);
    for (int r = 0; r < t; ++r)
        for (int c = 0; c < d; ++c) {
            double acc = dwb.empty() ? 0.0 : dwb[c];
            for (int kk = 0; kk < k; ++kk) {
                const int src = r - pad + kk;
                acc += ((src < 0 || src >= t) ? 0.0 : g[src * d + c]) * dww[c * k + kk];
            }
            dw[r * d + c] = acc;
        }
    std::vector<double> n(t * d, 0.0), a(t * d, 0.0);
    for (int r = 0; r < t; ++r)
        for (int c = 0; c < d; ++c)
            n[r * d + c] = (dw[r * d + c] - mean[c]) / std::sqrt(var[c] + 1e-5) * bnw[c] + bnb[c];
    for (int i = 0; i < t * d; ++i) a[i] = n[i] / (1.0 + std::exp(-n[i]));
    dbl_linear(a, pw2w, pw2b, y, t, d, d);
}

// Naive double rel-pos MHA (query-major, S[q,k] = BD[k-q+T-1,q]).
void dbl_mha(const std::vector<double>& x, const std::vector<double>& pe,
    const std::vector<double>& qw, const std::vector<double>& qb, const std::vector<double>& kw,
    const std::vector<double>& kb, const std::vector<double>& vw, const std::vector<double>& vb,
    const std::vector<double>& pw, const std::vector<double>& bu, const std::vector<double>& bv,
    const std::vector<double>& ow, const std::vector<double>& ob, std::vector<double>& y, int t,
    int c, int nh) {
    const int dk = c / nh, p = 2 * t - 1;
    const std::vector<double> empty;
    std::vector<double> q(t * c), k(t * c), v(t * c), pp(p * c);
    dbl_linear(x, qw, qb, q, t, c, c);
    dbl_linear(x, kw, kb, k, t, c, c);
    dbl_linear(x, vw, vb, v, t, c, c);
    dbl_linear(pe, pw, empty, pp, p, c, c);
    const double sdk = std::sqrt(static_cast<double>(dk));
    std::vector<double> merged(t * c, 0.0);
    for (int hd = 0; hd < nh; ++hd) {
        for (int i = 0; i < t; ++i) {
            std::vector<double> sc(t), pr(t);
            for (int j = 0; j < t; ++j) {
                double ac = 0.0, bd = 0.0;
                for (int e = 0; e < dk; ++e) {
                    ac += (q[i * c + hd * dk + e] + bu[hd * dk + e]) * k[j * c + hd * dk + e];
                    bd += (q[i * c + hd * dk + e] + bv[hd * dk + e]) *
                          pp[(j - i + t - 1) * c + hd * dk + e];
                }
                sc[j] = (ac + bd) / sdk;
            }
            double mx = sc[0];
            for (int j = 1; j < t; ++j) mx = std::max(mx, sc[j]);
            double sum = 0.0;
            for (int j = 0; j < t; ++j) {
                pr[j] = std::exp(sc[j] - mx);
                sum += pr[j];
            }
            for (int j = 0; j < t; ++j) pr[j] /= sum;
            for (int e = 0; e < dk; ++e) {
                double acc = 0.0;
                for (int j = 0; j < t; ++j) acc += pr[j] * v[j * c + hd * dk + e];
                merged[i * c + hd * dk + e] = acc;
            }
        }
    }
    dbl_linear(merged, ow, ob, y, t, c, c);
}

void test_full_layer() {
    const int t = 4, d = 8, f = 16, nh = 2, k = 5, p = 2 * t - 1;
    std::uint64_t s = 6001;
    auto R = [&](std::vector<float>& v, double sc) {
        for (auto& z : v) z = static_cast<float>(lcg_uniform(s) * sc);
    };
    std::vector<float> x(t * d), pe(p * d);
    std::vector<float> g1(d), b1(d), w1(f * d), bb1(f), w2(d * f), bb2(d);
    std::vector<float> ga(d), ba(d), qw(d * d), qb(d), kw(d * d), kb(d), vw(d * d), vb(d), pw(d * d),
        bu(d), bv(d), ow(d * d), ob(d);
    std::vector<float> gc(d), bc(d), cpw1(2 * d * d), cpw1b(2 * d), cdw(d * k), cdwb(d), cbw(d),
        cbb(d), cmn(d), cvr(d), cpw2(d * d), cpw2b(d);
    std::vector<float> g2(d), b2(d), w3(f * d), bb3(f), w4(d * f), bb4(d), go(d), bo(d);
    R(x, 0.5);
    R(pe, 0.5);
    R(g1, 0.1);
    R(b1, 0.1);
    R(w1, 0.3);
    R(bb1, 0.3);
    R(w2, 0.3);
    R(bb2, 0.3);
    R(ga, 0.1);
    R(ba, 0.1);
    R(qw, 0.3);
    R(qb, 0.3);
    R(kw, 0.3);
    R(kb, 0.3);
    R(vw, 0.3);
    R(vb, 0.3);
    R(pw, 0.3);
    R(bu, 0.3);
    R(bv, 0.3);
    R(ow, 0.3);
    R(ob, 0.3);
    R(gc, 0.1);
    R(bc, 0.1);
    R(cpw1, 0.3);
    R(cpw1b, 0.3);
    R(cdw, 0.3);
    R(cdwb, 0.3);
    for (auto& z : cbw) z = 1.0F;
    R(cbb, 0.1);
    R(cmn, 0.2);
    for (auto& z : cvr) z = 0.5F + static_cast<float>(lcg_next(s) % 1000) / 1000.0F;
    R(cpw2, 0.3);
    R(cpw2b, 0.3);
    R(g2, 0.1);
    R(b2, 0.1);
    R(w3, 0.3);
    R(bb3, 0.3);
    R(w4, 0.3);
    R(bb4, 0.3);
    R(go, 0.1);
    R(bo, 0.1);
    for (auto& z : g1) z += 1.0F;
    for (auto& z : ga) z += 1.0F;
    for (auto& z : gc) z += 1.0F;
    for (auto& z : g2) z += 1.0F;
    for (auto& z : go) z += 1.0F;

    diar::ConformerLayerWeights w;
    w.n_ff1_g = g1.data();
    w.n_ff1_b = b1.data();
    w.n_sa_g = ga.data();
    w.n_sa_b = ba.data();
    w.n_conv_g = gc.data();
    w.n_conv_b = bc.data();
    w.n_ff2_g = g2.data();
    w.n_ff2_b = b2.data();
    w.n_out_g = go.data();
    w.n_out_b = bo.data();
    w.ff1_w1 = w1.data();
    w.ff1_b1 = bb1.data();
    w.ff1_w2 = w2.data();
    w.ff1_b2 = bb2.data();
    w.ff2_w1 = w3.data();
    w.ff2_b1 = bb3.data();
    w.ff2_w2 = w4.data();
    w.ff2_b2 = bb4.data();
    w.attn.q_w = qw.data();
    w.attn.q_b = qb.data();
    w.attn.k_w = kw.data();
    w.attn.k_b = kb.data();
    w.attn.v_w = vw.data();
    w.attn.v_b = vb.data();
    w.attn.pos_w = pw.data();
    w.attn.bu = bu.data();
    w.attn.bv = bv.data();
    w.attn.out_w = ow.data();
    w.attn.out_b = ob.data();
    w.conv.pw1_w = cpw1.data();
    w.conv.pw1_b = cpw1b.data();
    w.conv.dw_w = cdw.data();
    w.conv.dw_b = cdwb.data();
    w.conv.bn_w = cbw.data();
    w.conv.bn_b = cbb.data();
    w.conv.bn_mean = cmn.data();
    w.conv.bn_var = cvr.data();
    w.conv.pw2_w = cpw2.data();
    w.conv.pw2_b = cpw2b.data();
    std::vector<float> y(t * d, 0.0F);
    diar::conformer_layer_forward(x.data(), pe.data(), w, y.data(), t, d, f, nh, k);

    auto D = [](const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); };
    std::vector<double> res(D(x)), n1(t * d), f1(t * d), na(t * d), at(t * d), nc(t * d), cv(t * d),
        n2(t * d), f2(t * d), out(t * d);
    dbl_ln(res, D(g1), D(b1), n1, t, d);
    dbl_ff(n1, D(w1), D(bb1), D(w2), D(bb2), f1, t, d, f);
    for (int i = 0; i < t * d; ++i) res[i] += 0.5 * f1[i];
    dbl_ln(res, D(ga), D(ba), na, t, d);
    dbl_mha(na, D(pe), D(qw), D(qb), D(kw), D(kb), D(vw), D(vb), D(pw), D(bu), D(bv), D(ow), D(ob),
        at, t, d, nh);
    for (int i = 0; i < t * d; ++i) res[i] += at[i];
    dbl_ln(res, D(gc), D(bc), nc, t, d);
    dbl_conv(nc, D(cpw1), D(cpw1b), D(cdw), D(cdwb), D(cbw), D(cbb), D(cmn), D(cvr), D(cpw2),
        D(cpw2b), cv, t, d, k);
    for (int i = 0; i < t * d; ++i) res[i] += cv[i];
    dbl_ln(res, D(g2), D(b2), n2, t, d);
    dbl_ff(n2, D(w3), D(bb3), D(w4), D(bb4), f2, t, d, f);
    for (int i = 0; i < t * d; ++i) res[i] += 0.5 * f2[i];
    dbl_ln(res, D(go), D(bo), out, t, d);
    for (int i = 0; i < t * d; ++i) expect_near(y[i], out[i], 1e-4, "conformer layer");
    for (int i = 0; i < t * d; ++i) expect(std::isfinite(y[i]), "conformer layer finite");
}

}  // namespace

int main() {
    test_full_layer();
    std::cout << "conformer tests passed\n";
    return 0;
}
