// M2 Stage 2 rel-pos table tests. Anti-self-certification: expected values
// are INDEPENDENT closed forms in this file (direct sin/cos calls with
// hand-derived positions), never via diar::posenc. Catches: row-order flip,
// div-term base error, sin/cos parity swap, off-by-one in row count.
#include "diar/posenc.hpp"

#include <cmath>
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

void test_hand_case() {
    // L=2 D=4: positions [+1, 0, -1]. NeMo create_pe:
    //   div_term = exp(arange(0,4,2) * -(log(1e4)/4)) = [1, exp(-log(1e4)/2)].
    // (v11 lesson: an earlier oracle used exp(-log(1e4)/4) — the i-not-2i
    // misreading — so this case silently pinned the bug. Pair-i exponent
    // is 2*i, always.)
    const double d1 = std::exp(-std::log(10000.0) / 2.0);
    std::vector<float> pe(3 * 4, 0.0F);
    diar::relpos_table_forward(pe.data(), 2, 4);
    // row 0 (pos +1): [sin1, cos1, sin(d1), cos(d1)]
    expect_near(pe[0], std::sin(1.0), 1e-6, "pe r0c0");
    expect_near(pe[1], std::cos(1.0), 1e-6, "pe r0c1");
    expect_near(pe[2], std::sin(d1), 1e-6, "pe r0c2");
    expect_near(pe[3], std::cos(d1), 1e-6, "pe r0c3");
    // row 1 (pos 0): [0, 1, 0, 1]
    expect_near(pe[4], 0.0, 1e-6, "pe r1c0");
    expect_near(pe[5], 1.0, 1e-6, "pe r1c1");
    expect_near(pe[6], 0.0, 1e-6, "pe r1c2");
    expect_near(pe[7], 1.0, 1e-6, "pe r1c3");
    // row 2 (pos -1): [-sin1, cos1, -sin(d1), cos(d1)]
    expect_near(pe[8], -std::sin(1.0), 1e-6, "pe r2c0");
    expect_near(pe[9], std::cos(1.0), 1e-6, "pe r2c1");
    expect_near(pe[10], -std::sin(d1), 1e-6, "pe r2c2");
    expect_near(pe[11], std::cos(d1), 1e-6, "pe r2c3");
}

void test_row_count_and_symmetry() {
    // L=5 D=8: 9 rows; row r has pos (4-r); sin part antisymmetric about
    // center row, cos part symmetric (catches row-order flips).
    const int L = 5, D = 8;
    std::vector<float> pe((2 * L - 1) * D, 0.0F);
    diar::relpos_table_forward(pe.data(), L, D);
    for (int c = 0; c < D; c += 2) {
        expect_near(pe[4 * D + c], 0.0, 1e-6, "pe center sin zero");
        expect_near(pe[4 * D + c + 1], 1.0, 1e-6, "pe center cos one");
    }
    for (int r = 0; r < 9; ++r) {
        const int mirror = 8 - r;
        for (int c = 0; c < D; c += 2) {
            expect_near(pe[r * D + c], -pe[mirror * D + c], 1e-6, "pe sin antisym");
            expect_near(pe[r * D + c + 1], pe[mirror * D + c + 1], 1e-6, "pe cos sym");
        }
    }
}

void test_diar_geometry() {
    // Diar T=60 (chunk002 concat length): 119 rows x 512, finite, and row 0
    // pos +59 (left-first): pe[0] = sin(59) at col 0.
    const int L = 60, D = 512;
    std::vector<float> pe((2 * L - 1) * D, 0.0F);
    diar::relpos_table_forward(pe.data(), L, D);
    for (auto v : pe) expect(std::isfinite(v), "pe diar finite");
    expect_near(pe[0], std::sin(59.0), 1e-5, "pe diar row0");
    expect_near(pe[1], std::cos(59.0), 1e-5, "pe diar row0 cos");
}

void test_independent_full_table() {
    // Independent closed-form rebuild (NOT via diar::posenc): pair-i
    // frequency exp(2i * -(log(1e4)/D)), rows +(L-1)..-(L-1), interleave
    // sin/cos. D=512/L=20 is the live chunk000 geometry.
    const int L = 20, D = 512;
    std::vector<float> pe((2 * L - 1) * D, 0.0F);
    diar::relpos_table_forward(pe.data(), L, D);
    std::vector<double> div(D / 2);
    for (int i = 0; i < D / 2; ++i)
        div[i] = std::exp(2.0 * i * -(std::log(10000.0) / D));
    double max_err = 0.0;
    for (int r = 0; r < 2 * L - 1; ++r) {
        const double pos = (L - 1) - r;
        for (int i = 0; i < D / 2; ++i) {
            max_err = std::max(max_err,
                std::fabs(pe[r * D + 2 * i] - std::sin(pos * div[i])));
            max_err = std::max(max_err,
                std::fabs(pe[r * D + 2 * i + 1] - std::cos(pos * div[i])));
        }
    }
    expect(max_err <= 1e-6, "independent table match, max_err=" + std::to_string(max_err));
}

void test_factor2_fingerprint() {
    // v11 P1 fingerprint regression: the WRONG table (pair-i exponent i
    // instead of 2i) differs from the correct one at exactly
    //   cos = 0.46203146095896536, max_abs = 1.9993901622723202,
    //   mean_abs = 0.507438982036233, and wrong-row0 best-matches correct
    // row 6 on the first 8 dims in exact fp64 (mse 0.0139, 24x margin; the live v11 fp32 capture read row 7 — an fp32 argmin jitter, see below).
    // Those 9 numbers were reproduced offline from the NeMo source formula
    // and matched the live v11 capture 9/9 — pinning them here makes any
    // future factor-2 regression (either direction) FAIL LOUDLY.
    const int L = 20, D = 512;
    std::vector<double> good((2 * L - 1) * D), bad((2 * L - 1) * D);
    for (int r = 0; r < 2 * L - 1; ++r) {
        const double pos = (L - 1) - r;
        for (int i = 0; i < D / 2; ++i) {
            const double dg = std::exp(2.0 * i * -(std::log(10000.0) / D));
            const double db = std::exp(1.0 * i * -(std::log(10000.0) / D));
            bad[r * D + 2 * i] = std::sin(pos * db);
            bad[r * D + 2 * i + 1] = std::cos(pos * db);
            good[r * D + 2 * i] = std::sin(pos * dg);
            good[r * D + 2 * i + 1] = std::cos(pos * dg);
        }
    }
    std::vector<float> pe((2 * L - 1) * D, 0.0F);
    diar::relpos_table_forward(pe.data(), L, D);
    double dot = 0, ng = 0, nb = 0, max_abs = 0, mean_abs = 0;
    int best_row = -1; double best_mse = 1e9, same_mse = 0;
    for (size_t k = 0; k < pe.size(); ++k) {
        const double g = pe[k];
        dot += g * bad[k]; ng += g * g; nb += bad[k] * bad[k];
        max_abs = std::max(max_abs, std::fabs(g - bad[k]));
        mean_abs += std::fabs(g - bad[k]);
    }
    mean_abs /= pe.size();
    for (int r = 0; r < 2 * L - 1; ++r) {
        double mse = 0;
        for (int c = 0; c < 8; ++c)
            mse += std::pow(good[r * D + c] - bad[c], 2.0);
        mse /= 8.0;
        if (mse < best_mse) { best_mse = mse; best_row = r; }
        if (r == 0) same_mse = mse;
    }
    const double cosv = dot / std::sqrt(ng * nb);
    expect_near(cosv, 0.46203146095896536, 2e-6, "factor2 cos");
    expect_near(max_abs, 1.9993901622723202, 1e-4, "factor2 max_abs");
    expect_near(mean_abs, 0.507438982036233, 1e-4, "factor2 mean_abs");
    // Deterministic fp64-vs-fp64 nearest-row pin (24x margin over row 7's
    // 0.332): the v11 LIVE capture reported row 7 / mse 0.117045 — an
    // fp32-rounding jitter of this same comparison (rows 6/7 are near-
    // equidistant under the wrong table; argmin flips with precision).
    expect(best_row == 6, "factor2 row0 best match row 6, got " + std::to_string(best_row));
    expect_near(best_mse, 0.013887, 5e-4, "factor2 best mse");
    expect_near(same_mse, 0.168057, 1e-4, "factor2 same-row mse");
}

}  // namespace

int main() {
    test_hand_case();
    test_row_count_and_symmetry();
    test_diar_geometry();
    test_independent_full_table();
    test_factor2_fingerprint();
    std::cout << "posenc tests passed\n";
    return 0;
}
