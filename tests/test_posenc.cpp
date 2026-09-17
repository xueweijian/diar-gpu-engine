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
    // L=2 D=4: positions [+1, 0, -1]. div = [1, exp(-log10000/4)].
    const double d1 = std::exp(-std::log(10000.0) / 4.0);
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

}  // namespace

int main() {
    test_hand_case();
    test_row_count_and_symmetry();
    test_diar_geometry();
    std::cout << "posenc tests passed\n";
    return 0;
}
