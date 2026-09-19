// test_cublas_layout.cpp — the Step 3 convention vaccine.
//
// Pins, without any CUDA dependency:
//  1. linear_gemm_plan returns the zero-pack tuple (m,n,k,opA,opB,ld*)
//     whose cuBLAS semantics are byte-identical to nn::linear's output.
//  2. linear_cublas_semantics_ref (an index-level emulation of what
//     cuBLAS will do with that plan) is BIT-identical to nn::linear on
//     float — same accumulation order is the whole point.
//  3. float_to_half round-trips exactly through gguf's half_to_float on
//     every finite half, matches hand-computed boundary cases (RNE ties),
//     and handles inf/nan/subnormal/overflow per IEEE-754.
#include "diar/cublas_layout.hpp"

#include "diar/gguf.hpp"
#include "diar/nn.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstring>
#include <random>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

void test_plan_tuple() {
    const auto p = diar::linear_gemm_plan(/*t=*/260, /*in=*/512, /*out=*/2048);
    check(p.m == 2048 && p.n == 260 && p.k == 512, "plan mnk");
    check(p.op_a == 1 && p.op_b == 0, "plan ops (T,N)");
    check(p.lda == 512 && p.ldb == 512 && p.ldc == 2048, "plan ld triple");
    // Non-square sanity: transposed weight store must still map without
    // packing for odd shapes (t not multiple of anything).
    const auto q = diar::linear_gemm_plan(17, 129, 65);
    check(q.m == 65 && q.n == 17 && q.k == 129 && q.ldc == 65, "plan odd shape");
}

void test_semantics_bit_identical_to_nn() {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    struct Shape { int t, in, out; };
    const Shape shapes[] = {
        {1, 8, 4}, {2, 3, 5}, {260, 512, 2048}, {260, 2048, 512},
        {17, 129, 65}, {7, 1, 1}, {33, 64, 256},
    };
    for (const Shape& s : shapes) {
        std::vector<float> x(static_cast<size_t>(s.t) * s.in);
        std::vector<float> w(static_cast<size_t>(s.out) * s.in);
        for (float& v : x) v = dist(rng);
        for (float& v : w) v = dist(rng);
        std::vector<float> y_nn(static_cast<size_t>(s.t) * s.out);
        std::vector<float> y_cb(static_cast<size_t>(s.t) * s.out, 0.0f);
        diar::nn::linear_forward(x.data(), w.data(), nullptr, y_nn.data(),
                                 static_cast<std::size_t>(s.t),
                                 static_cast<std::size_t>(s.in),
                                 static_cast<std::size_t>(s.out));
        diar::linear_cublas_semantics_ref(x.data(), w.data(), y_cb.data(),
                                          s.t, s.in, s.out);
        check(std::memcmp(y_nn.data(), y_cb.data(), y_nn.size() * 4) == 0,
              "cublas-semantics bit-identical to nn::linear");
    }
}

void test_half_round_trip_and_boundaries() {
    // Every finite half must round-trip exactly.
    for (std::uint32_t h = 0; h < 0x10000u; ++h) {
        if (h >= 0x7C00u && h < 0x8000u) continue;  // NaN range
        if (h >= 0xFC00u) continue;                 // NaN range (negative)
        const std::uint16_t hh = static_cast<std::uint16_t>(h);
        const float f = diar::half_to_float(hh);
        const std::uint16_t back = diar::float_to_half(f);
        check(back == hh, "half round-trip exact");
    }
    // Hand-pinned cases (RNE at ties).
    struct Case { float f; std::uint16_t h; };
    const Case cases[] = {
        {0.0f, 0x0000}, {-0.0f, 0x8000},
        {1.0f, 0x3C00}, {-1.0f, 0xBC00},
        {0.5f, 0x3800}, {2.0f, 0x4000},
        {65504.0f, 0x7BFF},                       // max finite half
        {65520.0f, 0x7C00},                       // rounds to inf
        {5.9604644775390625e-8f, 0x0001},         // 2^-24, min subnormal
        {2.9802322387695312e-8f, 0x0000},         // 2^-25, RNE tie -> 0 (even)
        {4.4703483581542969e-8f, 0x0001},         // 1.5*2^-25 = 0.75 units -> nearer 1
        {6.103515625e-5f, 0x0400},                // 2^-14, min normal
        {3.0517578125e-5f, 0x0200},               // 2^-15
        {1.0e10f, 0x7C00},                        // overflow
    };
    for (const Case& c : cases) {
        check(diar::float_to_half(c.f) == c.h, "float_to_half pinned case");
    }
    check(diar::float_to_half(std::numeric_limits<float>::quiet_NaN()) & 0x7E00,
          "nan keeps nan marker");
    const std::uint16_t pinf = diar::float_to_half(
        std::numeric_limits<float>::infinity());
    check(pinf == 0x7C00, "+inf");
    const std::uint16_t ninf = diar::float_to_half(
        -std::numeric_limits<float>::infinity());
    check(ninf == 0xFC00, "-inf");
}

void test_pack_unpack() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    const int O = 65, I = 129;
    std::vector<float> w(static_cast<size_t>(O) * I);
    for (float& v : w) v = dist(rng);
    std::vector<std::uint16_t> packed(w.size());
    diar::pack_weights_f16(w.data(), packed.data(), O, I);
    std::vector<float> back(w.size());
    diar::unpack_weights_f16(packed.data(), back.data(), O, I);
    for (size_t i = 0; i < w.size(); ++i) {
        check(back[i] == diar::half_to_float(packed[i]), "unpack consistent");
        check(std::fabs(back[i] - w[i]) <=
                  4.0e-3f * std::fmax(1.0f, std::fabs(w[i])),
              "pack quantization within half epsilon");
    }
}

}  // namespace

int main() {
    test_plan_tuple();
    test_semantics_bit_identical_to_nn();
    test_half_round_trip_and_boundaries();
    test_pack_unpack();
    if (g_fail == 0) {
        std::printf("cublas_layout: ALL GREEN\n");
        return 0;
    }
    std::printf("cublas_layout: %d FAILURES\n", g_fail);
    return 1;
}
