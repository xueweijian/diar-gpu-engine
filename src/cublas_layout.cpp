// cublas_layout.cpp — CPU reference implementations of the cuBLAS call
// contract. See the header for the pin story. CUDA TU is backend_cuda.cpp;
// this file compiles everywhere (no CUDA includes) so local CI holds the
// convention vaccine.
#include "diar/cublas_layout.hpp"

#include "diar/gguf.hpp"  // half_to_float (read direction, M2-pinned)

namespace diar {

CublasGemmPlan linear_gemm_plan(int t, int in_features, int out_features) noexcept {
  // C[O,T]_cm = op(A)[O,I] * op(B)[I,T]; A = w (stored row-major [O,I]
  // == col-major [I,O], so op_a = T, lda = I), B = x (stored row-major
  // [T,I] == col-major [I,T], so op_b = N, ldb = I). ld(C) = O, and the
  // C buffer read row-major is exactly y [T,O]. Derived once, tested
  // forever (test_cublas_layout.cpp).
  return CublasGemmPlan{out_features, t, in_features,
                        /*op_a=*/1, /*op_b=*/0,
                        /*lda=*/in_features, /*ldb=*/in_features,
                        /*ldc=*/out_features};
}

void linear_cublas_semantics_ref(const float* x, const float* w,
                                 float* y, int t, int in_features,
                                 int out_features) noexcept {
  // Emulate cuBLAS: C is column-major [O, T] with ldc = O; accumulate
  // over k ascending (matches nn::linear's innermost order => bit-identic
  // float sums). For column-major C, element (o, tt) lives at C[o + tt*O]
  // in memory == y_rowmajor[tt*O + o].
  const int O = out_features, I = in_features;
  for (int tt = 0; tt < t; ++tt) {
    for (int o = 0; o < O; ++o) {
      float acc = 0.0f;
      for (int i = 0; i < I; ++i) {
        // op(A) = T: A_cm element (o, i) from w row-major [O, I]:
        // w_cm[i + o*I] == w[o*I + i]
        // op(B) = N: B_cm element (i, tt) from x row-major [T, I]:
        // x_cm[i + tt*I] == x[tt*I + i]
        acc += w[o * I + i] * x[tt * I + i];
      }
      y[tt * O + o] = acc;
    }
  }
}

std::uint16_t float_to_half(float f) noexcept {
  // IEEE-754 binary16, round-to-nearest-even. Self-contained on purpose
  // (gguf.cpp is an embedded blob). Round-trips with diar::half_to_float.
  std::uint32_t bits = 0;
  __builtin_memcpy(&bits, &f, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exp8 = (bits >> 23) & 0xFFu;
  const std::uint32_t man23 = bits & 0x7FFFFFu;
  if (exp8 == 0xFFu) {  // inf / nan (nan keeps a quiet-bit marker)
    return static_cast<std::uint16_t>(sign | 0x7C00u | (man23 ? 0x200u : 0));
  }
  const int e = static_cast<int>(exp8) - 127;
  if (e >= 16) {  // overflow -> inf
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  if (e >= -14) {  // normal half; the implicit 1 is bit 23 of the float,
    // NOT part of the 10-bit mantissa — round bits come from man23 alone.
    std::uint32_t h = (static_cast<std::uint32_t>(e + 15) << 10) | (man23 >> 13);
    const std::uint32_t rb = man23 & 0x1FFFu;
    if (rb > 0x1000u || (rb == 0x1000u && (h & 1u))) h += 1;
    if ((h & 0x7C00u) == 0x7C00u) {  // carry overflowed to inf
      return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    return static_cast<std::uint16_t>(sign | h);
  }
  if (e >= -25) {  // subnormal half: man10 = RNE(full * 2^(e+1))
    const std::uint32_t full = man23 | 0x800000u;
    const int shift = -e - 1;  // in [10, 24]
    std::uint32_t man10 = full >> shift;
    const std::uint32_t rb = full & ((1u << shift) - 1u);
    const std::uint32_t hp = 1u << (shift - 1);
    if (rb > hp || (rb == hp && (man10 & 1u))) man10 += 1;
    // man10 == 0x400 here encodes the smallest normal — correct as-is.
    return static_cast<std::uint16_t>(sign | man10);
  }
  return static_cast<std::uint16_t>(sign);  // underflow -> signed zero
}

void pack_weights_f16(const float* w_rowmajor_o_i, std::uint16_t* out_half,
                      int out_features, int in_features) noexcept {
  const size_t n = static_cast<size_t>(out_features) *
                   static_cast<size_t>(in_features);
  for (size_t i = 0; i < n; ++i) out_half[i] = float_to_half(w_rowmajor_o_i[i]);
}

void unpack_weights_f16(const std::uint16_t* half, float* out_rowmajor_o_i,
                        int out_features, int in_features) noexcept {
  const size_t n = static_cast<size_t>(out_features) *
                   static_cast<size_t>(in_features);
  for (size_t i = 0; i < n; ++i) out_rowmajor_o_i[i] = half_to_float(half[i]);
}

}  // namespace diar
