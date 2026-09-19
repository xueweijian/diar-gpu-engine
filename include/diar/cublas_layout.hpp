// cublas_layout.hpp — the row-major <-> column-major contract between the
// engine's memory layout (everything row-major, M2-pinned) and cuBLAS's
// column-major world. This header is deliberately CPU-only and dependency-
// free so the *convention* is testable locally without a CUDA toolchain;
// the CUDA TU (backend_cuda.cpp) is the only consumer of the real cuBLAS.
//
// The vaccine: an earlier harness transposed its reference copy
// (dd[c*g.m+r] instead of dd[c+r*g.n]) and "passed" — a convention bug of
// exactly this class. The layout math below is pinned by
// tests/test_cublas_layout.cpp with an index-level oracle; if you touch
// anything in this file, that test must stay bit-identical.
//
// M3 Step 3, see docs/M3-CUDA-PLAN.md §3.
#ifndef DIAR_CUBLAS_LAYOUT_HPP_
#define DIAR_CUBLAS_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>

namespace diar {

// Engine-side linear (nn::linear, M2-pinned):
//   x: row-major [T, I]
//   w: row-major [O, I]   (out,in — the upstream PyTorch Linear weight)
//   y: row-major [T, O]   y[t,o] = sum_i x[t,i]*w[o,i]  (+ b[o] handled by
//                         the caller as a separate fused/successive op)
//
// cuBLAS computes C[M,N] = op(A)[M,K] * op(B)[K,N], all column-major.
// With zero packing (pointer reuse, no transposition cost):
//   op(A) = CUBLAS_OP_T on w (ld = I)  -> A[M=O, K=I]
//   op(B) = CUBLAS_OP_N on x (ld = I)  -> B[K=I, N=T]
//   C with ld = O                       -> column-major [O, T], whose
//   memory is byte-identical to row-major [T, O]. No pack, no copy.
//
// These helpers return the (m, n, k, opA, opB, lda, ldb, ldc) tuple as
// plain integers so tests can verify them without linking cuBLAS:
// opA/opB use 0 = N (no-op), 1 = T (transpose) — the numeric values of
// cublasOperation_t.

struct CublasGemmPlan {
  int m, n, k;
  int op_a;    // 0 = N, 1 = T
  int op_b;    // 0 = N, 1 = T
  int lda, ldb, ldc;
};

// Plan the zero-pack cublasSgemm/GemmEx call for one engine linear.
CublasGemmPlan linear_gemm_plan(int t, int in_features, int out_features) noexcept;

// Index-level oracle used by the local test: performs exactly what cuBLAS
// would — C_cm[ldc-major] accumulation over k in ascending order — and
// writes into y as row-major [t, out_features]. Same summation order as
// nn::linear, hence bit-identical on float. Deliberately naive.
void linear_cublas_semantics_ref(const float* x, const float* w,
                                 float* y, int t, int in_features,
                                 int out_features) noexcept;

// float -> IEEE-754 half (round-to-nearest-even; zero/subnormal/inf/nan
// preserved), self-contained so gguf.cpp stays untouched (it is an
// embedded blob — editing it would force a K5/K6 re-embed). Round-trips
// exactly with diar::half_to_float (gguf.hpp), pinned by the local test.
std::uint16_t float_to_half(float f) noexcept;

// fp16-storage packers (Step 5 opt-in).
// Weight store layout for GemmEx 16F: the SAME zero-pack trick works with
// CUDA_R_16F pointers, so pack = plain elementwise float->half on the
// host (once, at load) — no transposition either.
void pack_weights_f16(const float* w_rowmajor_o_i, std::uint16_t* out_half,
                      int out_features, int in_features) noexcept;
void unpack_weights_f16(const std::uint16_t* half, float* out_rowmajor_o_i,
                        int out_features, int in_features) noexcept;

}  // namespace diar

#endif  // DIAR_CUBLAS_LAYOUT_HPP_
