// backend_cuda.hpp — the CUDA translation of the backend::Context surface.
// First cut (M3 Step 3): linear via cublasSgemm using the frozen
// zero-pack layout contract (cublas_layout.hpp), pointwise kernels for the
// conformer-FF chain, per-call H2D/D2H (residency is Step 4), blocking.
// Everything unimplemented returns Status::unsupported (CPU fallback path).
//
// Compiled only with DIAR_WITH_CUDA (Kaggle Step 3 kernel / future CI);
// when absent this header is inert and backend.cpp answers Kind::cpu.
#ifndef DIAR_BACKEND_CUDA_HPP_
#define DIAR_BACKEND_CUDA_HPP_

#include "diar/backend.hpp"

#include <cstddef>

#ifdef DIAR_WITH_CUDA
#include <cuda_runtime.h>  // at FILE SCOPE — inside a namespace it would
// drag the whole CUDA runtime API into diar::backend (v4's compile error)
#endif

namespace diar::backend {
namespace cuda {

// Device-side copies of the freeze-now selection-table rows (plan §1).
// Keyed by CC; only rows the fleet actually needs. First cut is fp32-only,
// so the table carries just the bias-add vector width decision.
struct CcConfig {
    int pointwise_block;  // threads per block for elementwise kernels
};
// Returns the frozen row for the running device's CC, or the safe default
// for unknown CCs. Never guesses: unknown CCs get the default row, and the
// bench reports the CC it saw so the table can be extended deliberately.
CcConfig config_for_cc(int major, int minor) noexcept;

}  // namespace cuda

// Defined only when compiled with DIAR_WITH_CUDA (backend_cuda.cpp).
// backend.cpp's factory calls this for Kind::cuda.
Context* make_cuda_context(int cc_major, int cc_minor);

// Bench-harness helpers (Step 3 kernel): device introspection + the empty
// launch used for the launch-overhead measurement.
int bench_device_cc();
int bench_device_sm_count();
const char* bench_device_name();

#ifdef DIAR_WITH_CUDA
void bench_empty_launch(cudaStream_t s, int block);
void bench_ff_residual(float* ff, const float* res, std::size_t n,
                       cudaStream_t s, int block);
#endif

}  // namespace diar

#endif  // DIAR_BACKEND_CUDA_HPP_
