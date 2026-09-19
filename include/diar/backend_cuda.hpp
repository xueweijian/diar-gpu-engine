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
#include <cublas_v2.h>     // both at FILE SCOPE — inside a namespace they
#include <cuda_runtime.h>  // would drag the whole CUDA API into diar::backend
// (v4's compile error)
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

// ---- Step 4: device-resident forward wave (MHA + conv + full layer) -------
//
// Design note: the Context:: linear interface copies per call (host->device
// per op), which is the right migration surface for isolated ops but the
// wrong shape for a resident forward chain. The Step 4 entry points below
// take DEVICE pointers everywhere (allocated via GpuArena once at startup)
// and never touch the host — per-chunk steady state has zero H2D traffic.
// They mirror diar::relpos_mha_forward / diar::conformer_layer_forward
// (M2 reference implementations, NeMo-pinned semantics).
//
// GpuArena: one cudaMalloc, bump-allocated named spans. "Weights device
// resident" = all 17 layer weights uploaded once; per-chunk work touches
// only activations + scratch.
struct GpuArena {
    void init(std::size_t n_floats);               // one cudaMalloc
    float* alloc(const char* name, std::size_t n); // register + bump
    float* span(const char* name) const;           // lookup, throws if absent
    ~GpuArena();
   private:
    float* base_ = nullptr;
    std::size_t cap_ = 0, used_ = 0;
    // name registry as parallel vectors (no std::map in device TU for now)
    const char** names_ = nullptr;
    float** ptrs_ = nullptr;
    std::size_t count_ = 0, reg_cap_ = 0;
};

// All pointer fields are DEVICE pointers (PyTorch [out,in] row-major, same
// values as the host-side RelPosMhaWeights / ConformerConvWeights /
// ConformerLayerWeights views).
struct MhaDevWeights {
    const float *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *pos_w, *bu, *bv,
        *out_w, *out_b;
};
struct ConvDevWeights {
    const float *pw1_w, *pw1_b, *dw_w, *dw_b, *bn_w, *bn_b, *bn_mean,
        *bn_var, *pw2_w, *pw2_b;
};
struct LayerDevWeights {
    // 5 LayerNorms (gamma,beta pairs), order ff1/sa/conv/ff2/out
    const float *n_ff1_g, *n_ff1_b, *n_sa_g, *n_sa_b, *n_conv_g, *n_conv_b,
        *n_ff2_g, *n_ff2_b, *n_out_g, *n_out_b;
    // ff1/ff2 bodies
    const float *ff1_w1, *ff1_b1, *ff1_w2, *ff1_b2, *ff2_w1, *ff2_b1,
        *ff2_w2, *ff2_b2;
    MhaDevWeights attn;
    ConvDevWeights conv;
};

// Scratch layout for one MHA call (floats): returned requirement for
// t frames, c dims, h heads. Caller allocates once per (t) and reuses.
std::size_t mha_scratch_floats(int t, int c, int h);

// rel-pos MHA forward, device-resident. x [t,c], pos [2t-1,c], y [t,c];
// ws = scratch (>= mha_scratch_floats). Blocks on stream before return.
void gpu_relpos_mha(cublasHandle_t cublas, cudaStream_t stream, int block,
                    const MhaDevWeights& w, const float* x, const float* pos,
                    float* y, float* ws, int t, int c, int heads);

// Full conformer layer, device-resident. Mirror of
// diar::conformer_layer_forward (pos_emb [2t-1,c] computed once per chunk
// by the host and shared by all 17 layers). ws = scratch (>=
// conformer_layer_scratch_floats). Conv module pinned: k=9, BN, symmetric
// pad 4, pw1 2D->GLU->dw->BN->SiLU->pw2 (conv.hpp upstream pins).
std::size_t conformer_layer_scratch_floats(int t, int c, int d_ff, int h);
void gpu_conformer_layer(cublasHandle_t cublas, cudaStream_t stream,
                         int block, const LayerDevWeights& w,
                         const float* x, const float* pos, float* y,
                         float* ws, int t, int c, int d_ff, int heads);

// ---- Step 4b: transformer block (18-layer stack, plan §3 item 2) ---------
//
// Plain (non-rel-pos, unmasked) MHA + post-LN + ReLU FF. Mirrors
// diar::transformer_block_forward (M2 K1 gate: numpy mirror matched NeMo
// fp32 to 1.1e-6, so the CPU reference is the parity anchor). Post-LN
// order: attn -> +res -> LN1 -> FF -> +res -> LN2. Weight layout PyTorch
// [out,in] exactly as TransformerBlockWeights (device pointers).
struct TransformerDevWeights {
    const float *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    const float *ln1_g, *ln1_b, *ln2_g, *ln2_b;
    const float *f1_w, *f1_b, *f2_w, *f2_b;
};

std::size_t transformer_block_scratch_floats(int t, int h, int inner,
                                             int heads);

// x [t,h] -> y [t,h]; ws >= transformer_block_scratch_floats.
void gpu_transformer_block(cublasHandle_t cublas, cudaStream_t stream,
                           int block, const TransformerDevWeights& w,
                           const float* x, float* y, float* ws, int t, int h,
                           int inner, int heads);

// Raw device-resident linear (same plan as Step 3's Context::linear, no
// host copies): gate-evidence probe for the cuBLAS reduction-noise tier.
void bench_linear(cublasHandle_t cublas, cudaStream_t stream, int block,
                  const float* x, const float* w, const float* b, float* y,
                  int t, int in, int out);

// Isolated-op bench entries (parity vs CPU nn:: reference): same kernels
// the resident layer calls, exposed so the harness can gate them one by one.
void bench_softmax_rows(const float* x, float* y, int rows, int cols,
                        cudaStream_t s, int block);
void bench_glu(const float* x, float* y, int t, int c, cudaStream_t s,
               int block);  // x [t,2c] -> y [t,c]
void bench_dwconv_bn_silu(const float* x, const float* w, const float* b,
                          const float* bn_g, const float* bn_b,
                          const float* mean, const float* var, float* y,
                          int t, int c, int k, cudaStream_t s, int block);
#endif

}  // namespace diar

#endif  // DIAR_BACKEND_CUDA_HPP_
