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
#include <cuda_fp16.h>     // (v4's compile error). fp16 storage route (Step 5).
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

// ---- Step 6b: shared pointwise entry points (stem route + fused head) ----
// Thin exported wrappers over the file-local kernels so encoder/stem route
// files can reuse the exact same device code as the layer bodies.
void dev_relu_inplace(float* x, std::size_t n, cudaStream_t s, int block);
void dev_sigmoid(const float* x, float* y, std::size_t n, cudaStream_t s,
                 int block);

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

// ---- Step 5: fp16 weight storage (opt-in GemmEx 16F-in / 32F-acc route) ---
//
// H-suffix structs mirror the fp32 ones field-for-field with the SAME field
// names, except every GEMM weight (*_w) is __half. Biases, LayerNorm and BN
// parameters stay fp32 (they feed pointwise kernels, not GEMMs). The layer
// bodies below are templates over the weights struct type, so the fp32 and
// fp16 routes share ONE source of truth — no drift between paths.
//
// Semantics: weights are quantized fp32->fp16 (RNE) once at upload
// (cublas_layout.hpp float_to_half + pack_weights_f16); each GEMM input
// activation is cast fp32->fp16 on device right before cublasGemmEx
// (computeType CUBLAS_COMPUTE_32F, C output fp32 — accumulation stays
// fp32 on every card: P100 runs it on fp16-capable CUDA cores, T4/V100
// may dispatch HMMA with fp32 accumulate; the K6 four-fixture gate is
// the arbiter for whether this route ships as default).
struct MhaDevWeightsH {
    const __half *q_w, *k_w, *v_w, *pos_w, *out_w;  // GEMM weights
    const float *q_b, *k_b, *v_b, *bu, *bv, *out_b;
};
struct ConvDevWeightsH {
    const __half *pw1_w, *pw2_w;
    const float *pw1_b, *pw2_b, *dw_w, *dw_b, *bn_w, *bn_b, *bn_mean,
        *bn_var;
};
struct LayerDevWeightsH {
    const float *n_ff1_g, *n_ff1_b, *n_sa_g, *n_sa_b, *n_conv_g, *n_conv_b,
        *n_ff2_g, *n_ff2_b, *n_out_g, *n_out_b;
    const float *ff1_b1, *ff1_b2, *ff2_b1, *ff2_b2;
    const __half *ff1_w1, *ff1_w2, *ff2_w1, *ff2_w2;
    MhaDevWeightsH attn;
    ConvDevWeightsH conv;
};
struct TransformerDevWeightsH {
    const __half *q_w, *k_w, *v_w, *o_w, *f1_w, *f2_w;
    const float *q_b, *k_b, *v_b, *o_b;
    const float *ln1_g, *ln1_b, *ln2_g, *ln2_b;
    const float *f1_b, *f2_b;
};

// Scratch layout for one MHA call (floats): returned requirement for
// t frames, c dims, h heads. Caller allocates once per (t) and reuses.
// Includes the cast16 region (fp16 route activation casts, unused by the
// fp32 route) so both routes can share one allocation.
std::size_t mha_scratch_floats(int t, int c, int h);

// rel-pos MHA forward, device-resident. x [t,c], pos [2t-1,c], y [t,c];
// ws = scratch (>= mha_scratch_floats). Blocks on stream before return.
// Template over the weights struct: MhaDevWeights = fp32 Sgemm route,
// MhaDevWeightsH = fp16-storage GemmEx route (Step 5). Both instantiations
// are emitted in backend_cuda.cpp.
template <typename W>
void gpu_relpos_mha(cublasHandle_t cublas, cudaStream_t stream, int block,
                    const W& w, const float* x, const float* pos,
                    float* y, float* ws, int t, int c, int heads);
extern template void gpu_relpos_mha<MhaDevWeights>(
    cublasHandle_t, cudaStream_t, int, const MhaDevWeights&, const float*,
    const float*, float*, float*, int, int, int);
extern template void gpu_relpos_mha<MhaDevWeightsH>(
    cublasHandle_t, cudaStream_t, int, const MhaDevWeightsH&, const float*,
    const float*, float*, float*, int, int, int);

// Full conformer layer, device-resident. Mirror of
// diar::conformer_layer_forward (pos_emb [2t-1,c] computed once per chunk
// by the host and shared by all 17 layers). ws = scratch (>=
// conformer_layer_scratch_floats). Conv module pinned: k=9, BN, symmetric
// pad 4, pw1 2D->GLU->dw->BN->SiLU->pw2 (conv.hpp upstream pins).
// Template over the weights struct (fp32 / fp16-storage routes, Step 5).
std::size_t conformer_layer_scratch_floats(int t, int c, int d_ff, int h);
template <typename W>
void gpu_conformer_layer(cublasHandle_t cublas, cudaStream_t stream,
                         int block, const W& w,
                         const float* x, const float* pos, float* y,
                         float* ws, int t, int c, int d_ff, int heads);
extern template void gpu_conformer_layer<LayerDevWeights>(
    cublasHandle_t, cudaStream_t, int, const LayerDevWeights&, const float*,
    const float*, float*, float*, int, int, int, int);
extern template void gpu_conformer_layer<LayerDevWeightsH>(
    cublasHandle_t, cudaStream_t, int, const LayerDevWeightsH&, const float*,
    const float*, float*, float*, int, int, int, int);

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
// Template over the weights struct (fp32 / fp16-storage routes, Step 5).
template <typename W>
void gpu_transformer_block(cublasHandle_t cublas, cudaStream_t stream,
                           int block, const W& w,
                           const float* x, float* y, float* ws, int t, int h,
                           int inner, int heads);
extern template void gpu_transformer_block<TransformerDevWeights>(
    cublasHandle_t, cudaStream_t, int, const TransformerDevWeights&,
    const float*, float*, float*, int, int, int, int);
extern template void gpu_transformer_block<TransformerDevWeightsH>(
    cublasHandle_t, cudaStream_t, int, const TransformerDevWeightsH&,
    const float*, float*, float*, int, int, int, int);

// Raw device-resident linear (same plan as Step 3's Context::linear, no
// host copies): gate-evidence probe for the cuBLAS reduction-noise tier.
void bench_linear(cublasHandle_t cublas, cudaStream_t stream, int block,
                  const float* x, const float* w, const float* b, float* y,
                  int t, int in, int out);

// device-resident linear, exported for encoder_cuda.cpp (Step 6 encoder
// proj GEMM — the same body the resident conformer/transformer layers
// call; fp32 Sgemm route, or fp16-storage GemmEx with the activation cast
// into cast16, cast16 >= t*in halves).
//
// Template over the weight storage type: float = fp32 Sgemm, __half =
// fp16-storage GemmEx. Explicit instantiations live in backend_cuda.cpp
// (extern template: other TUs reference the same symbols — a NON-template
// declaration here would shadow them and leave every call site unresolved
// at link; that was the first CI link failure of Step 6).
template <typename WT>
void dev_linear(cublasHandle_t cublas, cudaStream_t stream, int block,
                const float* x, const WT* w, const float* bias, float* y,
                int t, int in, int out, __half* cast16);
extern template void dev_linear<float>(cublasHandle_t, cudaStream_t, int,
    const float*, const float*, const float*, float*, int, int, int, __half*);
extern template void dev_linear<__half>(cublasHandle_t, cudaStream_t, int,
    const float*, const __half*, const float*, float*, int, int, int, __half*);

// ---- Step 5 bench entries (fp16-storage route) ----------------------------
// fp16-weights linear: w is a DEVICE __half array (host-packed via
// pack_weights_f16), x stays fp32 and is cast into cast16 (DEVICE scratch
// >= t*in halves, carved from the layer scratch cast16 region).
void bench_linear_h(cublasHandle_t cublas, cudaStream_t stream, int block,
                    const float* x, const __half* w, const float* b,
                    float* y, int t, int in, int out, __half* cast16);
// device fp32->fp16 cast alone (bit-identity gate vs host float_to_half).
void bench_cast_f32_f16(const float* x, __half* y, std::size_t n,
                        cudaStream_t s, int block);

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
