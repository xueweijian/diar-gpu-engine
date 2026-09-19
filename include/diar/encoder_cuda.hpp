// encoder_cuda.hpp — M3 Step 6: the device-resident encoder route.
//
// CudaEncoderRoute implements diar::EncoderRoute (sortformer.hpp) with the
// Step-4/5 gated device chain: conformer x N -> encoder_proj -> transformer
// x M, weights uploaded ONCE at construction (GpuArena), one H2D (x, pe)
// and one D2H (px) per chunk, single stream + fixed kernel order —
// determinism by construction (G-C rerun must be bit-identical).
//
// Two storage routes over ONE set of templated layer bodies (backend_cuda):
//   fp32  — cublasSgemm everywhere (the parity anchor route)
//   fp16  — GemmEx 16F-in / 32F-acc / 32F-out, weights RNE-quantized once
//           at upload (Step 5 semantics; opt-in until it passes the K6
//           four-fixture gate through this engine-level path)
// stem/concat/xscale/PE stay on the CPU (profile-exonerated, <1%).
// Step 6b: with_head=true ALSO binds the scoring head (ReLU -> hidden
// Linear -> ReLU -> spks Linear -> sigmoid, diar_head_forward's math) to
// the device chain — encoder_forward_headed D2Hs ONLY preds [L,S],
// skipping the px round-trip entirely (10 ms CPU head + 48x less D2H).
//
// Compiled only with DIAR_WITH_CUDA; CPU builds never see this header.
#ifndef DIAR_ENCODER_CUDA_HPP_
#define DIAR_ENCODER_CUDA_HPP_

#include "diar/sortformer.hpp"

#ifdef DIAR_WITH_CUDA

namespace diar::backend {

class CudaEncoderRoute final : public EncoderRoute {
public:
    // w: the SAME SortformerWeights the engine was built from (values are
    // read at construction only — the engine keeps its own copy).
    // fp16_storage: GemmEx 16F route (see above). max_l: the route serves
    // windows with L <= max_l and refuses larger ones (CPU fallback in
    // sortformer_run_chunk); size it from the stream geometry
    // (spkcache_len + fifo_len + max chunk T3, plus margin).
    // Throws std::runtime_error on any CUDA/cuBLAS failure (fail-loud:
    // a half-initialized route must never answer "true").
    CudaEncoderRoute(const SortformerWeights& w, bool fp16_storage, int max_l,
                     bool with_head = false);
    ~CudaEncoderRoute() override;

    CudaEncoderRoute(const CudaEncoderRoute&) = delete;
    CudaEncoderRoute& operator=(const CudaEncoderRoute&) = delete;

    bool encoder_forward(const float* x, const float* pe, float* px, int L,
        const EncoderRouteConfig& cfg) override;

    // Step 6b fused head (see EncoderRoute in sortformer.hpp).
    bool provides_head() const override { return head_; }
    bool encoder_forward_headed(const float* x, const float* pe, float* preds,
        int L, const EncoderRouteConfig& cfg) override;

    // ---- bench/diagnostic counters (read by diar-bench after a run) ----
    long calls() const { return calls_; }        // forwarded chunks
    long refused() const { return refused_; }    // L > max_l (CPU fallback)
    int max_l() const { return max_l_; }
    bool fp16_storage() const { return fp16_; }
    bool head_bound() const { return head_; }
    long headed_calls() const { return headed_calls_; }
    // Bytes moved per chunk (for the bench JSON's H2D/D2H accounting):
    // x + pe up, px down.
    std::size_t bytes_h2d(int L) const;
    std::size_t bytes_d2h(int L) const;

private:
    struct Impl;
    Impl* impl_ = nullptr;

    // Shared device chain (conformer -> proj -> transformer). Returns the
    // final [L,X] device pointer, or nullptr on budget refusal. Throws on
    // config mismatch / CUDA failure.
    const float* chain_run(const float* x, const float* pe, int L,
        const EncoderRouteConfig& cfg);

    // cfg snapshot taken at construction (validated against the weights).
    int max_l_ = 0;
    bool fp16_ = false, head_ = false;
    long calls_ = 0, refused_ = 0, headed_calls_ = 0;
    int d_ = 0, x_ = 0, enc_layers_ = 0, enc_heads_ = 0, enc_ff_ = 0,
        conv_k_ = 0, tf_layers_ = 0, tf_heads_ = 0, tf_inner_ = 0, nspk_ = 0;
};

}  // namespace diar::backend

#endif  // DIAR_WITH_CUDA
#endif  // DIAR_ENCODER_CUDA_HPP_
