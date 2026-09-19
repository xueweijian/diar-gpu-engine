#pragma once
// diar/stem_cuda.hpp — M3 Step 6b: device-resident pre_encode stem route.
//
// CudaStemRoute implements diar::StemRoute (sortformer.hpp) with the same
// engineering shape as CudaEncoderRoute: all stem weights uploaded ONCE at
// construction into a GpuArena, activations device-resident and reused,
// one H2D (mel window) + one D2H (chunk_embs) per chunk, single stream.
//
// Numerics contract: same math as subsampling_forward (the v13 masked-tail
// semantics included) up to fp32 GEMM reassociation — the Step-6 G-B2 face
// (fp32 engine-with-route vs CPU engine, gate 0.05) gates the residue
// end-to-end; expect ~1e-6 level, never a semantic divergence.
//
// Layout strategy (pos-major activations [N, C]):
//   stage 0  im2col(mel) [N0,9] -> dev_linear(c0) [N0,256] -> relu
//            -> row mask (rows >= l1 zeroed; contiguous in pos-major)
//   stage k  depthwise conv2d k3 s2 p1 (direct kernel, no relu)
//            -> dev_linear(pw) -> relu -> row mask (rows >= l{k+1})
//   flatten  transpose kernel [N2,256] -> [t3, 256*f3] (c-major f-minor,
//            the torch (C,F) reshape pin)
//   out      dev_linear(out_w) [t3, 512]
// The per-stage masks are the zero_time_rows equivalent: in pos-major the
// masked prefix [l*f .. t*f) is ONE contiguous span per channel block only
// in channel-major — in pos-major [N,C] the time rows >= l form the
// contiguous tail of N (positions are t-major), so a plain memsetAsync on
// [(l*f)*C, (t*f)*C) is exact.

#include "diar/sortformer.hpp"

#include <cstddef>

namespace diar {
struct SortformerWeights;
}

namespace diar::backend {

class CudaStemRoute final : public StemRoute {
public:
    // w supplies the SubsamplingWeights (stem()) + config pins. max_t_mel
    // is the largest mel window the route will serve (t_mel beyond it is
    // the only legal refusal). Geometry is pinned to the diar model:
    // feat_in=128, conv_channels=256, d_model=512, factor 8 (a mismatching
    // config throws — never a silent wrong answer).
    CudaStemRoute(const SortformerWeights& w, int max_t_mel);
    ~CudaStemRoute() override;
    CudaStemRoute(const CudaStemRoute&) = delete;
    CudaStemRoute& operator=(const CudaStemRoute&) = delete;

    bool stem_forward(const float* mel, int t_mel, int feat_len, float* y,
                      int t3, const StemRouteConfig& cfg) override;

    long calls() const { return calls_; }
    long refused() const { return refused_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    int max_t_mel_ = 0;
    int feat_in_ = 0, chan_ = 0, d_model_ = 0, factor_ = 0;
    long calls_ = 0, refused_ = 0;
};

}  // namespace diar::backend
