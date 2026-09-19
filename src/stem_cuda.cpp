// stem_cuda.cpp — M3 Step 6b: device-resident pre_encode stem route
// (see stem_cuda.hpp for the contract and layout strategy).
//
// Geometry + mask semantics are the subsampling.cpp pins: k=3 s=2 p=1
// stages, per-stage lengths iterated from feat_len (v13 masked-tail),
// flatten (C,F) c-major f-minor, out Linear to d_model.
#include "diar/stem_cuda.hpp"

#ifdef DIAR_WITH_CUDA

#include "diar/backend_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace diar::backend {
namespace {

void stem_cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CudaStemRoute: ") + what +
                                 ": " + cudaGetErrorString(err));
}
void stem_cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("CudaStemRoute: ") + what +
                                 ": cublasStatus " + std::to_string(int(st)));
}

// torch floor-mode conv length, k=3 s=2 p=1 (subsampling.cpp pin).
int col_len(int in) { return (in + 2 - 3) / 2 + 1; }

// im2col for stage 0: one thread per output position; tap order kh*3+kw
// matches the [out,kh,kw] weight flatten. The input time mask (rows >=
// feat_len) is applied HERE — equivalent to the CPU pre-zeroed mel copy.
__global__ void im2col0_kernel(const float* __restrict__ mel,
                               float* __restrict__ col, int t_mel, int f_in,
                               int w0, int t1, int feat_mask) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = t1 * w0;
    if (p >= n) return;
    const int ho = p / w0, wo = p - ho * w0;
    const int w0in = f_in;
    for (int tap = 0; tap < 9; ++tap) {
        const int kh = tap / 3, kw = tap - kh * 3;
        const int hi = 2 * ho - 1 + kh, wi = 2 * wo - 1 + kw;
        float v = 0.0F;
        if (hi >= 0 && hi < feat_mask && hi < t_mel && wi >= 0 && wi < w0in)
            v = mel[static_cast<std::size_t>(hi) * f_in + wi];
        col[static_cast<std::size_t>(p) * 9 + tap] = v;
    }
}

// Depthwise conv2d k3 s2 p1 on pos-major activations [t*f, C]; channel is
// the fastest index (coalesced), tap order ascending like the CPU loop.
__global__ void dwconv2d_posmajor_kernel(const float* __restrict__ x,
                                         const float* __restrict__ w,
                                         const float* __restrict__ b,
                                         float* __restrict__ y, int t_in,
                                         int f_in, int t_out, int f_out,
                                         int c) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = t_out * f_out * c;
    if (idx >= n) return;
    const int ch = idx % c;
    const int p = idx / c;
    const int ho = p / f_out, wo = p - ho * f_out;
    float acc = b[ch];
    const float* wch = w + static_cast<std::size_t>(ch) * 9;
    for (int tap = 0; tap < 9; ++tap) {
        const int kh = tap / 3, kw = tap - kh * 3;
        const int hi = 2 * ho - 1 + kh, wi = 2 * wo - 1 + kw;
        if (hi < 0 || hi >= t_in || wi < 0 || wi >= f_in) continue;
        acc += x[static_cast<std::size_t>(hi * f_in + wi) * c + ch] * wch[tap];
    }
    y[idx] = acc;
}

// flatten (C,F): flat[t, c*f3 + f] = s2[(t*f3 + f), c] — the torch
// (b,c,t,f).transpose(1,2).reshape(b,t,-1) pin from subsampling.cpp.
__global__ void flatten_cf_kernel(const float* __restrict__ s2,
                                  float* __restrict__ flat, int t3, int c,
                                  int f3) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = t3 * c * f3;
    if (idx >= n) return;
    const int f = idx % f3;
    const int cc = (idx / f3) % c;
    const int t = idx / (f3 * c);
    flat[idx] = s2[static_cast<std::size_t>(t * f3 + f) * c + cc];
}

}  // namespace

struct CudaStemRoute::Impl {
    GpuArena arena;
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;
    int block = 128;

    // weights (device)
    const float *c0_w = nullptr, *c0_b = nullptr;
    const float *dw1_w = nullptr, *dw1_b = nullptr;
    const float *pw1_w = nullptr, *pw1_b = nullptr;
    const float *dw2_w = nullptr, *dw2_b = nullptr;
    const float *pw2_w = nullptr, *pw2_b = nullptr;
    const float *out_w = nullptr, *out_b = nullptr;

    // activations (device, pos-major unless noted)
    float* dmel = nullptr;   // [max_t_mel * feat_in]
    float* im0 = nullptr;    // [t1max * f1 * 9]
    float* s0 = nullptr;     // [t1max * f1 * C]
    float* d1 = nullptr;     // [t2max * f2 * C]
    float* s1 = nullptr;     // [t2max * f2 * C]
    float* d2 = nullptr;     // [t3max * f3 * C]
    float* s2 = nullptr;     // [t3max * f3 * C]
    float* flat = nullptr;   // [t3max * C * f3]
    float* dy = nullptr;     // [t3max * d_model]
};

CudaStemRoute::CudaStemRoute(const SortformerWeights& w, int max_t_mel)
    : max_t_mel_(max_t_mel) {
    if (max_t_mel <= 0) throw std::invalid_argument("CudaStemRoute: max_t_mel <= 0");
    const SortformerConfig& c = w.config();
    feat_in_ = c.feat_in;
    chan_ = c.subsampling_conv_channels;
    d_model_ = c.d_model;
    factor_ = c.subsampling_factor;
    if (feat_in_ != 128 || chan_ != 256 || d_model_ != 512 || factor_ != 8)
        throw std::invalid_argument(
            "CudaStemRoute: geometry pinned to feat_in=128/conv=256/d=512/"
            "factor=8 (got " + std::to_string(feat_in_) + "/" +
            std::to_string(chan_) + "/" + std::to_string(d_model_) + "/" +
            std::to_string(factor_) + ")");

    const SubsamplingWeights& sw = w.stem();
    const int C = chan_;

    impl_ = new Impl();
    Impl& I = *impl_;
    stem_cuda_check(cudaStreamCreate(&I.stream), "cudaStreamCreate");
    stem_cublas_check(cublasCreate(&I.cublas), "cublasCreate");
    stem_cublas_check(cublasSetStream(I.cublas, I.stream), "cublasSetStream");
    stem_cublas_check(cublasSetMathMode(I.cublas, CUBLAS_DEFAULT_MATH),
                      "cublasSetMathMode");

    const int f1 = col_len(feat_in_), f2 = col_len(f1), f3 = col_len(f2);
    const int t1m = col_len(max_t_mel_), t2m = col_len(t1m), t3m = col_len(t2m);

    const std::size_t wts =
        (256u * 9 + 256) * 3 +               // c0 + dw1 + dw2 (with biases)
        (static_cast<std::size_t>(C) * C + C) * 2 +   // pw1 + pw2
        (static_cast<std::size_t>(d_model_) * C * f3 + d_model_);  // out
    const std::size_t acts =
        static_cast<std::size_t>(max_t_mel_) * feat_in_ +
        static_cast<std::size_t>(t1m) * f1 * 9 +
        static_cast<std::size_t>(t1m) * f1 * C * 2 +   // s0 (+mask slack)
        static_cast<std::size_t>(t2m) * f2 * C * 2 +   // d1 + s1
        static_cast<std::size_t>(t3m) * f3 * C * 2 +   // d2 + s2
        static_cast<std::size_t>(t3m) * C * f3 +
        static_cast<std::size_t>(t3m) * d_model_;
    I.arena.init(wts + acts + (1u << 20));

    auto up = [&I](const char* name, const float* src, std::size_t n) {
        float* d = I.arena.alloc(name, n);
        stem_cuda_check(cudaMemcpy(d, src, n * sizeof(float),
                                   cudaMemcpyHostToDevice),
                        "weights H2D");
        return d;
    };
    I.c0_w = up("c0w", sw.c0_w, static_cast<std::size_t>(C) * 9);
    I.c0_b = up("c0b", sw.c0_b, C);
    I.dw1_w = up("dw1w", sw.dw1_w, static_cast<std::size_t>(C) * 9);
    I.dw1_b = up("dw1b", sw.dw1_b, C);
    I.pw1_w = up("pw1w", sw.pw1_w, static_cast<std::size_t>(C) * C);
    I.pw1_b = up("pw1b", sw.pw1_b, C);
    I.dw2_w = up("dw2w", sw.dw2_w, static_cast<std::size_t>(C) * 9);
    I.dw2_b = up("dw2b", sw.dw2_b, C);
    I.pw2_w = up("pw2w", sw.pw2_w, static_cast<std::size_t>(C) * C);
    I.pw2_b = up("pw2b", sw.pw2_b, C);
    I.out_w = up("ow", sw.out_w,
                 static_cast<std::size_t>(d_model_) * C * f3);
    I.out_b = up("ob", sw.out_b, d_model_);

    I.dmel = I.arena.alloc("dmel",
        static_cast<std::size_t>(max_t_mel_) * feat_in_);
    I.im0 = I.arena.alloc("im0", static_cast<std::size_t>(t1m) * f1 * 9);
    I.s0 = I.arena.alloc("s0", static_cast<std::size_t>(t1m) * f1 * C);
    I.d1 = I.arena.alloc("d1", static_cast<std::size_t>(t2m) * f2 * C);
    I.s1 = I.arena.alloc("s1", static_cast<std::size_t>(t2m) * f2 * C);
    I.d2 = I.arena.alloc("d2", static_cast<std::size_t>(t3m) * f3 * C);
    I.s2 = I.arena.alloc("s2", static_cast<std::size_t>(t3m) * f3 * C);
    I.flat = I.arena.alloc("flat", static_cast<std::size_t>(t3m) * C * f3);
    I.dy = I.arena.alloc("dy", static_cast<std::size_t>(t3m) * d_model_);
}

CudaStemRoute::~CudaStemRoute() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.cublas) cublasDestroy(I.cublas);
    if (I.stream) cudaStreamDestroy(I.stream);
    delete impl_;
}

bool CudaStemRoute::stem_forward(const float* mel, int t_mel, int feat_len,
                                 float* y, int t3, const StemRouteConfig& cfg) {
    if (t_mel <= 0 || t_mel > max_t_mel_) {
        refused_++;
        return false;
    }
    if (cfg.feat_in != feat_in_ || cfg.conv_channels != chan_ ||
        cfg.d_model != d_model_ || cfg.subsampling_factor != factor_)
        throw std::invalid_argument(
            "CudaStemRoute: config mismatch (route built from different "
            "weights)");

    Impl& I = *impl_;
    const int C = chan_;
    const int f1 = col_len(feat_in_), f2 = col_len(f1), f3 = col_len(f2);
    const int t1 = col_len(t_mel), t2 = col_len(t1), t3c = col_len(t2);
    if (t3 != t3c)
        throw std::invalid_argument(
            "CudaStemRoute: t3=" + std::to_string(t3) + " but geometry says " +
            std::to_string(t3c) + " for t_mel=" + std::to_string(t_mel));

    const int fl = (feat_len <= 0 || feat_len > t_mel) ? t_mel : feat_len;
    const int l1 = col_len(fl), l2 = col_len(l1), l3 = col_len(l2);

    stem_cuda_check(cudaMemcpyAsync(I.dmel, mel,
                                    static_cast<std::size_t>(t_mel) * feat_in_ *
                                        sizeof(float),
                                    cudaMemcpyHostToDevice, I.stream),
                    "mel H2D");

    // Stage 0: im2col -> GEMM(1->C) -> ReLU -> mask(l1)
    {
        const int n = t1 * f1;
        im2col0_kernel<<<(n + I.block - 1) / I.block, I.block, 0, I.stream>>>(
            I.dmel, I.im0, t_mel, feat_in_, f1, t1, fl);
        stem_cuda_check(cudaGetLastError(), "im2col0 launch");
        dev_linear(I.cublas, I.stream, I.block, I.im0, I.c0_w, I.c0_b, I.s0,
                   n, 9, C, nullptr);
        dev_relu_inplace(I.s0, static_cast<std::size_t>(n) * C, I.stream,
                         I.block);
        if (l1 < t1)
            stem_cuda_check(cudaMemsetAsync(
                                I.s0 + static_cast<std::size_t>(l1) * f1 * C, 0,
                                static_cast<std::size_t>(t1 - l1) * f1 * C *
                                    sizeof(float),
                                I.stream),
                            "mask l1 memset");
    }
    // Stage 1: DW -> PW -> ReLU -> mask(l2)
    {
        const int n1 = t2 * f2;
        dwconv2d_posmajor_kernel<<<(n1 * C + I.block - 1) / I.block, I.block,
                                   0, I.stream>>>(
            I.s0, I.dw1_w, I.dw1_b, I.d1, t1, f1, t2, f2, C);
        stem_cuda_check(cudaGetLastError(), "dw1 launch");
        dev_linear(I.cublas, I.stream, I.block, I.d1, I.pw1_w, I.pw1_b, I.s1,
                   n1, C, C, nullptr);
        dev_relu_inplace(I.s1, static_cast<std::size_t>(n1) * C, I.stream,
                         I.block);
        if (l2 < t2)
            stem_cuda_check(cudaMemsetAsync(
                                I.s1 + static_cast<std::size_t>(l2) * f2 * C, 0,
                                static_cast<std::size_t>(t2 - l2) * f2 * C *
                                    sizeof(float),
                                I.stream),
                            "mask l2 memset");
    }
    // Stage 2: DW -> PW -> ReLU -> mask(l3)
    {
        const int n2 = t3 * f3;
        dwconv2d_posmajor_kernel<<<(n2 * C + I.block - 1) / I.block, I.block,
                                   0, I.stream>>>(
            I.s1, I.dw2_w, I.dw2_b, I.d2, t2, f2, t3, f3, C);
        stem_cuda_check(cudaGetLastError(), "dw2 launch");
        dev_linear(I.cublas, I.stream, I.block, I.d2, I.pw2_w, I.pw2_b, I.s2,
                   n2, C, C, nullptr);
        dev_relu_inplace(I.s2, static_cast<std::size_t>(n2) * C, I.stream,
                         I.block);
        if (l3 < t3)
            stem_cuda_check(cudaMemsetAsync(
                                I.s2 + static_cast<std::size_t>(l3) * f3 * C, 0,
                                static_cast<std::size_t>(t3 - l3) * f3 * C *
                                    sizeof(float),
                                I.stream),
                            "mask l3 memset");
    }
    // Flatten (C,F) + out Linear -> [t3, d_model]
    {
        const int n = t3 * C * f3;
        flatten_cf_kernel<<<(n + I.block - 1) / I.block, I.block, 0,
                            I.stream>>>(I.s2, I.flat, t3, C, f3);
        stem_cuda_check(cudaGetLastError(), "flatten launch");
        dev_linear(I.cublas, I.stream, I.block, I.flat, I.out_w, I.out_b,
                   I.dy, t3, C * f3, d_model_, nullptr);
    }

    stem_cuda_check(cudaMemcpyAsync(y, I.dy,
                                    static_cast<std::size_t>(t3) * d_model_ *
                                        sizeof(float),
                                    cudaMemcpyDeviceToHost, I.stream),
                    "embs D2H");
    stem_cuda_check(cudaStreamSynchronize(I.stream), "stream sync");
    calls_++;
    return true;
}

}  // namespace diar::backend

#endif  // DIAR_WITH_CUDA
