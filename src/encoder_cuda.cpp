// encoder_cuda.cpp — M3 Step 6: device-resident encoder route
// implementation (see encoder_cuda.hpp for the contract).
//
// The layer bodies are the Step-4/5-gated templates from backend_cuda.cpp
// (extern-explicit-instantiated there for both weight routes); this file
// only adds: (a) the real-weights upload path from SortformerWeights, and
// (b) the per-chunk driver (H2D x/pe -> conformer x N -> proj -> transformer
// x M -> D2H px, single stream, fixed order).
#include "diar/encoder_cuda.hpp"

#ifdef DIAR_WITH_CUDA

#include "diar/backend_cuda.hpp"
#include "diar/cublas_layout.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace diar::backend {
namespace {

void enc_cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CudaEncoderRoute: ") + what +
                                 ": " + cudaGetErrorString(err));
}
void enc_cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("CudaEncoderRoute: ") + what +
                                 ": cublasStatus " + std::to_string(int(st)));
}

float* upload_f32(GpuArena& a, const char* name, const float* src,
                  std::size_t n) {
    float* d = a.alloc(name, n);
    enc_cuda_check(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice),
                   "weights H2D");
    return d;
}

// fp16-storage upload: quantize once on host (RNE, the same packer the
// Step-5 kernel gates bit-identity on) and upload halves into the arena.
// rows*cols is the GEMM weight shape [out,in] (pack_weights_f16 contract).
__half* upload_f16(GpuArena& a, const char* name, const float* src, int rows,
                   int cols) {
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    float* d = a.alloc(name, (n + 1) / 2);
    std::vector<std::uint16_t> packed(n);
    pack_weights_f16(src, packed.data(), rows, cols);
    enc_cuda_check(cudaMemcpy(d, packed.data(), n * sizeof(__half),
                              cudaMemcpyHostToDevice),
                   "weights H2D (fp16)");
    return reinterpret_cast<__half*>(d);
}

}  // namespace

struct CudaEncoderRoute::Impl {
    GpuArena arena;
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;
    int block = 128;

    // One of the two weight sets is populated, per the route.
    std::vector<LayerDevWeights> conf;
    std::vector<TransformerDevWeights> tf;
    std::vector<LayerDevWeightsH> conf_h;
    std::vector<TransformerDevWeightsH> tf_h;
    const float* proj_w = nullptr;      // fp32 route
    const __half* proj_w16 = nullptr;   // fp16 route
    const float* proj_b = nullptr;

    // Per-chunk activations (device-resident, reused).
    float *dx = nullptr, *dpe = nullptr;
    float *conf0 = nullptr, *conf1 = nullptr;
    float *dpx = nullptr, *tf0 = nullptr, *tf1 = nullptr;
    float* ws_conf = nullptr;
    float* ws_tf = nullptr;
    __half* proj_cast16 = nullptr;  // fp16 route proj activation cast
};

CudaEncoderRoute::CudaEncoderRoute(const SortformerWeights& w, bool fp16_storage,
                                   int max_l)
    : max_l_(max_l), fp16_(fp16_storage) {
    if (max_l <= 0) throw std::invalid_argument("CudaEncoderRoute: max_l <= 0");
    const SortformerConfig& c = w.config();
    d_ = c.d_model;
    x_ = c.transformer_hidden;
    enc_layers_ = c.encoder_layers;
    enc_heads_ = c.encoder_heads;
    enc_ff_ = c.encoder_d_ff;
    conv_k_ = c.conv_kernel;
    tf_layers_ = c.transformer_layers;
    tf_heads_ = c.transformer_heads;
    tf_inner_ = c.transformer_inner;
    // The device conv body pins k=9 (dwconv_bn_silu call in the layer
    // template). A different kernel must fail here, not run wrong numbers.
    if (conv_k_ != 9)
        throw std::invalid_argument(
            "CudaEncoderRoute: conv_kernel=" + std::to_string(conv_k_) +
            " but the device layer body pins 9");

    impl_ = new Impl();
    Impl& I = *impl_;
    enc_cuda_check(cudaStreamCreate(&I.stream), "cudaStreamCreate");
    enc_cublas_check(cublasCreate(&I.cublas), "cublasCreate");
    enc_cublas_check(cublasSetStream(I.cublas, I.stream), "cublasSetStream");
    enc_cublas_check(cublasSetMathMode(I.cublas, CUBLAS_DEFAULT_MATH),
                     "cublasSetMathMode");

    const int C = d_, F = enc_ff_, X = x_, TI = tf_inner_, N = enc_layers_,
              M = tf_layers_;
    // Arena budget (floats). Weight terms mirror the layer structs; the
    // fp16 route needs less, but oversizing is harmless (one alloc).
    const std::size_t conf_w =
        (10u * C + 4u * F * C + 4u * F + 4u * C + 5u * C * C + 6u * C +
         3u * C * C + static_cast<std::size_t>(C) * conv_k_ + 8u * C);
    const std::size_t tf_w =
        (4u * X * X + 4u * X + 4u * X + static_cast<std::size_t>(TI) * X +
         TI + static_cast<std::size_t>(X) * TI + X);
    const std::size_t ml = static_cast<std::size_t>(max_l_);
    const std::size_t acts = ml * C + (2 * ml - 1) * C + 2 * ml * C +
                             2 * ml * C + 2 * ml * X +
                             conformer_layer_scratch_floats(max_l, C, F,
                                                            enc_heads_) +
                             transformer_block_scratch_floats(max_l, X, TI,
                                                              tf_heads_) +
                             (ml * C + 1) / 2;  // proj cast16 (fp16 route)
    I.arena.init(static_cast<std::size_t>(N) * conf_w +
                 static_cast<std::size_t>(M) * tf_w +
                 static_cast<std::size_t>(X) * C + X + acts + (1u << 20));

    // ---- weights ----
    for (int i = 0; i < N; ++i) {
        const ConformerLayerWeights& lw = w.conformer(i);
        const RelPosMhaWeights& a = lw.attn;
        const ConformerConvWeights& v = lw.conv;
        const std::string tag = "c" + std::to_string(i) + ".";
        if (!fp16_) {
            LayerDevWeights d{};
            const float* up[26] = {
                lw.n_ff1_g, lw.n_ff1_b, lw.n_sa_g, lw.n_sa_b, lw.n_conv_g,
                lw.n_conv_b, lw.n_ff2_g, lw.n_ff2_b, lw.n_out_g, lw.n_out_b,
                lw.ff1_w1, lw.ff1_b1, lw.ff1_w2, lw.ff1_b2, lw.ff2_w1,
                lw.ff2_b1, lw.ff2_w2, lw.ff2_b2};
            d.n_ff1_g = upload_f32(I.arena, (tag + "n1g").c_str(), up[0], C);
            d.n_ff1_b = upload_f32(I.arena, (tag + "n1b").c_str(), up[1], C);
            d.n_sa_g = upload_f32(I.arena, (tag + "nsg").c_str(), up[2], C);
            d.n_sa_b = upload_f32(I.arena, (tag + "nsb").c_str(), up[3], C);
            d.n_conv_g = upload_f32(I.arena, (tag + "ncg").c_str(), up[4], C);
            d.n_conv_b = upload_f32(I.arena, (tag + "ncb").c_str(), up[5], C);
            d.n_ff2_g = upload_f32(I.arena, (tag + "n2g").c_str(), up[6], C);
            d.n_ff2_b = upload_f32(I.arena, (tag + "n2b").c_str(), up[7], C);
            d.n_out_g = upload_f32(I.arena, (tag + "nog").c_str(), up[8], C);
            d.n_out_b = upload_f32(I.arena, (tag + "nob").c_str(), up[9], C);
            d.ff1_w1 = upload_f32(I.arena, (tag + "f1w1").c_str(), up[10],
                                  static_cast<std::size_t>(F) * C);
            d.ff1_b1 = upload_f32(I.arena, (tag + "f1b1").c_str(), up[11], F);
            d.ff1_w2 = upload_f32(I.arena, (tag + "f1w2").c_str(), up[12],
                                  static_cast<std::size_t>(C) * F);
            d.ff1_b2 = upload_f32(I.arena, (tag + "f1b2").c_str(), up[13], C);
            d.ff2_w1 = upload_f32(I.arena, (tag + "f2w1").c_str(), up[14],
                                  static_cast<std::size_t>(F) * C);
            d.ff2_b1 = upload_f32(I.arena, (tag + "f2b1").c_str(), up[15], F);
            d.ff2_w2 = upload_f32(I.arena, (tag + "f2w2").c_str(), up[16],
                                  static_cast<std::size_t>(C) * F);
            d.ff2_b2 = upload_f32(I.arena, (tag + "f2b2").c_str(), up[17], C);
            d.attn.q_w = upload_f32(I.arena, (tag + "aq").c_str(), a.q_w,
                                    static_cast<std::size_t>(C) * C);
            d.attn.q_b = upload_f32(I.arena, (tag + "aqb").c_str(), a.q_b, C);
            d.attn.k_w = upload_f32(I.arena, (tag + "ak").c_str(), a.k_w,
                                    static_cast<std::size_t>(C) * C);
            d.attn.k_b = upload_f32(I.arena, (tag + "akb").c_str(), a.k_b, C);
            d.attn.v_w = upload_f32(I.arena, (tag + "av").c_str(), a.v_w,
                                    static_cast<std::size_t>(C) * C);
            d.attn.v_b = upload_f32(I.arena, (tag + "avb").c_str(), a.v_b, C);
            d.attn.pos_w = upload_f32(I.arena, (tag + "ap").c_str(), a.pos_w,
                                      static_cast<std::size_t>(C) * C);
            d.attn.bu = upload_f32(I.arena, (tag + "abu").c_str(), a.bu, C);
            d.attn.bv = upload_f32(I.arena, (tag + "abv").c_str(), a.bv, C);
            d.attn.out_w = upload_f32(I.arena, (tag + "ao").c_str(), a.out_w,
                                      static_cast<std::size_t>(C) * C);
            d.attn.out_b = upload_f32(I.arena, (tag + "aob").c_str(), a.out_b,
                                      C);
            d.conv.pw1_w = upload_f32(I.arena, (tag + "vp1w").c_str(), v.pw1_w,
                                      static_cast<std::size_t>(2 * C) * C);
            d.conv.pw1_b = upload_f32(I.arena, (tag + "vp1b").c_str(), v.pw1_b,
                                      2 * C);
            d.conv.dw_w = upload_f32(I.arena, (tag + "vdw").c_str(), v.dw_w,
                                     static_cast<std::size_t>(C) * conv_k_);
            d.conv.dw_b = upload_f32(I.arena, (tag + "vdb").c_str(), v.dw_b,
                                     C);
            d.conv.bn_w = upload_f32(I.arena, (tag + "vbnw").c_str(), v.bn_w,
                                     C);
            d.conv.bn_b = upload_f32(I.arena, (tag + "vbnb").c_str(), v.bn_b,
                                     C);
            d.conv.bn_mean = upload_f32(I.arena, (tag + "vbnm").c_str(),
                                        v.bn_mean, C);
            d.conv.bn_var = upload_f32(I.arena, (tag + "vbnv").c_str(),
                                       v.bn_var, C);
            d.conv.pw2_w = upload_f32(I.arena, (tag + "vp2w").c_str(), v.pw2_w,
                                      static_cast<std::size_t>(C) * C);
            d.conv.pw2_b = upload_f32(I.arena, (tag + "vp2b").c_str(), v.pw2_b,
                                      C);
            I.conf.push_back(d);
        } else {
            LayerDevWeightsH d{};
            d.n_ff1_g = upload_f32(I.arena, (tag + "n1g").c_str(), lw.n_ff1_g, C);
            d.n_ff1_b = upload_f32(I.arena, (tag + "n1b").c_str(), lw.n_ff1_b, C);
            d.n_sa_g = upload_f32(I.arena, (tag + "nsg").c_str(), lw.n_sa_g, C);
            d.n_sa_b = upload_f32(I.arena, (tag + "nsb").c_str(), lw.n_sa_b, C);
            d.n_conv_g = upload_f32(I.arena, (tag + "ncg").c_str(), lw.n_conv_g, C);
            d.n_conv_b = upload_f32(I.arena, (tag + "ncb").c_str(), lw.n_conv_b, C);
            d.n_ff2_g = upload_f32(I.arena, (tag + "n2g").c_str(), lw.n_ff2_g, C);
            d.n_ff2_b = upload_f32(I.arena, (tag + "n2b").c_str(), lw.n_ff2_b, C);
            d.n_out_g = upload_f32(I.arena, (tag + "nog").c_str(), lw.n_out_g, C);
            d.n_out_b = upload_f32(I.arena, (tag + "nob").c_str(), lw.n_out_b, C);
            d.ff1_b1 = upload_f32(I.arena, (tag + "f1b1").c_str(), lw.ff1_b1, F);
            d.ff1_b2 = upload_f32(I.arena, (tag + "f1b2").c_str(), lw.ff1_b2, C);
            d.ff2_b1 = upload_f32(I.arena, (tag + "f2b1").c_str(), lw.ff2_b1, F);
            d.ff2_b2 = upload_f32(I.arena, (tag + "f2b2").c_str(), lw.ff2_b2, C);
            d.ff1_w1 = upload_f16(I.arena, (tag + "f1w1").c_str(), lw.ff1_w1, F, C);
            d.ff1_w2 = upload_f16(I.arena, (tag + "f1w2").c_str(), lw.ff1_w2, C, F);
            d.ff2_w1 = upload_f16(I.arena, (tag + "f2w1").c_str(), lw.ff2_w1, F, C);
            d.ff2_w2 = upload_f16(I.arena, (tag + "f2w2").c_str(), lw.ff2_w2, C, F);
            d.attn.q_w = upload_f16(I.arena, (tag + "aq").c_str(), a.q_w, C, C);
            d.attn.q_b = upload_f32(I.arena, (tag + "aqb").c_str(), a.q_b, C);
            d.attn.k_w = upload_f16(I.arena, (tag + "ak").c_str(), a.k_w, C, C);
            d.attn.k_b = upload_f32(I.arena, (tag + "akb").c_str(), a.k_b, C);
            d.attn.v_w = upload_f16(I.arena, (tag + "av").c_str(), a.v_w, C, C);
            d.attn.v_b = upload_f32(I.arena, (tag + "avb").c_str(), a.v_b, C);
            d.attn.pos_w = upload_f16(I.arena, (tag + "ap").c_str(), a.pos_w, C, C);
            d.attn.bu = upload_f32(I.arena, (tag + "abu").c_str(), a.bu, C);
            d.attn.bv = upload_f32(I.arena, (tag + "abv").c_str(), a.bv, C);
            d.attn.out_w = upload_f16(I.arena, (tag + "ao").c_str(), a.out_w, C, C);
            d.attn.out_b = upload_f32(I.arena, (tag + "aob").c_str(), a.out_b,
                                      C);
            d.conv.pw1_w = upload_f16(I.arena, (tag + "vp1w").c_str(), v.pw1_w, 2 * C, C);
            d.conv.pw1_b = upload_f32(I.arena, (tag + "vp1b").c_str(), v.pw1_b,
                                      2 * C);
            d.conv.dw_w = upload_f32(I.arena, (tag + "vdw").c_str(), v.dw_w,
                                     static_cast<std::size_t>(C) * conv_k_);
            d.conv.dw_b = upload_f32(I.arena, (tag + "vdb").c_str(), v.dw_b,
                                     C);
            d.conv.bn_w = upload_f32(I.arena, (tag + "vbnw").c_str(), v.bn_w,
                                     C);
            d.conv.bn_b = upload_f32(I.arena, (tag + "vbnb").c_str(), v.bn_b,
                                     C);
            d.conv.bn_mean = upload_f32(I.arena, (tag + "vbnm").c_str(),
                                        v.bn_mean, C);
            d.conv.bn_var = upload_f32(I.arena, (tag + "vbnv").c_str(),
                                       v.bn_var, C);
            d.conv.pw2_w = upload_f16(I.arena, (tag + "vp2w").c_str(), v.pw2_w, C, C);
            d.conv.pw2_b = upload_f32(I.arena, (tag + "vp2b").c_str(), v.pw2_b,
                                      C);
            I.conf_h.push_back(d);
        }
    }
    for (int i = 0; i < M; ++i) {
        const TransformerBlockWeights& tw = w.transformer(i);
        const std::string tag = "t" + std::to_string(i) + ".";
        if (!fp16_) {
            TransformerDevWeights d{};
            d.q_w = upload_f32(I.arena, (tag + "q").c_str(), tw.q_w,
                               static_cast<std::size_t>(X) * X);
            d.q_b = upload_f32(I.arena, (tag + "qb").c_str(), tw.q_b, X);
            d.k_w = upload_f32(I.arena, (tag + "k").c_str(), tw.k_w,
                               static_cast<std::size_t>(X) * X);
            d.k_b = upload_f32(I.arena, (tag + "kb").c_str(), tw.k_b, X);
            d.v_w = upload_f32(I.arena, (tag + "v").c_str(), tw.v_w,
                               static_cast<std::size_t>(X) * X);
            d.v_b = upload_f32(I.arena, (tag + "vb").c_str(), tw.v_b, X);
            d.o_w = upload_f32(I.arena, (tag + "o").c_str(), tw.o_w,
                               static_cast<std::size_t>(X) * X);
            d.o_b = upload_f32(I.arena, (tag + "ob").c_str(), tw.o_b, X);
            d.ln1_g = upload_f32(I.arena, (tag + "l1g").c_str(), tw.ln1_g, X);
            d.ln1_b = upload_f32(I.arena, (tag + "l1b").c_str(), tw.ln1_b, X);
            d.ln2_g = upload_f32(I.arena, (tag + "l2g").c_str(), tw.ln2_g, X);
            d.ln2_b = upload_f32(I.arena, (tag + "l2b").c_str(), tw.ln2_b, X);
            d.f1_w = upload_f32(I.arena, (tag + "f1").c_str(), tw.f1_w,
                                static_cast<std::size_t>(TI) * X);
            d.f1_b = upload_f32(I.arena, (tag + "f1b").c_str(), tw.f1_b, TI);
            d.f2_w = upload_f32(I.arena, (tag + "f2").c_str(), tw.f2_w,
                                static_cast<std::size_t>(X) * TI);
            d.f2_b = upload_f32(I.arena, (tag + "f2b").c_str(), tw.f2_b, X);
            I.tf.push_back(d);
        } else {
            TransformerDevWeightsH d{};
            d.q_w = upload_f16(I.arena, (tag + "q").c_str(), tw.q_w, X, X);
            d.q_b = upload_f32(I.arena, (tag + "qb").c_str(), tw.q_b, X);
            d.k_w = upload_f16(I.arena, (tag + "k").c_str(), tw.k_w, X, X);
            d.k_b = upload_f32(I.arena, (tag + "kb").c_str(), tw.k_b, X);
            d.v_w = upload_f16(I.arena, (tag + "v").c_str(), tw.v_w, X, X);
            d.v_b = upload_f32(I.arena, (tag + "vb").c_str(), tw.v_b, X);
            d.o_w = upload_f16(I.arena, (tag + "o").c_str(), tw.o_w, X, X);
            d.o_b = upload_f32(I.arena, (tag + "ob").c_str(), tw.o_b, X);
            d.ln1_g = upload_f32(I.arena, (tag + "l1g").c_str(), tw.ln1_g, X);
            d.ln1_b = upload_f32(I.arena, (tag + "l1b").c_str(), tw.ln1_b, X);
            d.ln2_g = upload_f32(I.arena, (tag + "l2g").c_str(), tw.ln2_g, X);
            d.ln2_b = upload_f32(I.arena, (tag + "l2b").c_str(), tw.ln2_b, X);
            d.f1_w = upload_f16(I.arena, (tag + "f1").c_str(), tw.f1_w, TI, X);
            d.f1_b = upload_f32(I.arena, (tag + "f1b").c_str(), tw.f1_b, TI);
            d.f2_w = upload_f16(I.arena, (tag + "f2").c_str(), tw.f2_w, X, TI);
            d.f2_b = upload_f32(I.arena, (tag + "f2b").c_str(), tw.f2_b, X);
            I.tf_h.push_back(d);
        }
    }
    I.proj_b = upload_f32(I.arena, "pjb", w.proj_b(), X);
    if (fp16_)
        I.proj_w16 = upload_f16(I.arena, "pjw", w.proj_w(), X, C);
    else
        I.proj_w = upload_f32(I.arena, "pjw", w.proj_w(),
                              static_cast<std::size_t>(X) * C);

    // ---- activations ----
    const std::size_t lc = static_cast<std::size_t>(max_l_) * C;
    const std::size_t lx = static_cast<std::size_t>(max_l_) * X;
    I.dx = I.arena.alloc("dx", lc);
    I.dpe = I.arena.alloc("dpe", 2 * lc - C);
    I.conf0 = I.arena.alloc("cf0", lc);
    I.conf1 = I.arena.alloc("cf1", lc);
    I.dpx = I.arena.alloc("dpx", lx);
    I.tf0 = I.arena.alloc("tf0", lx);
    I.tf1 = I.arena.alloc("tf1", lx);
    I.ws_conf = I.arena.alloc(
        "wsc", conformer_layer_scratch_floats(max_l_, C, F, enc_heads_));
    I.ws_tf = I.arena.alloc(
        "wst", transformer_block_scratch_floats(max_l_, X, TI, tf_heads_));
    I.proj_cast16 = reinterpret_cast<__half*>(I.arena.alloc("pjc", (lc + 1) / 2));
}

CudaEncoderRoute::~CudaEncoderRoute() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.cublas) cublasDestroy(I.cublas);
    if (I.stream) cudaStreamDestroy(I.stream);
    delete impl_;
}

std::size_t CudaEncoderRoute::bytes_h2d(int L) const {
    const std::size_t d = static_cast<std::size_t>(L) * d_;
    return (d + (2 * static_cast<std::size_t>(L) - 1) * d_) * sizeof(float);
}
std::size_t CudaEncoderRoute::bytes_d2h(int L) const {
    return static_cast<std::size_t>(L) * x_ * sizeof(float);
}

bool CudaEncoderRoute::encoder_forward(const float* x, const float* pe,
                                       float* px, int L,
                                       const EncoderRouteConfig& cfg) {
    if (L > max_l_) {
        refused_++;
        return false;  // CPU fallback in sortformer_run_chunk
    }
    if (cfg.d_model != d_ || cfg.encoder_layers != enc_layers_ ||
        cfg.encoder_heads != enc_heads_ || cfg.encoder_d_ff != enc_ff_ ||
        cfg.conv_kernel != conv_k_ || cfg.transformer_layers != tf_layers_ ||
        cfg.transformer_hidden != x_ || cfg.transformer_inner != tf_inner_ ||
        cfg.transformer_heads != tf_heads_)
        throw std::invalid_argument(
            "CudaEncoderRoute: config mismatch (route built from different "
            "weights)");
    if (L <= 0) return false;

    Impl& I = *impl_;
    const int C = d_, X = x_;
    const std::size_t lc = static_cast<std::size_t>(L) * C;
    const std::size_t lx = static_cast<std::size_t>(L) * X;

    enc_cuda_check(cudaMemcpyAsync(I.dx, x, lc * sizeof(float),
                                   cudaMemcpyHostToDevice, I.stream),
                   "x H2D");
    enc_cuda_check(cudaMemcpyAsync(I.dpe, pe,
                                   (2 * lc - C) * sizeof(float),
                                   cudaMemcpyHostToDevice, I.stream),
                   "pe H2D");

    // conformer x N (ping-pong; odd N -> output lands in conf0)
    const float* cur = I.dx;
    float* nxt = I.conf0;
    for (int i = 0; i < enc_layers_; ++i) {
        if (fp16_)
            gpu_conformer_layer(I.cublas, I.stream, I.block, I.conf_h[i],
                                cur, I.dpe, nxt, I.ws_conf, L, C, enc_ff_,
                                enc_heads_);
        else
            gpu_conformer_layer(I.cublas, I.stream, I.block, I.conf[i], cur,
                                I.dpe, nxt, I.ws_conf, L, C, enc_ff_,
                                enc_heads_);
        const float* prev = cur;
        cur = nxt;
        nxt = const_cast<float*>(prev) == I.dx
                  ? I.conf1
                  : const_cast<float*>(prev);
    }

    // encoder_proj D -> X (doubles as the tf chain's layer-0 input)
    if (fp16_)
        dev_linear(I.cublas, I.stream, I.block, cur, I.proj_w16, I.proj_b,
                   I.dpx, L, C, X, I.proj_cast16);
    else
        dev_linear(I.cublas, I.stream, I.block, cur, I.proj_w, I.proj_b,
                   I.dpx, L, C, X, I.proj_cast16);

    // transformer x M (ping-pong between tf0/tf1; dpx is the layer-0 input)
    const float* cur2 = I.dpx;
    float* nxt2 = I.tf0;
    for (int i = 0; i < tf_layers_; ++i) {
        if (fp16_)
            gpu_transformer_block(I.cublas, I.stream, I.block, I.tf_h[i],
                                  cur2, nxt2, I.ws_tf, L, X, tf_inner_,
                                  tf_heads_);
        else
            gpu_transformer_block(I.cublas, I.stream, I.block, I.tf[i], cur2,
                                  nxt2, I.ws_tf, L, X, tf_inner_, tf_heads_);
        const float* prev = cur2;
        cur2 = nxt2;
        nxt2 = const_cast<float*>(prev) == I.dpx
                   ? I.tf1
                   : const_cast<float*>(prev);
    }

    enc_cuda_check(cudaMemcpyAsync(px, cur2, lx * sizeof(float),
                                   cudaMemcpyDeviceToHost, I.stream),
                   "px D2H");
    enc_cuda_check(cudaStreamSynchronize(I.stream), "stream sync");
    calls_++;
    return true;
}

}  // namespace diar::backend

#endif  // DIAR_WITH_CUDA
