#pragma once

// M2 Stage 1 CPU tensor core: dependency-free FP32 reference operators.
// Upstream pins (NeMo-Speech.cpp a5b6953, NOT assumptions):
//   LayerNorm eps 1e-5            <- src/runtime/ggml/nn.cpp LayerNorm::build_graph
//                                    (ggml_norm(..., 1e-5))
//   BatchNorm1d(infer) eps 1e-5    <- same file BatchNorm1d::set_data (host
//                                    uploads 1e-5; graph is (x-mean)/sqrt(var+eps)*w+b)
//   attention scale 1/sqrt(d_k)    <- rel_pos_attention.cpp + fastconformer.cpp:919
//                                    + transformer_encoder.cpp (NeMo divides q and k
//                                    each by d_k^0.25; single scale is the same product)
//   xscale x*sqrt(d_model)         <- fastconformer.cpp build_graph_from_embeddings
//                                    (NeMo PositionalEncoding.forward)
//   ConformerFF / conv SiLU        <- fastconformer.cpp:236,374 + ConformerConv comment
//   transformer FF / head ReLU     <- transformer_encoder.h comment +
//                                    sortformer_model.cpp build_graph section 4
//                                    (relu->linear->relu->linear->sigmoid)
//   rel-pos shift trick            <- rel_pos_attention.cpp unfused path
//                                    (zero-row pad + reshape/view strip; the kept
//                                    strip is S[k,j] = BD[k-j+T-1,j], proven by
//                                    index simulation over ggml row-major ne=[N0,N1]
//                                    with metadata-only reshapes)
//   GLU gate = second half         <- fastconformer.cpp ConformerConv::build_graph
//                                    (part_a * sigmoid(part_b), a = first half)
//   depthwise symmetric pad        <- fastconformer.cpp ConformerConv ctor
//                                    (pad = (k-1)/2 both sides, zeros; diar uses
//                                    ConvNorm::BatchNorm + ConvContext::Symmetric,
//                                    CacheMode::Disabled -- causal/cache-aware is
//                                    M3 business, explicitly out of scope here)
//   subsampling symmetric          <- fastconformer.cpp SubSampling (k=3 s=2 p=1,
//                                    3 stages => factor 8, ceil-div lengths)
//
// Layout contract (pinned for Stage 2 teacher-forced I/O):
//   - activations are row-major frame-major [T, C]: index t*C + c.
//   - Linear weights are PyTorch layout [out, in]: y[t,o] = sum_i x[t,i]*W[o,i] + b[o].
//     (GGUF stores ggml-transposed copies; the Stage 2 weight loader -- not these
//     ops -- owns that conversion. These ops never see GGUF bytes.)
//   - Conv1d weights [out, in, K]: index ((o*Cin)+i)*K + k, zeros padded symmetrically.
//   - Depthwise weights [C, K]: index o*K + k.
//   - Conv2d tensors [C, H, W] contiguous (((c*H)+h)*W+w); weights
//     [out, in, KH, KW] ((((o*Cin)+i)*KH+kh)*KW+kw). For pre_encode the caller
//     transposes mel_window (T_mel, 128) to (1, 128, T_mel) first.
//   - rel-shift BD is [(2T-1), T] (relpos rows, query cols), output [T, T]
//     (key rows, query cols).
// All math is FP32 (float accumulators). Double precision lives ONLY in the
// test oracle (tests/test_nn.cpp) per the anti-self-certification rule.

#include <cstddef>

namespace diar::nn {

constexpr float kLayerNormEps = 1e-5F;
constexpr float kBatchNormEps = 1e-5F;

void linear_forward(const float* x, const float* weight, const float* bias, float* y,
    std::size_t t, std::size_t in, std::size_t out);

void layernorm_forward(const float* x, const float* gamma, const float* beta, float* y,
    std::size_t t, std::size_t c, float eps = kLayerNormEps);

void batchnorm1d_infer_forward(const float* x, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, float* y, std::size_t t,
    std::size_t c, float eps = kBatchNormEps);

void softmax_last_dim(const float* x, float* y, std::size_t rows, std::size_t cols);

void relu_inplace(float* x, std::size_t n);
void silu_forward(const float* x, float* y, std::size_t n);
void sigmoid_forward(const float* x, float* y, std::size_t n);

void glu_forward(const float* x, float* y, std::size_t t, std::size_t c);

void conv1d_forward(const float* x, const float* weight, const float* bias, float* y, int t_in,
    int c_in, int c_out, int kernel, int stride, int padding);

void conv1d_depthwise_forward(const float* x, const float* weight, const float* bias, float* y,
    int t_in, int channels, int kernel, int stride, int padding);

void conv2d_forward(const float* x, const float* weight, const float* bias, float* y, int c_in,
    int h_in, int w_in, int c_out, int kh, int kw, int stride_h, int stride_w, int pad_h,
    int pad_w);

void rel_shift_forward(const float* bd, float* out, std::size_t t);

void xscale_forward(const float* x, float* y, std::size_t n, float scale);

}  // namespace diar::nn
