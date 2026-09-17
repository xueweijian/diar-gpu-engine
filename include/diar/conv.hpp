#pragma once

// M2 Stage 2 Conformer convolution module (non-cached / offline path only).
// Reference FP32 implementation; inference semantics (dropouts eval
// identities, pad_mask=None in the offline teacher-forced gates).
//
// Upstream pins (NOT assumptions):
//   NeMo torch ConformerConvolution.forward (conformer_modules.py @ main):
//     x = pointwise_conv1 applied as Linear over last dim (weight
//         [2D,D].squeeze(-1)); x = glu(x, dim=-1) [the 'glu_' default];
//         x = x.transpose(1,2) [channel-first (B,D,T)];
//         x = depthwise_conv(x) [CausalConv1D, groups=D, k=9, symmetric
//         context (k-1)/2 each side];
//         batch_norm over channels [BatchNorm1d(D), running stats, eps 1e-5];
//         x = x.transpose(1,2); x = Swish()(x) [== SiLU];
//         x = pointwise_conv2 as Linear. NOTE the norm sits BEFORE the
//         transpose-back, i.e. over (B,D,T) channel dim — identical values
//         to a [T,D] frame-major BN over dim 1 (this ref calls the frame
//         version directly, no transpose needed).
//   C++ ggml port ConformerConv ctor + build_graph + build_post_glu
//     (fastconformer.cpp): pointwise k=1 convs, GLU part_a*sigmoid(part_b)
//     with a = FIRST half (matches torch glu dim=-1 split), depthwise pad
//     (k-1)/2 symmetric, BatchNorm then SiLU then pointwise_conv2. This ref
//     follows the torch order (BN -> transpose -> SiLU -> Linear), which is
//     value-identical to the ggml order (BN -> permute -> SiLU -> mul_mat).
//
// Diar config (sortformer GGUF, parse_config in sortformer_model.cpp +
// plan §1): d_model=512, conv_kernel_size=9, ConvNorm::BatchNorm,
// ConvContext::Symmetric, use_bias=true. The ctor takes kernel/norm as
// params so the op stays general; the diar gate instantiates k=9/BN.
//
// Layout contract: activations [T,C] frame-major; Conv1d weights
// [out,in,K] (pointwise K=1 stored [out,in]); depthwise [C,K]; BN
// (w,b,mean,var) per-channel + eps (default 1e-5). T <= 0 no-op.

namespace diar {

enum class ConvNormKind { BatchNorm, LayerNorm };

struct ConformerConvWeights {
    const float* pw1_w = nullptr;  // pointwise_conv1 [2D,D] (+b[2D])
    const float* pw1_b = nullptr;
    const float* dw_w = nullptr;   // depthwise [D,K] (+b[D])
    const float* dw_b = nullptr;
    const float* bn_w = nullptr;  // norm weight [D] (+b[D])
    const float* bn_b = nullptr;
    const float* bn_mean = nullptr;  // BN running stats (nullptr for LN)
    const float* bn_var = nullptr;
    const float* pw2_w = nullptr;  // pointwise_conv2 [D,D] (+b[D])
    const float* pw2_b = nullptr;
};

// x [T,D] -> y [T,D].
void conformer_conv_forward(const float* x, const ConformerConvWeights& w, float* y, int t,
    int d_model, int kernel = 9, ConvNormKind norm = ConvNormKind::BatchNorm,
    float eps = 1e-5F);

}  // namespace diar
