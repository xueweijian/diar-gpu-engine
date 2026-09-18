// M3 Step 2 — CPU backend: 1:1 delegation to the nn:: reference ops.
// The CUDA translation unit (backend_cuda.cpp) lands with DIAR_WITH_CUDA.
#include "diar/backend.hpp"

#include "diar/nn.hpp"

#include <new>
#include <stdexcept>

namespace diar::backend {
namespace {

class CpuContext final : public Context {
public:
    Kind kind() const override { return Kind::cpu; }

    Status linear(const LinearOp& o, const float* x, const float* w,
                  const float* b, float* y) override {
        nn::linear_forward(x, w, b, y, o.rows, o.in_dim, o.out_dim);
        return Status::ok;
    }
    Status layernorm(const LayerNormOp& o, const float* x, const float* gamma,
                     const float* beta, float* y) override {
        nn::layernorm_forward(x, gamma, beta, y, o.rows, o.dim);
        return Status::ok;
    }
    Status softmax(const SoftmaxOp& o, const float* x, float* y) override {
        nn::softmax_last_dim(x, y, o.rows, o.cols);
        return Status::ok;
    }
    Status pointwise_silu(const PointwiseOp& o, const float* x, float* y) override {
        nn::silu_forward(x, y, o.n);
        return Status::ok;
    }
    Status pointwise_relu(const PointwiseOp& o, float* x) override {
        nn::relu_inplace(x, o.n);
        return Status::ok;
    }
    Status pointwise_sigmoid(const PointwiseOp& o, const float* x, float* y) override {
        nn::sigmoid_forward(x, y, o.n);
        return Status::ok;
    }
    Status pointwise_xscale(const PointwiseOp& o, const float* x, float* y) override {
        nn::xscale_forward(x, y, o.n, o.scale);
        return Status::ok;
    }
    Status glu(const GluOp& o, const float* x, float* y) override {
        nn::glu_forward(x, y, o.t, o.c);
        return Status::ok;
    }
    Status conv1d(const Conv1dOp& o, const float* x, const float* w,
                  const float* b, float* y) override {
        if (o.depthwise) {
            nn::conv1d_depthwise_forward(x, w, b, y, o.t_in, o.c_in, o.k,
                                         o.stride, o.pad);
        } else {
            nn::conv1d_forward(x, w, b, y, o.t_in, o.c_in, o.c_out, o.k,
                               o.stride, o.pad);
        }
        return Status::ok;
    }
    Status conv2d(const Conv2dOp& o, const float* x, const float* w,
                  const float* b, float* y) override {
        nn::conv2d_forward(x, w, b, y, o.c_in, o.h_in, o.w_in, o.c_out, o.k,
                           o.k, o.stride, o.stride, o.pad, o.pad);
        return Status::ok;
    }
};

}  // namespace

std::unique_ptr<Context> create(Kind kind) {
    switch (kind) {
    case Kind::cpu:
        return std::make_unique<CpuContext>();
    case Kind::cuda:
        // Backend translation unit not compiled in this configuration
        // (DIAR_WITH_CUDA=OFF). Loud failure beats a silent CPU fallback.
        throw std::runtime_error(
            "diar::backend: CUDA backend not compiled (DIAR_WITH_CUDA=OFF)");
    }
    throw std::logic_error("unreachable");
}

const char* status_name(Status s) noexcept {
    switch (s) {
    case Status::ok: return "ok";
    case Status::unsupported: return "unsupported";
    }
    return "?";
}

}  // namespace diar::backend
