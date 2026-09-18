// M3 Step 2 — backend skeleton (plan M3-P100-CUDA-PLAN.md §3).
//
// An opaque device/dispatch layer in front of the nn:: primitives. The CPU
// backend delegates 1:1 to the existing reference implementations (bit
// identity pinned by tests/test_backend.cpp); a CUDA backend, when compiled
// in (CMake DIAR_WITH_CUDA, OFF by default), answers per-op with Status;
// anything not yet migrated returns Status::unsupported and callers fall
// back to the CPU path. No engine-logic changes are allowed in this layer.
#ifndef DIAR_BACKEND_HPP
#define DIAR_BACKEND_HPP

#include <cstddef>
#include <memory>

namespace diar::backend {

enum class Kind { cpu, cuda };

enum class Status { ok, unsupported };

// Activation layout everywhere: [T, C] row-major (nn.hpp contract).
struct LinearOp {
    std::size_t rows, in_dim, out_dim;
};
struct LayerNormOp {
    std::size_t rows, dim;  // eps fixed at the nn.hpp pin (1e-5)
};
struct SoftmaxOp {
    std::size_t rows, cols;  // last-dim softmax
};
struct PointwiseOp {
    std::size_t n;  // relu / silu / sigmoid / xscale(scale)
    float scale;    // xscale only
};
struct GluOp {
    std::size_t t, c;
};
struct Conv1dOp {
    int t_in, c_in, c_out, k, stride, pad;  // symmetric zero pad (nn.hpp)
    bool depthwise;
};
struct Conv2dOp {
    int c_in, c_out, h_in, w_in, k, stride, pad;
};

// Opaque stream/context. A Context owns device resources for one engine
// instance; CPU contexts are stateless shells.
class Context {
public:
    virtual ~Context() = default;
    virtual Kind kind() const = 0;

    // Every op takes host-visible pointers; the backend decides whether the
    // copy happens at all (CPU: none). Blocking calls — async arrives with
    // the CUDA implementation (Step 4/5).
    virtual Status linear(const LinearOp&, const float* x, const float* w,
                          const float* b, float* y) = 0;
    virtual Status layernorm(const LayerNormOp&, const float* x,
                             const float* gamma, const float* beta, float* y) = 0;
    virtual Status softmax(const SoftmaxOp&, const float* x, float* y) = 0;
    virtual Status pointwise_silu(const PointwiseOp&, const float* x, float* y) = 0;
    virtual Status pointwise_relu(const PointwiseOp&, float* x) = 0;
    virtual Status pointwise_sigmoid(const PointwiseOp&, const float* x, float* y) = 0;
    virtual Status pointwise_xscale(const PointwiseOp&, const float* x, float* y) = 0;
    virtual Status glu(const GluOp&, const float* x, float* y) = 0;
    virtual Status conv1d(const Conv1dOp&, const float* x, const float* w,
                          const float* b, float* y) = 0;
    virtual Status conv2d(const Conv2dOp&, const float* x, const float* w,
                          const float* b, float* y) = 0;

    // rel_shift stays CPU-side in the first migration wave (small, layout-
    // fiddly); it is intentionally absent from this surface.
};

// Factory: the only entry point. Throws std::runtime_error on a cuda request
// when compiled without DIAR_WITH_CUDA (CI stays CPU-only).
std::unique_ptr<Context> create(Kind kind);

const char* status_name(Status s) noexcept;

}  // namespace diar::backend

#endif  // DIAR_BACKEND_HPP
