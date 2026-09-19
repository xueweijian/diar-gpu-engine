// backend_cuda.cpp — real CUDA backend, first cut (M3 Step 3).
//
// Guard: whole TU is inert without DIAR_WITH_CUDA. The Kaggle Step 3
// kernel compiles it with nvcc (gencode sm_60/70/75 + compute_60 PTX);
// local CI compiles only the CPU branch of the bench harness.
//
// Contract pins:
//  - Layout: cublas_layout.hpp zero-pack plan (vaccine: the local bit-level
//    oracle must stay green — if you change this file's GEMM call shape,
//    change the plan, not an ad-hoc ld here).
//  - Math mode: CUBLAS_DEFAULT_MATH — pure FP32 CUDA cores on every card
//    (TF32/tensor-op 16F route arrives with Step 5 behind the fp16 gate).
//  - Fail-fast: any CUDA/cuBLAS error throws (Status is for "unsupported",
//    not for hiding broken state).
#include "diar/backend_cuda.hpp"

#ifdef DIAR_WITH_CUDA

#include "diar/cublas_layout.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace diar::backend {
namespace {

[[noreturn]] void die(const std::string& what, int code) {
    throw std::runtime_error(what + " failed: code " + std::to_string(code));
}

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        const cudaError_t _c = (expr);                                      \
        if (_c != cudaSuccess) die(#expr, static_cast<int>(_c));            \
    } while (0)

#define CUBLAS_CHECK(expr)                                                  \
    do {                                                                    \
        const cublasStatus_t _c = (expr);                                   \
        if (_c != CUBLAS_STATUS_SUCCESS) die(#expr, static_cast<int>(_c));  \
    } while (0)

// ---- pointwise kernels (grid-stride; elementwise) -------------------------

__global__ void silu_kernel(const float* x, float* y, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        y[i] = v / (1.0f + __expf(-v));  // fast-math intrinsic is fine for
        // pointwise; parity gate is 1e-5 and __expf error is ~2 ulp here.
    }
}

__global__ void add_bias_kernel(float* y, const float* b,
                                std::size_t rows, std::size_t cols) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < rows * cols) y[i] += b[i % cols];
}

// y = 0.5*ff + residual  (conformer FF second half: fc_factor + skip)
__global__ void ff_residual_kernel(float* ff, const float* residual,
                                   std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) ff[i] = 0.5f * ff[i] + residual[i];
}

__global__ void xscale_kernel(const float* x, float* y, float scale,
                              std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * scale;
}

__global__ void relu_inplace_kernel(float* x, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] = 0.0f;
}

__global__ void sigmoid_kernel(const float* x, float* y, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + __expf(-x[i]));
}

// layernorm: one block per row, block-reduce mean/var (biased, nn.hpp pin
// eps=1e-5). dim<=2048 fits one block easily for engine shapes.
__global__ void layernorm_kernel(const float* x, const float* gamma,
                                 const float* beta, float* y,
                                 std::size_t rows, std::size_t dim) {
    extern __shared__ float smem[];  // 2 * blockDim floats
    const std::size_t row = blockIdx.x;
    const std::size_t tid = threadIdx.x;
    const std::size_t block = blockDim.x;
    float* s_sum = smem;
    float* s_sq = smem + block;

    float local_sum = 0.0f, local_sq = 0.0f;
    for (std::size_t i = tid; i < dim; i += block) {
        const float v = x[row * dim + i];
        local_sum += v;
        local_sq += v * v;
    }
    s_sum[tid] = local_sum;
    s_sq[tid] = local_sq;
    __syncthreads();
    for (std::size_t s = block / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid] += s_sq[tid + s];
        }
        __syncthreads();
    }
    const float mean = s_sum[0] / static_cast<float>(dim);
    const float var = s_sq[0] / static_cast<float>(dim) - mean * mean;
    const float inv = rsqrtf(var + 1e-5f);
    for (std::size_t i = tid; i < dim; i += block) {
        y[row * dim + i] =
            (x[row * dim + i] - mean) * inv * gamma[i] + beta[i];
    }
}

// empty kernel for launch-overhead measurement (bench harness)
__global__ void empty_kernel() {}

// ---- device buffer RAII ----------------------------------------------------

struct DeviceBuffer {
    float* ptr = nullptr;
    std::size_t bytes = 0;
    void alloc(std::size_t n_floats) {
        CUDA_CHECK(cudaMalloc(&ptr, n_floats * sizeof(float)));
        bytes = n_floats * sizeof(float);
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};

void pw_launch(std::size_t n, int block) {
    // returns grid for n elements at given block size
}

// ---- Context implementation -------------------------------------------------

class CudaContext final : public Context {
public:
    explicit CudaContext(const cuda::CcConfig& cfg) : cfg_(cfg) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasCreate(&cublas_));
        CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
        CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_DEFAULT_MATH));
    }
    ~CudaContext() override {
        cublasDestroy(cublas_);
        cudaStreamDestroy(stream_);
    }

    Kind kind() const override { return Kind::cuda; }

    Status linear(const LinearOp& o, const float* x, const float* w,
                  const float* b, float* y) override {
        const std::size_t xn = o.rows * o.in_dim;
        const std::size_t wn = o.out_dim * o.in_dim;
        const std::size_t yn = o.rows * o.out_dim;
        DeviceBuffer dx, dw, dy;
        dx.alloc(xn);
        dw.alloc(wn);
        dy.alloc(yn);
        CUDA_CHECK(cudaMemcpyAsync(dx.ptr, x, xn * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(dw.ptr, w, wn * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        const auto plan =
            linear_gemm_plan(static_cast<int>(o.rows),
                             static_cast<int>(o.in_dim),
                             static_cast<int>(o.out_dim));
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            cublas_, static_cast<cublasOperation_t>(plan.op_a),
            static_cast<cublasOperation_t>(plan.op_b), plan.m, plan.n, plan.k,
            &alpha, dw.ptr, plan.lda, dx.ptr, plan.ldb, &beta, dy.ptr,
            plan.ldc));
        if (b) {
            DeviceBuffer db;
            db.alloc(o.out_dim);
            CUDA_CHECK(cudaMemcpyAsync(db.ptr, b, o.out_dim * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_));
            add_bias_kernel<<<grid_for(yn), cfg_.pointwise_block, 0,
                              stream_>>>(dy.ptr, db.ptr, o.rows, o.out_dim);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpyAsync(y, dy.ptr, yn * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return Status::ok;
    }

    Status layernorm(const LayerNormOp& o, const float* x, const float* gamma,
                     const float* beta, float* y) override {
        const std::size_t n = o.rows * o.dim;
        DeviceBuffer dx, dg, db, dy;
        dx.alloc(n);
        dg.alloc(o.dim);
        db.alloc(o.dim);
        dy.alloc(n);
        CUDA_CHECK(cudaMemcpyAsync(dx.ptr, x, n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(dg.ptr, gamma, o.dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(db.ptr, beta, o.dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        const int block = static_cast<int>(cfg_.pointwise_block);
        const std::size_t smem = 2 * static_cast<std::size_t>(block) * sizeof(float);
        layernorm_kernel<<<static_cast<unsigned>(o.rows), block, smem,
                           stream_>>>(dx.ptr, dg.ptr, db.ptr, dy.ptr, o.rows,
                                      o.dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(y, dy.ptr, n * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return Status::ok;
    }

    Status softmax(const SoftmaxOp&, const float*, float*) override {
        return Status::unsupported;  // Step 4 (MHA wave)
    }

    Status pointwise_silu(const PointwiseOp& o, const float* x,
                          float* y) override {
        DeviceBuffer dx, dy;
        dx.alloc(o.n);
        dy.alloc(o.n);
        CUDA_CHECK(cudaMemcpyAsync(dx.ptr, x, o.n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        silu_kernel<<<grid_for(o.n), cfg_.pointwise_block, 0, stream_>>>(
            dx.ptr, dy.ptr, o.n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(y, dy.ptr, o.n * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return Status::ok;
    }

    Status pointwise_relu(const PointwiseOp& o, float* x) override {
        DeviceBuffer dx;
        dx.alloc(o.n);
        CUDA_CHECK(cudaMemcpyAsync(dx.ptr, x, o.n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        relu_inplace_kernel<<<grid_for(o.n), cfg_.pointwise_block, 0,
                              stream_>>>(dx.ptr, o.n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(x, dx.ptr, o.n * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return Status::ok;
    }

    Status pointwise_sigmoid(const PointwiseOp& o, const float* x,
                             float* y) override {
        DeviceBuffer dx, dy;
        dx.alloc(o.n);
        dy.alloc(o.n);
        CUDA_CHECK(cudaMemcpyAsync(dx.ptr, x, o.n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        sigmoid_kernel<<<grid_for(o.n), cfg_.pointwise_block, 0, stream_>>>(
            dx.ptr, dy.ptr, o.n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(y, dy.ptr, o.n * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return Status::ok;
    }

    Status pointwise_xscale(const PointwiseOp& o, const float* x,
                            float* y) override {
        DeviceBuffer dx, dy;
        dx.alloc(o.n);
        dy.alloc(o.n);
        CUDA_CHECK(cudaMemcpyAsync(dx.ptr, x, o.n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_));
        xscale_kernel<<<grid_for(o.n), cfg_.pointwise_block, 0, stream_>>>(
            dx.ptr, dy.ptr, o.scale, o.n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(y, dy.ptr, o.n * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return Status::ok;
    }

    Status glu(const GluOp&, const float*, float*) override {
        return Status::unsupported;  // Step 4 (conv wave)
    }

    Status conv1d(const Conv1dOp&, const float*, const float*, const float*,
                  float*) override {
        return Status::unsupported;  // Step 4 (conv wave)
    }

    Status conv2d(const Conv2dOp&, const float*, const float*, const float*,
                  float*) override {
        return Status::unsupported;  // Step 4 (stem wave)
    }

    cudaStream_t stream() const { return stream_; }
    cublasHandle_t cublas() const { return cublas_; }

private:
    std::size_t grid_for(std::size_t n) const {
        const std::size_t block = static_cast<std::size_t>(cfg_.pointwise_block);
        return (n + block - 1) / block;
    }
    cuda::CcConfig cfg_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
};

}  // namespace

namespace cuda {

CcConfig config_for_cc(int major, int minor) noexcept {
    (void)minor;
    // Frozen table (plan §1): first row. Extend deliberately, never guess.
    if (major == 6 || major == 7) {
        return CcConfig{/*pointwise_block=*/256};
    }
    return CcConfig{/*pointwise_block=*/256};  // safe default row
}

}  // namespace cuda

// bench-harness entry points (used by tools/step3_bench_main.cpp in the
// CUDA branch; kept here so the kernel compiles exactly two objects).

std::size_t bench_grid_for(std::size_t n, int block) {
    const std::size_t b = static_cast<std::size_t>(block);
    return (n + b - 1) / b;
}

void bench_empty_launch(cudaStream_t s, int block) {
    empty_kernel<<<bench_grid_for(1, block), block, 0, s>>>();
    const cudaError_t c = cudaGetLastError();
    if (c != cudaSuccess) die("empty_kernel launch", static_cast<int>(c));
}

void bench_ff_residual(float* ff, const float* res, std::size_t n,
                       cudaStream_t s, int block) {
    ff_residual_kernel<<<bench_grid_for(n, block), block, 0, s>>>(ff, res, n);
    const cudaError_t c = cudaGetLastError();
    if (c != cudaSuccess) die("ff_residual launch", static_cast<int>(c));
}

// Factory hand-off: backend.cpp owns create(Kind); the CUDA branch lands
// here so exactly one TU per side defines a factory piece.
Context* make_cuda_context(int cc_major, int cc_minor) {
    return new CudaContext(cuda::config_for_cc(cc_major, cc_minor));
}

int bench_device_cc() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.major * 10 + prop.minor;
}

int bench_device_sm_count() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

const char* bench_device_name() {
    static std::string name;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    name = prop.name;
    return name.c_str();
}

}  // namespace diar::backend

#else  // !DIAR_WITH_CUDA

namespace diar::backend {
// Intentionally empty TU when built without CUDA; backend.cpp owns the
// factory so CI/CI-less builds link exactly one create(Kind).
}  // namespace diar::backend

#endif  // DIAR_WITH_CUDA
