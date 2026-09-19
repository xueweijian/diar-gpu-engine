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

#include <cmath>
#include <cstdlib>
#include <cstring>
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

// ---- Step 4: device-resident forward wave ---------------------------------
// (declared in backend_cuda.hpp; all pointers DEVICE-side, zero H2D per call)

namespace {

// One thread per row: max-sub + expf + sum + div (nn.hpp softmax order).
__global__ void softmax_rows_kernel(const float* x, float* y, int cols) {
    const int row = blockIdx.x;
    const float* xr = x + static_cast<std::size_t>(row) * cols;
    float* yr = y + static_cast<std::size_t>(row) * cols;
    float m = xr[0];
    for (int j = 1; j < cols; ++j) m = fmaxf(m, xr[j]);
    float s = 0.0f;
    for (int j = 0; j < cols; ++j) {
        const float e = __expf(xr[j] - m);
        yr[j] = e;
        s += e;
    }
    const float inv = 1.0f / s;
    for (int j = 0; j < cols; ++j) yr[j] *= inv;
}

// y[r,c] = x[r, c] * sigmoid(x[r, c+C]) — torch glu(dim=-1): per-row
// halves, a = FIRST half (matches nn::glu_forward row-major [t, 2c]).
__global__ void glu_kernel(const float* x, float* y, int t, int c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < t * c) {
        const int r = i / c;
        const int ci = i - r * c;
        const float* xr = x + static_cast<std::size_t>(r) * 2 * c;
        y[i] = xr[ci] / (1.0f + __expf(-xr[c + ci]));
    }
}

// depthwise k-tap conv (symmetric zero pad (k-1)/2, conv.hpp pin) + BN
// inference + SiLU, fused. One thread per (t,c).
__global__ void dwconv_bn_silu_kernel(const float* x, const float* w,
                                      const float* b, const float* bn_g,
                                      const float* bn_b, const float* mean,
                                      const float* var, float* y, int t,
                                      int c, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * c) return;
    const int ti = i / c;
    const int ci = i - ti * c;
    const int half = (k - 1) / 2;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) {
        const int s = ti + j - half;
        if (s >= 0 && s < t)
            acc += x[static_cast<std::size_t>(s) * c + ci] *
                   w[static_cast<std::size_t>(ci) * k + j];
    }
    acc += b[ci];
    const float inv = rsqrtf(var[ci] + 1e-5f);
    acc = (acc - mean[ci]) * inv * bn_g[ci] + bn_b[ci];
    y[i] = acc / (1.0f + __expf(-acc));  // SiLU
}

// y = a + s*b (residual add; s=1 plain, s=0.5 FF second half)
__global__ void add_scaled_kernel(float* a, const float* b, float s,
                                  std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i] += s * b[i];
}

// q += bias broadcast (rows x cols, bias[cols])
__global__ void add_rows_bias_kernel(float* q, const float* bias,
                                     std::size_t n, int cols) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) q[i] += bias[i % cols];
}

// One thread per (h, query-row): scores = (ac[h,i,:] + bd[h, i-j+t-1, j]) *
// inv_sqrt(dk) -> row softmax -> probs. The rel_shift index math replaces
// torch pad+view+drop-row (contract S[i,j] = BD[i-j+T-1, j], M2-proven).
__global__ void mha_softmax_kernel(const float* ac, const float* bd,
                                   float* probs, int t, int p, float inv) {
    const int flat = blockIdx.x;  // h * t + i
    const int i = flat % t;
    const std::size_t h = static_cast<std::size_t>(flat) / t;
    const float* ar = ac + (h * t + i) * t;
    const float* br = bd + h * t * p;  // [t, p] row-major: br[a*p + m]
    float* pr = probs + (h * t + i) * t;
    float m = -3.0e38f;
    for (int j = 0; j < t; ++j) {
        // rel_shift contract (sim_mha_chain.cpp): value(bd[i][j-i+t-1])
        const float s =
            (ar[j] + br[static_cast<std::size_t>(i) * p + (j - i + t - 1)]) *
            inv;
        pr[j] = s;
        m = fmaxf(m, s);
    }
    float sum = 0.0f;
    for (int j = 0; j < t; ++j) {
        const float e = __expf(pr[j] - m);
        pr[j] = e;
        sum += e;
    }
    const float isum = 1.0f / sum;
    for (int j = 0; j < t; ++j) pr[j] *= isum;
}

// One thread per (h, query-row), plain (non-rel-pos) attention:
// probs = softmax(ac[h,i,:] * inv). Same numerical shape as
// mha_softmax_kernel minus the bd term (transformer stack has no pos emb).
__global__ void plain_softmax_kernel(const float* ac, float* probs, int t,
                                     float inv) {
    const int flat = blockIdx.x;  // h * t + i
    const float* ar = ac + static_cast<std::size_t>(flat) * t;
    float* pr = probs + static_cast<std::size_t>(flat) * t;
    float m = -3.0e38f;
    for (int j = 0; j < t; ++j) {
        const float s = ar[j] * inv;
        pr[j] = s;
        m = fmaxf(m, s);
    }
    float sum = 0.0f;
    for (int j = 0; j < t; ++j) {
        const float e = __expf(pr[j] - m);
        pr[j] = e;
        sum += e;
    }
    const float isum = 1.0f / sum;
    for (int j = 0; j < t; ++j) pr[j] *= isum;
}

// device-resident linear: y = x @ w + bias (zero-pack plan, no host copies)
void dev_linear(cublasHandle_t cublas, int block, const float* x,
                const float* w, const float* bias, float* y, int t, int in,
                int out) {
    const auto plan = linear_gemm_plan(t, in, out);
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas,
                             static_cast<cublasOperation_t>(plan.op_a),
                             static_cast<cublasOperation_t>(plan.op_b),
                             plan.m, plan.n, plan.k, &alpha, w, plan.lda, x,
                             plan.ldb, &beta, y, plan.ldc));
    if (bias) {
        const std::size_t yn = static_cast<std::size_t>(t) * out;
        add_rows_bias_kernel<<<bench_grid_for(yn, block), block>>>(y, bias,
                                                                   yn, out);
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace

std::size_t mha_scratch_floats(int t, int c, int h) {
    const int p = 2 * t - 1;
    const std::size_t tc = static_cast<std::size_t>(t) * c;
    // qu qv q k v [5*tc] | p [p*c] | out_tmp [tc] | ac bd probs [h*t*(2t+p)]
    return 6 * tc + static_cast<std::size_t>(p) * c +
           static_cast<std::size_t>(h) * t * (2 * t + p);
}

// GpuArena: one cudaMalloc, bump-allocated named spans (weights device
// resident = uploaded once; per-chunk work touches activations + scratch).
void GpuArena::init(std::size_t n_floats) {
    cap_ = n_floats;
    CUDA_CHECK(cudaMalloc(&base_, n_floats * sizeof(float)));
    reg_cap_ = 256;
    names_ = static_cast<const char**>(std::calloc(reg_cap_, sizeof(char*)));
    ptrs_ = static_cast<float**>(std::calloc(reg_cap_, sizeof(float*)));
    if (!names_ || !ptrs_) die("arena registry alloc", -1);
}

float* GpuArena::alloc(const char* name, std::size_t n) {
    if (used_ + n > cap_) die("arena exhausted", static_cast<int>(used_ + n));
    if (count_ == reg_cap_) {
        reg_cap_ *= 2;
        names_ = static_cast<const char**>(
            std::realloc(names_, reg_cap_ * sizeof(char*)));
        ptrs_ = static_cast<float**>(
            std::realloc(ptrs_, reg_cap_ * sizeof(float*)));
        if (!names_ || !ptrs_) die("arena registry grow", -1);
    }
    float* p = base_ + used_;
    used_ += n;
    const std::size_t len = std::strlen(name);
    char* copy = static_cast<char*>(std::malloc(len + 1));
    if (!copy) die("arena name alloc", -1);
    std::memcpy(copy, name, len + 1);
    names_[count_] = copy;
    ptrs_[count_] = p;
    ++count_;
    return p;
}

float* GpuArena::span(const char* name) const {
    for (std::size_t i = 0; i < count_; ++i)
        if (std::strcmp(names_[i], name) == 0) return ptrs_[i];
    die("arena span missing", -1);
    return nullptr;
}

GpuArena::~GpuArena() {
    if (base_) cudaFree(base_);
    if (names_) {
        for (std::size_t i = 0; i < count_; ++i)
            std::free(const_cast<char*>(names_[i]));  // owned copies
    }
    std::free(names_);
    std::free(ptrs_);
}

void gpu_relpos_mha(cublasHandle_t cublas, cudaStream_t stream, int block,
                    const MhaDevWeights& w, const float* x, const float* pos,
                    float* y, float* ws, int t, int c, int heads) {
    const int p = 2 * t - 1;
    const int dk = c / heads;
    const std::size_t tc = static_cast<std::size_t>(t) * c;
    const std::size_t pc = static_cast<std::size_t>(p) * c;
    float* qu = ws;
    float* qv = qu + tc;
    float* dq = qv + tc;
    float* dkbuf = dq + tc;
    float* dv = dkbuf + tc;
    float* dp = dv + tc;
    float* otmp = dp + pc;
    const std::size_t per_h = static_cast<std::size_t>(heads) * t;
    float* ac = otmp + tc;          // [h, t, t]
    float* bd = ac + per_h * t;     // [h, t, p]
    float* probs = bd + per_h * p;  // [h, t, t]

    dev_linear(cublas, block, x, w.q_w, w.q_b, dq, t, c, c);
    dev_linear(cublas, block, x, w.k_w, w.k_b, dkbuf, t, c, c);
    dev_linear(cublas, block, x, w.v_w, w.v_b, dv, t, c, c);
    dev_linear(cublas, block, pos, w.pos_w, nullptr, dp, p, c, c);
    CUDA_CHECK(cudaMemcpyAsync(qu, dq, tc * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    add_rows_bias_kernel<<<bench_grid_for(tc, block), block, 0, stream>>>(
        qu, w.bu, tc, c);
    CUDA_CHECK(cudaMemcpyAsync(qv, dq, tc * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));
    add_rows_bias_kernel<<<bench_grid_for(tc, block), block, 0, stream>>>(
        qv, w.bv, tc, c);
    CUDA_CHECK(cudaGetLastError());

    // ac[h] = qu_h @ k_h^T ; bd[h] = qv_h @ p_h^T. Head h of a row-major
    // [T,C] tensor is columns [h*dk,(h+1)*dk) = a col-major [dk,T] with
    // ld=C — zero-pack head split (cublas_layout contract family).
    {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, t, t, dk, &alpha, dkbuf, c, dk,
            qu, c, dk, &beta, ac, t, static_cast<long long>(t) * t, heads));
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, p, t, dk, &alpha, dp, c, dk,
            qv, c, dk, &beta, bd, p, static_cast<long long>(t) * p, heads));
    }

    mha_softmax_kernel<<<per_h, 1, 0, stream>>>(
        ac, bd, probs, t, p, 1.0f / std::sqrt(static_cast<float>(dk)));
    CUDA_CHECK(cudaGetLastError());

    // ctx_h [t,dk] = probs_h @ v_h — written straight into the merged [t,c]
    // head slice (heads are contiguous column blocks: zero transpose).
    {
        const float alpha = 1.0f, beta = 0.0f;
        // A = dv op_N: A(m_idx, r) = dv[r*c + hr + m_idx] (lda=c, batch=hr).
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N, dk, t, t, &alpha, dv, c, dk,
            probs, t, static_cast<long long>(t) * t, &beta, otmp, c, dk,
            heads));
    }
    dev_linear(cublas, block, otmp, w.out_w, w.out_b, y, t, c, c);
}

std::size_t transformer_block_scratch_floats(int t, int h, int inner,
                                             int heads) {
    // dq dk dv ctx attn_out(->h1) h1_ln ff_out [7*t*h] | ff_mid [t*inner]
    // | ac probs [2*heads*t*t]
    const std::size_t th = static_cast<std::size_t>(t) * h;
    return 7 * th + static_cast<std::size_t>(t) * inner +
           2 * static_cast<std::size_t>(heads) * t * t;
}

void gpu_transformer_block(cublasHandle_t cublas, cudaStream_t stream,
                           int block, const TransformerDevWeights& w,
                           const float* x, float* y, float* ws, int t, int h,
                           int inner, int heads) {
    const int dk = h / heads;
    const std::size_t th = static_cast<std::size_t>(t) * h;
    float* dq = ws;
    float* dkbuf = dq + th;
    float* dv = dkbuf + th;
    float* ctx = dv + th;      // merged head slices [t,h]
    float* attn_out = ctx + th;  // becomes h1 after the residual add
    float* h1_ln = attn_out + th;
    float* ff_out = h1_ln + th;
    float* ff_mid = ff_out + th;                 // [t, inner]
    float* ac = ff_mid + static_cast<std::size_t>(t) * inner;  // [heads,t,t]
    float* probs = ac + static_cast<std::size_t>(heads) * t * t;

    // q/k/v projections (k=h)
    dev_linear(cublas, block, x, w.q_w, w.q_b, dq, t, h, h);
    dev_linear(cublas, block, x, w.k_w, w.k_b, dkbuf, t, h, h);
    dev_linear(cublas, block, x, w.v_w, w.v_b, dv, t, h, h);
    // ac[h] = q_h @ k_h^T — same zero-pack contiguous head-split contract
    // as the rel-pos MHA (col-major [dk,t] ld=h slices of row-major [t,h]).
    {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, t, t, dk, &alpha, dkbuf, h, dk,
            dq, h, dk, &beta, ac, t, static_cast<long long>(t) * t, heads));
    }
    plain_softmax_kernel<<<static_cast<std::size_t>(heads) * t, 1, 0, stream>>>(
        ac, probs, t, 1.0f / std::sqrt(static_cast<float>(dk)));
    CUDA_CHECK(cudaGetLastError());
    // ctx_h [t,dk] = probs_h @ v_h, written into the merged [t,h] slice.
    {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N, dk, t, t, &alpha, dv, h, dk,
            probs, t, static_cast<long long>(t) * t, &beta, ctx, h, dk,
            heads));
    }
    // post-LN block: attn proj -> +res -> LN1 -> FF(relu) -> +res -> LN2
    dev_linear(cublas, block, ctx, w.o_w, w.o_b, attn_out, t, h, h);
    add_scaled_kernel<<<bench_grid_for(th, block), block, 0, stream>>>(
        attn_out, x, 1.0f, th);  // h1 = attn_out + x
    layernorm_kernel<<<static_cast<unsigned>(t), block,
                       2 * block * sizeof(float), stream>>>(
        attn_out, w.ln1_g, w.ln1_b, h1_ln, t, h);
    dev_linear(cublas, block, h1_ln, w.f1_w, w.f1_b, ff_mid, t, h, inner);
    relu_inplace_kernel<<<bench_grid_for(static_cast<std::size_t>(t) * inner,
                                         block),
                          block, 0, stream>>>(
        ff_mid, static_cast<std::size_t>(t) * inner);
    dev_linear(cublas, block, ff_mid, w.f2_w, w.f2_b, ff_out, t, inner, h);
    add_scaled_kernel<<<bench_grid_for(th, block), block, 0, stream>>>(
        ff_out, h1_ln, 1.0f, th);  // h2 = ff_out + h1_ln
    layernorm_kernel<<<static_cast<unsigned>(t), block,
                       2 * block * sizeof(float), stream>>>(
        ff_out, w.ln2_g, w.ln2_b, y, t, h);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t conformer_layer_scratch_floats(int t, int c, int d_ff, int h) {
    // res tmp attn [3tc] | ff_up [t*ff] | pw1 [2tc] | glu_out dw_out [2tc]
    // | mha scratch
    const std::size_t tc = static_cast<std::size_t>(t) * c;
    return 7 * tc + static_cast<std::size_t>(t) * d_ff +
           mha_scratch_floats(t, c, h);
}

void gpu_conformer_layer(cublasHandle_t cublas, cudaStream_t stream,
                         int block, const LayerDevWeights& w, const float* x,
                         const float* pos, float* y, float* ws, int t, int c,
                         int d_ff, int heads) {
    const std::size_t tc = static_cast<std::size_t>(t) * c;
    float* res = ws;
    float* tmp = res + tc;
    float* attn = tmp + tc;
    float* ff_up = attn + tc;                  // [t, d_ff]
    float* pw1 = ff_up + static_cast<std::size_t>(t) * d_ff;  // [t, 2c]
    float* glu_out = pw1 + 2 * tc;
    float* dw_out = glu_out + tc;
    float* mha_ws = dw_out + tc;

    const int pw_block = block;
    // residual = x (this is the first write; x may alias nothing)
    CUDA_CHECK(cudaMemcpyAsync(res, x, tc * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    // FF1: x = LN(res); ff = silu(x@W1+b1)@W2+b2; res += 0.5*ff
    layernorm_kernel<<<static_cast<unsigned>(t), block,
                       2 * block * sizeof(float), stream>>>(
        res, w.n_ff1_g, w.n_ff1_b, tmp, t, c);
    dev_linear(cublas, block, tmp, w.ff1_w1, w.ff1_b1, ff_up, t, c, d_ff);
    silu_kernel<<<bench_grid_for(static_cast<std::size_t>(t) * d_ff, block),
                  block, 0, stream>>>(ff_up, ff_up,
                                      static_cast<std::size_t>(t) * d_ff);
    dev_linear(cublas, block, ff_up, w.ff1_w2, w.ff1_b2, tmp, t, d_ff, c);
    add_scaled_kernel<<<bench_grid_for(tc, pw_block), pw_block, 0, stream>>>(
        res, tmp, 0.5f, tc);

    // MHA: a = LN(res); attn = mha(a, pos); res += attn
    layernorm_kernel<<<static_cast<unsigned>(t), block,
                       2 * block * sizeof(float), stream>>>(
        res, w.n_sa_g, w.n_sa_b, tmp, t, c);
    gpu_relpos_mha(cublas, stream, block, w.attn, tmp, pos, attn, mha_ws, t,
                   c, heads);
    add_scaled_kernel<<<bench_grid_for(tc, pw_block), pw_block, 0, stream>>>(
        res, attn, 1.0f, tc);

    // conv: v = LN(res); pw1 -> GLU -> dw+BN+SiLU -> pw2; res += v
    layernorm_kernel<<<static_cast<unsigned>(t), block,
                       2 * block * sizeof(float), stream>>>(
        res, w.n_conv_g, w.n_conv_b, tmp, t, c);
    dev_linear(cublas, block, tmp, w.conv.pw1_w, w.conv.pw1_b, pw1, t, c, 2 * c);
    glu_kernel<<<bench_grid_for(tc, pw_block), pw_block, 0, stream>>>(
        pw1, glu_out, t, c);
    dwconv_bn_silu_kernel<<<bench_grid_for(tc, pw_block), pw_block, 0,
                            stream>>>(glu_out, w.conv.dw_w, w.conv.dw_b,
                                      w.conv.bn_w, w.conv.bn_b,
                                      w.conv.bn_mean, w.conv.bn_var, dw_out,
                                      t, c, 9);
    dev_linear(cublas, block, dw_out, w.conv.pw2_w, w.conv.pw2_b, tmp, t, c, c);
    add_scaled_kernel<<<bench_grid_for(tc, pw_block), pw_block, 0, stream>>>(
        res, tmp, 1.0f, tc);

    // FF2 + final LN
    layernorm_kernel<<<static_cast<unsigned>(t), block,
                       2 * block * sizeof(float), stream>>>(
        res, w.n_ff2_g, w.n_ff2_b, tmp, t, c);
    dev_linear(cublas, block, tmp, w.ff2_w1, w.ff2_b1, ff_up, t, c, d_ff);
    silu_kernel<<<bench_grid_for(static_cast<std::size_t>(t) * d_ff, block),
                  block, 0, stream>>>(ff_up, ff_up,
                                      static_cast<std::size_t>(t) * d_ff);
    dev_linear(cublas, block, ff_up, w.ff2_w2, w.ff2_b2, tmp, t, d_ff, c);
    add_scaled_kernel<<<bench_grid_for(tc, pw_block), pw_block, 0, stream>>>(
        res, tmp, 0.5f, tc);
    layernorm_kernel<<<static_cast<unsigned>(t), block,
                       2 * block * sizeof(float), stream>>>(
        res, w.n_out_g, w.n_out_b, y, t, c);
    CUDA_CHECK(cudaGetLastError());
}

// isolated-op bench entries (parity vs CPU references)

void bench_linear(cublasHandle_t cublas, cudaStream_t stream, int block,
                  const float* x, const float* w, const float* b, float* y,
                  int t, int in, int out) {
    dev_linear(cublas, block, x, w, b, y, t, in, out);
    CUDA_CHECK(cudaGetLastError());
}

void bench_softmax_rows(const float* x, float* y, int rows, int cols,
                        cudaStream_t s, int block) {
    (void)block;
    softmax_rows_kernel<<<rows, 1, 0, s>>>(x, y, cols);
    CUDA_CHECK(cudaGetLastError());
}

void bench_glu(const float* x, float* y, int t, int c, cudaStream_t s,
               int block) {
    glu_kernel<<<bench_grid_for(static_cast<std::size_t>(t) * c, block),
                 block, 0, s>>>(x, y, t, c);
    CUDA_CHECK(cudaGetLastError());
}

void bench_dwconv_bn_silu(const float* x, const float* w, const float* b,
                          const float* bn_g, const float* bn_b,
                          const float* mean, const float* var, float* y,
                          int t, int c, int k, cudaStream_t s, int block) {
    dwconv_bn_silu_kernel<<<bench_grid_for(static_cast<std::size_t>(t) * c,
                                           block),
                            block, 0, s>>>(x, w, b, bn_g, bn_b, mean, var, y,
                                           t, c, k);
    CUDA_CHECK(cudaGetLastError());
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
