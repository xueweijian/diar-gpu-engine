// tools/step4_bench_main.cpp — Step 4 bench harness (MHA + conv + layer).
//
// Same two-mode pattern as step3_bench_main.cpp:
//  - CPU selftest (no CUDA): reference-chain sanity + naive-oracle cross
//    checks so a Kaggle round-trip never burns on harness bugs.
//  - GPU bench (DIAR_WITH_CUDA, nvcc):
//      G-S4a  op parity: softmax / glu / dwconv+BN+SiLU  <= 1e-5
//      G-S4b  rel-pos MHA parity (t=12/20/26)            <= 5e-5
//      G-S4c  full conformer layer parity (t=20)         <= 2e-4
//      G-S4d  17-layer device-resident timeline (ms/chunk) — decision input
//  Reports [s4] JSON lines. All Step-4 chains run device-resident (GpuArena,
//  zero H2D per chunk) — this is the production shape, not the Context
//  per-call-copy path.
#include "diar/backend_cuda.hpp"
#include "diar/conformer.hpp"
#include "diar/conv.hpp"
#include "diar/cublas_layout.hpp"
#include "diar/mha.hpp"
#include "diar/nn.hpp"
#include "diar/posenc.hpp"

#ifdef DIAR_WITH_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ADL only reaches the step4 entries whose signatures carry a diar::backend
// type (MhaDevWeights / LayerDevWeights); the int-only and float*-only
// helpers must be pulled in explicitly (Step 3 fully qualified at each
// call site instead).
using diar::backend::bench_dwconv_bn_silu;
using diar::backend::bench_glu;
using diar::backend::bench_softmax_rows;
using diar::backend::conformer_layer_scratch_floats;
using diar::backend::gpu_conformer_layer;
using diar::backend::gpu_relpos_mha;
using diar::backend::mha_scratch_floats;
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace nn = diar::nn;

namespace {

struct TShape { const char* name; int t; };
const TShape kTShapes[] = {{"t12", 12}, {"t20", 20}, {"t26", 26}};

constexpr int kC = 512;
constexpr int kH = 8;
constexpr int kDk = kC / kH;   // 64
constexpr int kF = 2048;
constexpr int kK = 9;
constexpr int kLayerParityT = 20;
constexpr int kTimelineLayers = 17;
constexpr int kTimelineChunks = 30;

constexpr double kOpGate = 1e-5;
constexpr double kMhaGate = 5e-5;
constexpr double kLayerGate = 2e-4;

std::mt19937& rng() { static std::mt19937 r(20260919); return r; }

void fill_random(std::vector<float>& v) {
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (float& x : v) x = d(rng());
}

#ifdef DIAR_WITH_CUDA
double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}
#endif

void report(const std::string& key, const std::string& json) {
    std::printf("[s4] {\"k\":\"%s\",%s", key.c_str(), json.c_str());
    std::fflush(stdout);
}

#ifdef DIAR_WITH_CUDA
// %.9e float formatting — std::to_string gives 6 decimals and would flatten
// sub-1e-6 max_abs values to "0.000000" (JIT bit-identity check needs them).
std::string fmt9(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9e", v);
    return buf;
}
#endif

// ---- host-side weight sets (arena upload sources + CPU references) --------

struct MhaHost {
    std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b, pos_w, bu, bv, out_w,
        out_b;
    void build(int c, int h) {
        auto mk = [this](std::vector<float>& v, std::size_t n) {
            v.resize(n);
            fill_random(v);
        };
        mk(q_w, c * c); mk(q_b, c); mk(k_w, c * c); mk(k_b, c);
        mk(v_w, c * c); mk(v_b, c); mk(pos_w, c * c);
        bu.resize(c); fill_random(bu);   // [H,Dk] row-major == flat c
        bv.resize(c); fill_random(bv);
        mk(out_w, c * c); mk(out_b, c);
        (void)h;
    }
    diar::RelPosMhaWeights host_ref() const {
        return diar::RelPosMhaWeights{q_w.data(), q_b.data(), k_w.data(),
                                      k_b.data(), v_w.data(), v_b.data(),
                                      pos_w.data(), bu.data(), bv.data(),
                                      out_w.data(), out_b.data()};
    }
};

struct ConvHost {
    std::vector<float> pw1_w, pw1_b, dw_w, dw_b, bn_w, bn_b, bn_mean, bn_var,
        pw2_w, pw2_b;
    void build(int c, int k) {
        auto mk = [this](std::vector<float>& v, std::size_t n) {
            v.resize(n);
            fill_random(v);
        };
        mk(pw1_w, 2 * c * c); mk(pw1_b, 2 * c);
        mk(dw_w, c * k); mk(dw_b, c);
        mk(bn_w, c); mk(bn_b, c);
        bn_mean.resize(c);   // BN running stats: mean ~ small, var in (0.5,1)
        fill_random(bn_mean);
        bn_var.resize(c);
        std::uniform_real_distribution<float> d(0.5f, 1.0f);
        for (float& x : bn_var) x = d(rng());
        mk(pw2_w, c * c); mk(pw2_b, c);
    }
};

struct LayerHost {
    std::vector<float> n_g[5], n_b[5];
    std::vector<float> ff1_w1, ff1_b1, ff1_w2, ff1_b2, ff2_w1, ff2_b1, ff2_w2,
        ff2_b2;
    MhaHost mha;
    ConvHost conv;
    void build(int c, int d_ff, int k) {
        for (int i = 0; i < 5; ++i) {
            n_g[i].assign(c, 1.0f);
            n_b[i].assign(c, 0.0f);
        }
        auto mk = [this](std::vector<float>& v, std::size_t n) {
            v.resize(n);
            fill_random(v);
        };
        mk(ff1_w1, d_ff * c); mk(ff1_b1, d_ff);
        mk(ff1_w2, c * d_ff); mk(ff1_b2, c);
        mk(ff2_w1, d_ff * c); mk(ff2_b1, d_ff);
        mk(ff2_w2, c * d_ff); mk(ff2_b2, c);
        mha.build(c, kH);
        conv.build(c, k);
    }
    diar::ConformerLayerWeights host_ref() {
        diar::ConformerLayerWeights w{};
        w.n_ff1_g = n_g[0].data(); w.n_ff1_b = n_b[0].data();
        w.n_sa_g = n_g[1].data();  w.n_sa_b = n_b[1].data();
        w.n_conv_g = n_g[2].data(); w.n_conv_b = n_b[2].data();
        w.n_ff2_g = n_g[3].data(); w.n_ff2_b = n_b[3].data();
        w.n_out_g = n_g[4].data(); w.n_out_b = n_b[4].data();
        w.ff1_w1 = ff1_w1.data(); w.ff1_b1 = ff1_b1.data();
        w.ff1_w2 = ff1_w2.data(); w.ff1_b2 = ff1_b2.data();
        w.ff2_w1 = ff2_w1.data(); w.ff2_b1 = ff2_b1.data();
        w.ff2_w2 = ff2_w2.data(); w.ff2_b2 = ff2_b2.data();
        w.attn = mha.host_ref();
        diar::ConformerConvWeights cv{};
        cv.pw1_w = conv.pw1_w.data(); cv.pw1_b = conv.pw1_b.data();
        cv.dw_w = conv.dw_w.data();   cv.dw_b = conv.dw_b.data();
        cv.bn_w = conv.bn_w.data();   cv.bn_b = conv.bn_b.data();
        cv.bn_mean = conv.bn_mean.data(); cv.bn_var = conv.bn_var.data();
        cv.pw2_w = conv.pw2_w.data(); cv.pw2_b = conv.pw2_b.data();
        w.conv = cv;
        return w;
    }
};

}  // namespace

#ifdef DIAR_WITH_CUDA
// ============================ GPU branch ===================================

namespace {

#define CUDA4_CHECK(expr)                                                   \
    do {                                                                    \
        const cudaError_t _c = (expr);                                      \
        if (_c != cudaSuccess) {                                            \
            std::printf("[s4] {\"k\":\"fatal\",\"what\":\"%s\",\"code\":%d}\n", \
                        #expr, static_cast<int>(_c));                       \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

#define CUBLAS4_CHECK(expr)                                                 \
    do {                                                                    \
        const cublasStatus_t _c = (expr);                                   \
        if (_c != CUBLAS_STATUS_SUCCESS) {                                  \
            std::printf("[s4] {\"k\":\"fatal\",\"what\":\"%s\",\"code\":%d}\n", \
                        #expr, static_cast<int>(_c));                       \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

float* h2d(const std::vector<float>& v, GpuArena& arena, const char* name) {
    float* d = arena.alloc(name, v.size());
    CUDA4_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    return d;
}

// upload a full layer; returns device weight struct + keeps host alive via
// the caller-owned LayerHost (CPU reference uses the same host vectors).
diar::backend::LayerDevWeights upload_layer(
    diar::backend::GpuArena& arena, const LayerHost& h,
                                   const std::string& tag) {
    diar::backend::LayerDevWeights w{};
    auto up = [&](const std::vector<float>& v, const char* n) {
        return h2d(v, arena, (tag + n).c_str());
    };
    w.n_ff1_g = up(h.n_g[0], ".ng0"); w.n_ff1_b = up(h.n_b[0], ".nb0");
    w.n_sa_g = up(h.n_g[1], ".ng1");  w.n_sa_b = up(h.n_b[1], ".nb1");
    w.n_conv_g = up(h.n_g[2], ".ng2"); w.n_conv_b = up(h.n_b[2], ".nb2");
    w.n_ff2_g = up(h.n_g[3], ".ng3"); w.n_ff2_b = up(h.n_b[3], ".nb3");
    w.n_out_g = up(h.n_g[4], ".ng4"); w.n_out_b = up(h.n_b[4], ".nb4");
    w.ff1_w1 = up(h.ff1_w1, ".f1w1"); w.ff1_b1 = up(h.ff1_b1, ".f1b1");
    w.ff1_w2 = up(h.ff1_w2, ".f1w2"); w.ff1_b2 = up(h.ff1_b2, ".f1b2");
    w.ff2_w1 = up(h.ff2_w1, ".f2w1"); w.ff2_b1 = up(h.ff2_b1, ".f2b1");
    w.ff2_w2 = up(h.ff2_w2, ".f2w2"); w.ff2_b2 = up(h.ff2_b2, ".f2b2");
    w.attn.q_w = up(h.mha.q_w, ".qw"); w.attn.q_b = up(h.mha.q_b, ".qb");
    w.attn.k_w = up(h.mha.k_w, ".kw"); w.attn.k_b = up(h.mha.k_b, ".kb");
    w.attn.v_w = up(h.mha.v_w, ".vw"); w.attn.v_b = up(h.mha.v_b, ".vb");
    w.attn.pos_w = up(h.mha.pos_w, ".pw");
    w.attn.bu = up(h.mha.bu, ".bu"); w.attn.bv = up(h.mha.bv, ".bv");
    w.attn.out_w = up(h.mha.out_w, ".ow"); w.attn.out_b = up(h.mha.out_b, ".ob");
    w.conv.pw1_w = up(h.conv.pw1_w, ".c1w"); w.conv.pw1_b = up(h.conv.pw1_b, ".c1b");
    w.conv.dw_w = up(h.conv.dw_w, ".dw");   w.conv.dw_b = up(h.conv.dw_b, ".db");
    w.conv.bn_w = up(h.conv.bn_w, ".bw");   w.conv.bn_b = up(h.conv.bn_b, ".bb");
    w.conv.bn_mean = up(h.conv.bn_mean, ".bm"); w.conv.bn_var = up(h.conv.bn_var, ".bv2");
    w.conv.pw2_w = up(h.conv.pw2_w, ".c2w"); w.conv.pw2_b = up(h.conv.pw2_b, ".c2b");
    return w;
}

double median_of(std::vector<double>& v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const bool parity_only =
        argc > 1 && std::string(argv[1]) == "--parity-only";
    CUDA4_CHECK(cudaSetDevice(0));
    report("env", std::string("\"name\":\"") + bench_device_name() +
                  "\",\"cc\":" + std::to_string(bench_device_cc()) + "}\n");

    cublasHandle_t cublas;
    CUBLAS4_CHECK(cublasCreate(&cublas));
    CUBLAS4_CHECK(cublasSetMathMode(cublas, CUBLAS_DEFAULT_MATH));
    cudaStream_t stream;
    CUDA4_CHECK(cudaStreamCreate(&stream));
    CUBLAS4_CHECK(cublasSetStream(cublas, stream));
    constexpr int kBlock = 256;

    diar::backend::GpuArena arena;
    arena.init(64u * 1024u * 1024u);  // 64 MiB scratch+weights for parity part

    // ---- G-S4a: isolated op parity (device-resident inputs) --------------
    {
        const int t = 26;
        // softmax: rows = h*t (mha-shaped), cols = t
        {
            const int rows = kH * t, cols = t;
            std::vector<float> x(rows * cols);
            fill_random(x);
            float *dx, *dy;
            CUDA4_CHECK(cudaMalloc(&dx, x.size() * 4));
            CUDA4_CHECK(cudaMalloc(&dy, x.size() * 4));
            CUDA4_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4,
                                   cudaMemcpyHostToDevice));
            bench_softmax_rows(dx, dy, rows, cols, stream, kBlock);
            CUDA4_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> got(x.size());
            CUDA4_CHECK(cudaMemcpy(got.data(), dy, x.size() * 4,
                                   cudaMemcpyDeviceToHost));
            std::vector<float> ref(x.size());
            nn::softmax_last_dim(x.data(), ref.data(), rows, cols);
            const double m = max_abs(ref, got);
            report("parity_softmax", "\"max_abs\":" + fmt9(m) +
                   ",\"pass\":" + (m <= kOpGate ? "true" : "false") + "}\n");
            cudaFree(dx);
            cudaFree(dy);
        }
        // glu: [t, 2c] -> [t, c]
        {
            std::vector<float> x(static_cast<std::size_t>(t) * 2 * kC);
            fill_random(x);
            float *dx, *dy;
            CUDA4_CHECK(cudaMalloc(&dx, x.size() * 4));
            CUDA4_CHECK(cudaMalloc(&dy,
                                   static_cast<std::size_t>(t) * kC * 4));
            CUDA4_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4,
                                   cudaMemcpyHostToDevice));
            bench_glu(dx, dy, t, kC, stream, kBlock);
            CUDA4_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> got(static_cast<std::size_t>(t) * kC);
            CUDA4_CHECK(cudaMemcpy(got.data(), dy, got.size() * 4,
                                   cudaMemcpyDeviceToHost));
            std::vector<float> ref(got.size());
            nn::glu_forward(x.data(), ref.data(), t, kC);
            const double m = max_abs(ref, got);
            report("parity_glu", "\"max_abs\":" + fmt9(m) +
                   ",\"pass\":" + (m <= kOpGate ? "true" : "false") + "}\n");
            cudaFree(dx);
            cudaFree(dy);
        }
        // dwconv+BN+SiLU fused vs nn:: depthwise + BN + silu chain
        {
            std::vector<float> x(static_cast<std::size_t>(t) * kC);
            std::vector<float> w(kC * kK), b(kC), g(kC), bb(kC), mean(kC),
                var(kC);
            fill_random(x);
            fill_random(w);
            fill_random(b);
            fill_random(g);
            fill_random(bb);
            fill_random(mean);
            std::uniform_real_distribution<float> d(0.5f, 1.0f);
            for (float& v : var) v = d(rng());
            float *dx, *dw, *db, *dg, *dbb, *dm, *dv2, *dy;
            CUDA4_CHECK(cudaMalloc(&dx, x.size() * 4));
            CUDA4_CHECK(cudaMalloc(&dw, w.size() * 4));
            CUDA4_CHECK(cudaMalloc(&db, b.size() * 4));
            CUDA4_CHECK(cudaMalloc(&dg, g.size() * 4));
            CUDA4_CHECK(cudaMalloc(&dbb, bb.size() * 4));
            CUDA4_CHECK(cudaMalloc(&dm, mean.size() * 4));
            CUDA4_CHECK(cudaMalloc(&dv2, var.size() * 4));
            CUDA4_CHECK(cudaMalloc(&dy, x.size() * 4));
            CUDA4_CHECK(cudaMemcpy(dx, x.data(), x.size() * 4,
                                   cudaMemcpyHostToDevice));
            CUDA4_CHECK(cudaMemcpy(dw, w.data(), w.size() * 4,
                                   cudaMemcpyHostToDevice));
            CUDA4_CHECK(cudaMemcpy(db, b.data(), b.size() * 4,
                                   cudaMemcpyHostToDevice));
            CUDA4_CHECK(cudaMemcpy(dg, g.data(), g.size() * 4,
                                   cudaMemcpyHostToDevice));
            CUDA4_CHECK(cudaMemcpy(dbb, bb.data(), bb.size() * 4,
                                   cudaMemcpyHostToDevice));
            CUDA4_CHECK(cudaMemcpy(dm, mean.data(), mean.size() * 4,
                                   cudaMemcpyHostToDevice));
            CUDA4_CHECK(cudaMemcpy(dv2, var.data(), var.size() * 4,
                                   cudaMemcpyHostToDevice));
            bench_dwconv_bn_silu(dx, dw, db, dg, dbb, dm, dv2, dy, t, kC, kK,
                                 stream, kBlock);
            CUDA4_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> got(x.size());
            CUDA4_CHECK(cudaMemcpy(got.data(), dy, x.size() * 4,
                                   cudaMemcpyDeviceToHost));
            // CPU reference: depthwise (symmetric pad 4) -> BN -> silu
            std::vector<float> dw_out(x.size()), bn_out(x.size()), ref(x.size());
            nn::conv1d_depthwise_forward(x.data(), w.data(), b.data(),
                                         dw_out.data(), t, kC, kK, 1, 4);
            nn::batchnorm1d_infer_forward(dw_out.data(), g.data(), bb.data(),
                                          mean.data(), var.data(),
                                          bn_out.data(), t, kC);
            nn::silu_forward(bn_out.data(), ref.data(), x.size());
            const double m = max_abs(ref, got);
            report("parity_dwconv", "\"max_abs\":" + fmt9(m) +
                   ",\"pass\":" + (m <= kOpGate ? "true" : "false") + "}\n");
            for (float* p : {dx, dw, db, dg, dbb, dm, dv2, dy}) cudaFree(p);
        }
    }

    // ---- G-S4b: MHA parity (fresh arena per shape is fine; use one big) --
    {
        for (const TShape& s : kTShapes) {
            const int t = s.t, p = 2 * t - 1;
            MhaHost mh;
            mh.build(kC, kH);
            diar::backend::GpuArena a;
            a.init(8u * 1024u * 1024u);
            diar::backend::MhaDevWeights dw = [&] {
                diar::backend::MhaDevWeights w{};
                w.q_w = h2d(mh.q_w, a, "qw"); w.q_b = h2d(mh.q_b, a, "qb");
                w.k_w = h2d(mh.k_w, a, "kw"); w.k_b = h2d(mh.k_b, a, "kb");
                w.v_w = h2d(mh.v_w, a, "vw"); w.v_b = h2d(mh.v_b, a, "vb");
                w.pos_w = h2d(mh.pos_w, a, "pw");
                w.bu = h2d(mh.bu, a, "bu"); w.bv = h2d(mh.bv, a, "bv");
                w.out_w = h2d(mh.out_w, a, "ow");
                w.out_b = h2d(mh.out_b, a, "ob");
                return w;
            }();
            std::vector<float> x(static_cast<std::size_t>(t) * kC);
            std::vector<float> pos(static_cast<std::size_t>(p) * kC);
            fill_random(x);
            fill_random(pos);
            float* dx = h2d(x, a, "x");
            float* dpos = h2d(pos, a, "pos");
            float* dy = a.alloc("y", static_cast<std::size_t>(t) * kC);
            std::vector<float> ws(mha_scratch_floats(t, kC, kH));
            float* dws = a.alloc("ws", ws.size());
            gpu_relpos_mha(cublas, stream, kBlock, dw, dx, dpos, dy, dws, t,
                           kC, kH);
            CUDA4_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> got(static_cast<std::size_t>(t) * kC);
            CUDA4_CHECK(cudaMemcpy(got.data(), dy, got.size() * 4,
                                   cudaMemcpyDeviceToHost));
            std::vector<float> ref(got.size());
            diar::relpos_mha_forward(x.data(), pos.data(), mh.host_ref(),
                                     ref.data(), t, kC, kH);
            const double m = max_abs(ref, got);
            report("parity_mha", std::string("\"shape\":\"") + s.name +
                   "\",\"max_abs\":" + fmt9(m) +
                   ",\"gate\":" + std::to_string(kMhaGate) +
                   ",\"pass\":" + (m <= kMhaGate ? "true" : "false") + "}\n");
        }
    }

    // ---- G-S4c: full conformer layer parity ------------------------------
    // (runs in --parity-only mode too: the PTX-JIT pass must gate the layer)
    {
        const int t = kLayerParityT, p = 2 * t - 1;
        LayerHost lh;
        lh.build(kC, kF, kK);
        diar::backend::GpuArena a;
        a.init(64u * 1024u * 1024u);
        diar::backend::LayerDevWeights dlw = upload_layer(a, lh, "L");
        std::vector<float> x(static_cast<std::size_t>(t) * kC);
        std::vector<float> pos(static_cast<std::size_t>(p) * kC);
        fill_random(x);
        diar::relpos_table_forward(pos.data(), t, kC);
        float* dx = h2d(x, a, "x");
        float* dpos = h2d(pos, a, "pos");
        float* dy = a.alloc("y", static_cast<std::size_t>(t) * kC);
        const std::size_t ws_n =
            conformer_layer_scratch_floats(t, kC, kF, kH);
        report("layer_scratch", "\"floats\":" + std::to_string(ws_n) + "}\n");
        float* dws = a.alloc("ws", ws_n);
        gpu_conformer_layer(cublas, stream, kBlock, dlw, dx, dpos, dy, dws, t,
                            kC, kF, kH);
        CUDA4_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> got(static_cast<std::size_t>(t) * kC);
        CUDA4_CHECK(cudaMemcpy(got.data(), dy, got.size() * 4,
                               cudaMemcpyDeviceToHost));
        std::vector<float> ref(got.size());
        const diar::ConformerLayerWeights hw = lh.host_ref();
        diar::conformer_layer_forward(x.data(), pos.data(), hw, ref.data(), t,
                                      kC, kF, kH, kK);
        const double m = max_abs(ref, got);
        report("parity_layer", "\"max_abs\":" + fmt9(m) +
               ",\"gate\":" + std::to_string(kLayerGate) +
               ",\"pass\":" + (m <= kLayerGate ? "true" : "false") + "}\n");
    }

    // ---- G-S4d: 17-layer device-resident timeline ------------------------
    // Per-layer weight floats (exact): LN 10c | FF 4*(d_ff*c) + 4*d_ff + 4c
    // | MHA 5c^2 + 6c | conv 3c^2 + c*k + 8c. t=20 is the production chunk
    // geometry (224 chunks / 357.4s at 80ms framing); t=26 is a stress point.
    if (!parity_only) {
        const std::size_t per_layer =
            10u * kC + 4u * static_cast<std::size_t>(kF) * kC + 4u * kF +
            4u * kC + 5u * kC * kC + 6u * kC + 3u * kC * kC +
            static_cast<std::size_t>(kC) * kK + 8u * kC;
        diar::backend::GpuArena a;
        a.init(kTimelineLayers * per_layer + 8u * 1024u * 1024u);
        std::vector<diar::backend::LayerDevWeights> layers;
        for (int i = 0; i < kTimelineLayers; ++i) {
            LayerHost lh;
            lh.build(kC, kF, kK);
            layers.push_back(upload_layer(a, lh, "T" + std::to_string(i)));
        }
        for (const int t : {20, 26}) {
            const int p = 2 * t - 1;
            const std::size_t tc = static_cast<std::size_t>(t) * kC;
            std::vector<float> x(tc);
            std::vector<float> pos(static_cast<std::size_t>(p) * kC);
            fill_random(x);
            diar::relpos_table_forward(pos.data(), t, kC);
            float* dpos = h2d(pos, a, "pos" + std::to_string(t));
            float* dx = h2d(x, a, "x" + std::to_string(t));
            float* dy = a.alloc("y" + std::to_string(t), tc);
            float* dws = a.alloc("ws" + std::to_string(t),
                                 conformer_layer_scratch_floats(t, kC, kF, kH));
            // alternate y/x as input/output across layers (per-chunk chain)
            cudaEvent_t e0, e1;
            CUDA4_CHECK(cudaEventCreate(&e0));
            CUDA4_CHECK(cudaEventCreate(&e1));
            const int kWarm = 3;
            for (int c = 0; c < kWarm + kTimelineChunks; ++c) {
                if (c == kWarm) CUDA4_CHECK(cudaEventRecord(e0, stream));
                float* in = dx;
                float* out = dy;
                for (int l = 0; l < kTimelineLayers; ++l) {
                    gpu_conformer_layer(cublas, stream, kBlock, layers[l], in,
                                        dpos, out, dws, t, kC, kF, kH);
                    std::swap(in, out);
                }
            }
            CUDA4_CHECK(cudaEventRecord(e1, stream));
            CUDA4_CHECK(cudaEventSynchronize(e1));
            float total_ms = 0.0f;
            CUDA4_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));
            const double per_chunk = total_ms / kTimelineChunks;
            report("timeline",
                   "\"layers\":" + std::to_string(kTimelineLayers) +
                       ",\"t\":" + std::to_string(t) +
                       ",\"ms_per_chunk\":" + fmt9(per_chunk) +
                       ",\"us_per_layer\":" +
                       fmt9(per_chunk * 1000.0 / kTimelineLayers) + "}\n");
            CUDA4_CHECK(cudaEventDestroy(e0));
            CUDA4_CHECK(cudaEventDestroy(e1));
        }
    }

    report("verdict", "\"mode\":\"gpu-bench-complete\"}\n");
    return 0;
}

#else
// ============================ CPU selftest =================================
// Reference-chain sanity: every CPU function the GPU gates compare against
// must run and produce finite, sane values; naive-oracle cross-checks pin
// the reference functions themselves (M2 tests own the deep oracle work).
namespace {

void expect(bool ok, const char* what) {
    if (!ok) {
        report("selftest_fail", std::string("\"what\":\"") + what + "\"}\n");
        std::exit(1);
    }
}

}  // namespace

int main() {
    report("env", "\"name\":\"cpu-selftest\"}\n");
    const int t = 8, p = 2 * t - 1;
    // MHA reference: finite + row-stochastic internals exercised via layer
    LayerHost lh;
    lh.build(kC, kF, kK);
    std::vector<float> x(static_cast<std::size_t>(t) * kC);
    std::vector<float> pos(static_cast<std::size_t>(p) * kC);
    fill_random(x);
    diar::relpos_table_forward(pos.data(), t, kC);
    std::vector<float> y(x.size());
    const diar::ConformerLayerWeights hw = lh.host_ref();
    diar::conformer_layer_forward(x.data(), pos.data(), hw, y.data(), t, kC,
                                  kF, kH, kK);
    double peak = 0.0;
    for (float v : y) {
        expect(std::isfinite(v), "layer output finite");
        peak = std::max(peak, static_cast<double>(std::fabs(v)));
    }
    expect(peak > 1e-6, "layer output non-degenerate");
    // softmax row-sum property through the same reference the GPU gates use
    std::vector<float> sm(4 * 8), sm_ref(4 * 8);
    fill_random(sm);
    nn::softmax_last_dim(sm.data(), sm_ref.data(), 4, 8);
    for (int r = 0; r < 4; ++r) {
        double s = 0.0;
        for (int c = 0; c < 8; ++c)
            s += sm_ref[static_cast<std::size_t>(r) * 8 + c];
        expect(std::fabs(s - 1.0) < 1e-5, "softmax rows sum to 1");
    }
    // glu oracle: a * sigmoid(b), a = first half
    {
        std::vector<float> gx(2 * 5 * 3), gy(5 * 3);
        fill_random(gx);
        nn::glu_forward(gx.data(), gy.data(), 5, 3);
        for (int i = 0; i < 15; ++i) {
            const int r = i / 3, ci = i - r * 3;
            const float a = gx[static_cast<std::size_t>(r) * 6 + ci];
            const float b = gx[static_cast<std::size_t>(r) * 6 + 3 + ci];
            const float want = a / (1.0f + std::exp(-b));
            expect(std::fabs(want - gy[i]) < 1e-6, "glu naive oracle");
        }
    }
    // dwconv naive oracle (symmetric pad) on a tiny case
    {
        const int tt = 5, cc = 2, kk = 3;
        std::vector<float> cx{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<float> cw{0.5f, 1.0f, -0.5f, 0.25f, -1.0f, 2.0f};
        std::vector<float> cb{0.1f, -0.1f};
        std::vector<float> cy(tt * cc);
        nn::conv1d_depthwise_forward(cx.data(), cw.data(), cb.data(),
                                     cy.data(), tt, cc, kk, 1, 1);
        // hand-computed: out[ti,ci] = sum_j x[ti+j-1,ci]*w[ci*3+j], pad zero
        const float want[10] = {-0.400000f, 5.900000f, 1.100000f, 8.400000f, 3.100000f, 10.900000f, 5.100000f, 13.400000f, 12.600000f, -8.100000f};
        for (int i = 0; i < 10; ++i)
            expect(std::fabs(cy[i] - want[i]) < 1e-5, "dwconv naive oracle");
    }
    report("verdict", "\"mode\":\"cpu-selftest-complete\"}\n");
    return 0;
}

#endif  // DIAR_WITH_CUDA
