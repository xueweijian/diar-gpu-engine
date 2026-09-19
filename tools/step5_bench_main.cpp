// tools/step5_bench_main.cpp — Step 5 bench harness (fp16 weight storage).
//
// Derived from step4_bench_main.cpp: every G-S4 gate stays (regression —
// the layer bodies are templates now, so the fp32 route must reproduce the
// Step-4 numbers bit-for-bit) plus the G-S5 family on the fp16-storage
// route (GemmEx 16F-in / 32F-acc, weights packed RNE on host):
//   G-S5a  device fp32->fp16 cast bit-identity vs host float_to_half
//   G-S5b  fp16 linear parity (5 engine shapes) vs CPU fp32 oracle
//          (+ hq row: same but CPU ref uses dequantized-half weights,
//           isolating activation-cast + accumulation from weight quant)
//   G-S5c  conformer layer fp16 parity (t=20)
//   G-S5d  transformer block fp16 parity (t=12/20/26)
//   G-S5e  GEMM p50 fp32 vs fp16 per shape (bandwidth evidence)
//   G-S5f  fp16 timelines: conformer x17 and full chain (conf+proj+tf),
//          compared against the fp32 timelines by the framework gate
// Same two-mode pattern as step3/step4_bench_main.cpp:
//  - CPU selftest (no CUDA): reference-chain sanity + naive-oracle cross
//    checks so a Kaggle round-trip never burns on harness bugs.
//  - GPU bench (DIAR_WITH_CUDA, nvcc):
//      G-S4a  op parity: softmax / glu / dwconv+BN+SiLU  <= 1e-5
//      G-S4b  rel-pos MHA parity (t=12/20/26)            <= 2*parity_gate(512)
//      G-S4b2 raw GEMM baseline at k=512 (gate evidence)  <= parity_gate(512)
//      G-S4c  full conformer layer parity (t=20)         <= 2e-4
//      G-S4e  transformer block parity (t=12/20/26)      <= 2*parity_gate(768)
//      G-S4d  17-layer conformer timeline (ms/chunk) — decision input
//      G-S4d2 18-layer transformer timeline
//      G-S4d3 full-encoder timeline (conf x17 + proj + tf x18) vs G-D gate
//  Reports [s5] JSON lines. All Step-4 chains run device-resident (GpuArena,
//  zero H2D per chunk) — this is the production shape, not the Context
//  per-call-copy path.
#include "diar/backend_cuda.hpp"
#include "diar/conformer.hpp"
#include "diar/conv.hpp"
#include "diar/cublas_layout.hpp"
#include "diar/layers.hpp"
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
using diar::backend::bench_cast_f32_f16;
using diar::backend::bench_dwconv_bn_silu;
using diar::backend::bench_glu;
using diar::backend::bench_linear_h;
using diar::backend::bench_softmax_rows;
using diar::backend::conformer_layer_scratch_floats;
using diar::backend::gpu_conformer_layer;
using diar::backend::gpu_relpos_mha;
using diar::backend::gpu_transformer_block;
using diar::backend::mha_scratch_floats;
using diar::backend::transformer_block_scratch_floats;
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
// transformer stack (plan §3 item 2): H=192, inner=768, 8 heads, 18 layers
constexpr int kTH = 192;
constexpr int kTI = 768;
constexpr int kTHeads = 8;
constexpr int kTimelineTfLayers = 18;

constexpr double kOpGate = 1e-5;

// Cross-implementation gate (CPU naive vs cuBLAS FMA/split-k blocks), same
// k-scaling rule as step3: 2e-7 per k-unit, floor 1e-5, cap 5e-4. Measured
// on T4 (step3 v5): 2-5e-5 at k=512, 1.3e-4 at k=2048.
inline double parity_gate(int k) {
    return std::min(5e-4, std::max(1e-5, 2e-7 * static_cast<double>(k)));
}
// MHA chain = q/k/v/pos proj (k=512) -> head GEMMs (k=64) -> rel-shift
// softmax -> ctx -> out proj (k=512): two chained k=512 tiers. Gate set to
// 2 * parity_gate(512) with measured 7.3-8.1e-5 (SASS == PTX bit-identical,
// so the residual is genuine reduction-order noise, not a codegen or
// indexing effect; the layer gate below covers the same MHA in context).
const double kMhaGate = 2.0 * parity_gate(512);
constexpr double kLayerGate = 2e-4;
// Transformer block: q/k/v/o proj (k=192) -> plain attention (T^2 softmax
// chain, same tier argument as the rel-pos MHA) -> FF (k=768 up / k=768
// down reduction via inner=768). Deepest GEMM tier is 768; gate mirrors
// the MHA pattern at that tier.
const double kTfGate = 2.0 * parity_gate(768);

// ---- Step 5 (fp16-storage route) gate tiers -------------------------------
// Unlike the fp32 tiers, the fp16 parity error is dominated by INPUT
// QUANTIZATION (2^-11 RNE on the weights once at upload + on every GEMM
// activation on device), accumulated over k products as a random walk:
// measured on T4 (kernel v1) 1.2e-3 @k=192, 2.0-2.4e-3 @k=512,
// 2.2e-3 @k=768, 3.7e-3 @k=2048 — i.e. ~1.5e-4*sqrt(k), NOT linear in k.
// Gate tier 1.5e-4*sqrt(k) with floor 3e-3 (>=1.45x margin at every
// engine shape). Recalibrated from v1 measurements (Step-4 v2 precedent);
// the K6 four-fixture gate stays the production arbiter for shipping this
// route as default.
inline double fp16_gate(int k) {
    return std::min(4e-2, std::max(3e-3, 1.5e-4 * std::sqrt(static_cast<double>(k))));
}
// Layer-chain tiers: LN renormalization resets the residual scale each
// block, so the chain stays near the linear tier; generous first run,
// recalibrate after v1 measurements.
constexpr double kLayerHGate = 5e-2;
constexpr double kTfHGate = 2e-2;
struct HShape { const char* name; int t, in, out; };
const HShape kHShapes[] = {
    {"ff_up_512x2048", 20, kC, kF},   {"ff_down_2048x512", 20, kF, kC},
    {"proj_512x512", 20, kC, kC},     {"tf_f1_192x768", 20, kTH, kTI},
    {"tf_f2_768x192", 20, kTI, kTH}};

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
    std::printf("[s5] {\"k\":\"%s\",%s", key.c_str(), json.c_str());
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

struct TfHost {
    std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    std::vector<float> ln1_g, ln1_b, ln2_g, ln2_b;
    std::vector<float> f1_w, f1_b, f2_w, f2_b;
    void build(int h, int inner) {
        auto mk = [this](std::vector<float>& v, std::size_t n) {
            v.resize(n);
            fill_random(v);
        };
        mk(q_w, static_cast<std::size_t>(h) * h); mk(q_b, h);
        mk(k_w, static_cast<std::size_t>(h) * h); mk(k_b, h);
        mk(v_w, static_cast<std::size_t>(h) * h); mk(v_b, h);
        mk(o_w, static_cast<std::size_t>(h) * h); mk(o_b, h);
        ln1_g.assign(h, 1.0f); ln1_b.assign(h, 0.0f);
        ln2_g.assign(h, 1.0f); ln2_b.assign(h, 0.0f);
        mk(f1_w, static_cast<std::size_t>(inner) * h); mk(f1_b, inner);
        mk(f2_w, static_cast<std::size_t>(h) * inner); mk(f2_b, h);
    }
    diar::TransformerBlockWeights host_ref() const {
        return diar::TransformerBlockWeights{
            q_w.data(), q_b.data(), k_w.data(), k_b.data(),
            v_w.data(), v_b.data(), o_w.data(), o_b.data(),
            ln1_g.data(), ln1_b.data(), ln2_g.data(), ln2_b.data(),
            f1_w.data(), f1_b.data(), f2_w.data(), f2_b.data()};
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
            std::printf("[s5] {\"k\":\"fatal\",\"what\":\"%s\",\"code\":%d}\n", \
                        #expr, static_cast<int>(_c));                       \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

#define CUBLAS4_CHECK(expr)                                                 \
    do {                                                                    \
        const cublasStatus_t _c = (expr);                                   \
        if (_c != CUBLAS_STATUS_SUCCESS) {                                  \
            std::printf("[s5] {\"k\":\"fatal\",\"what\":\"%s\",\"code\":%d}\n", \
                        #expr, static_cast<int>(_c));                       \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

float* h2d(const std::vector<float>& v, diar::backend::GpuArena& arena,
              const std::string& name) {
    float* d = arena.alloc(name.c_str(), v.size());
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

diar::backend::TransformerDevWeights upload_tf(
    diar::backend::GpuArena& arena, const TfHost& h, const std::string& tag) {
    diar::backend::TransformerDevWeights w{};
    auto up = [&](const std::vector<float>& v, const char* n) {
        return h2d(v, arena, (tag + n).c_str());
    };
    w.q_w = up(h.q_w, ".qw"); w.q_b = up(h.q_b, ".qb");
    w.k_w = up(h.k_w, ".kw"); w.k_b = up(h.k_b, ".kb");
    w.v_w = up(h.v_w, ".vw"); w.v_b = up(h.v_b, ".vb");
    w.o_w = up(h.o_w, ".ow"); w.o_b = up(h.o_b, ".ob");
    w.ln1_g = up(h.ln1_g, ".l1g"); w.ln1_b = up(h.ln1_b, ".l1b");
    w.ln2_g = up(h.ln2_g, ".l2g"); w.ln2_b = up(h.ln2_b, ".l2b");
    w.f1_w = up(h.f1_w, ".f1w"); w.f1_b = up(h.f1_b, ".f1b");
    w.f2_w = up(h.f2_w, ".f2w"); w.f2_b = up(h.f2_b, ".f2b");
    return w;
}

double median_of(std::vector<double>& v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// host-pack a fp32 weight vector to fp16 (RNE via cublas_layout, the same
// packer the production upload path uses) and upload; element count n maps
// to (n+1)/2 arena floats.
__half* h2h(const std::vector<float>& v, diar::backend::GpuArena& arena,
            const std::string& name) {
    std::vector<std::uint16_t> packed(v.size());
    diar::pack_weights_f16(v.data(), packed.data(),
                           static_cast<int>(v.size()), 1);
    float* d = arena.alloc(name.c_str(), (v.size() + 1) / 2);
    CUDA4_CHECK(cudaMemcpy(d, packed.data(), v.size() * 2,
                           cudaMemcpyHostToDevice));
    return reinterpret_cast<__half*>(d);
}

// fp16-storage mirror uploads: GEMM weights via h2h, everything that feeds
// pointwise kernels (biases, LN, BN, dw conv) stays fp32.
diar::backend::LayerDevWeightsH upload_layer_h(
    diar::backend::GpuArena& arena, const LayerHost& h,
    const std::string& tag) {
    diar::backend::LayerDevWeightsH w{};
    auto f = [&](const std::vector<float>& v, const char* n) {
        return h2d(v, arena, (tag + n).c_str());
    };
    auto hh = [&](const std::vector<float>& v, const char* n) {
        return h2h(v, arena, (tag + n).c_str());
    };
    w.n_ff1_g = f(h.n_g[0], ".ng0"); w.n_ff1_b = f(h.n_b[0], ".nb0");
    w.n_sa_g = f(h.n_g[1], ".ng1");  w.n_sa_b = f(h.n_b[1], ".nb1");
    w.n_conv_g = f(h.n_g[2], ".ng2"); w.n_conv_b = f(h.n_b[2], ".nb2");
    w.n_ff2_g = f(h.n_g[3], ".ng3"); w.n_ff2_b = f(h.n_b[3], ".nb3");
    w.n_out_g = f(h.n_g[4], ".ng4"); w.n_out_b = f(h.n_b[4], ".nb4");
    w.ff1_w1 = hh(h.ff1_w1, ".f1w1"); w.ff1_b1 = f(h.ff1_b1, ".f1b1");
    w.ff1_w2 = hh(h.ff1_w2, ".f1w2"); w.ff1_b2 = f(h.ff1_b2, ".f1b2");
    w.ff2_w1 = hh(h.ff2_w1, ".f2w1"); w.ff2_b1 = f(h.ff2_b1, ".f2b1");
    w.ff2_w2 = hh(h.ff2_w2, ".f2w2"); w.ff2_b2 = f(h.ff2_b2, ".f2b2");
    w.attn.q_w = hh(h.mha.q_w, ".qw"); w.attn.q_b = f(h.mha.q_b, ".qb");
    w.attn.k_w = hh(h.mha.k_w, ".kw"); w.attn.k_b = f(h.mha.k_b, ".kb");
    w.attn.v_w = hh(h.mha.v_w, ".vw"); w.attn.v_b = f(h.mha.v_b, ".vb");
    w.attn.pos_w = hh(h.mha.pos_w, ".pw");
    w.attn.bu = f(h.mha.bu, ".bu"); w.attn.bv = f(h.mha.bv, ".bv");
    w.attn.out_w = hh(h.mha.out_w, ".ow");
    w.attn.out_b = f(h.mha.out_b, ".ob");
    w.conv.pw1_w = hh(h.conv.pw1_w, ".c1w");
    w.conv.pw1_b = f(h.conv.pw1_b, ".c1b");
    w.conv.dw_w = f(h.conv.dw_w, ".dw"); w.conv.dw_b = f(h.conv.dw_b, ".db");
    w.conv.bn_w = f(h.conv.bn_w, ".bw"); w.conv.bn_b = f(h.conv.bn_b, ".bb");
    w.conv.bn_mean = f(h.conv.bn_mean, ".bm");
    w.conv.bn_var = f(h.conv.bn_var, ".bv2");
    w.conv.pw2_w = hh(h.conv.pw2_w, ".c2w");
    w.conv.pw2_b = f(h.conv.pw2_b, ".c2b");
    return w;
}

diar::backend::TransformerDevWeightsH upload_tf_h(
    diar::backend::GpuArena& arena, const TfHost& h, const std::string& tag) {
    diar::backend::TransformerDevWeightsH w{};
    auto f = [&](const std::vector<float>& v, const char* n) {
        return h2d(v, arena, (tag + n).c_str());
    };
    auto hh = [&](const std::vector<float>& v, const char* n) {
        return h2h(v, arena, (tag + n).c_str());
    };
    w.q_w = hh(h.q_w, ".qw"); w.q_b = f(h.q_b, ".qb");
    w.k_w = hh(h.k_w, ".kw"); w.k_b = f(h.k_b, ".kb");
    w.v_w = hh(h.v_w, ".vw"); w.v_b = f(h.v_b, ".vb");
    w.o_w = hh(h.o_w, ".ow"); w.o_b = f(h.o_b, ".ob");
    w.ln1_g = f(h.ln1_g, ".l1g"); w.ln1_b = f(h.ln1_b, ".l1b");
    w.ln2_g = f(h.ln2_g, ".l2g"); w.ln2_b = f(h.ln2_b, ".l2b");
    w.f1_w = hh(h.f1_w, ".f1w"); w.f1_b = f(h.f1_b, ".f1b");
    w.f2_w = hh(h.f2_w, ".f2w"); w.f2_b = f(h.f2_b, ".f2b");
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    const bool parity_only =
        argc > 1 && std::string(argv[1]) == "--parity-only";
    CUDA4_CHECK(cudaSetDevice(0));
    report("env", std::string("\"name\":\"") +
                      diar::backend::bench_device_name() + "\",\"cc\":" +
                      std::to_string(diar::backend::bench_device_cc()) + "}\n");

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

    // ---- G-S4b2: raw GEMM baseline (evidence for the MHA gate tier) ------
    // Measures CPU-sequential vs cuBLAS reduction-order noise at the same
    // reduction depth as the MHA projections. The MHA gate must sit above
    // this; identical values across SASS/PTX runs confirm codegen is not a
    // factor.
    for (const int out_dim : {512, 2048}) {
        const int in_dim = 512, t = 20;
        std::vector<float> x(static_cast<std::size_t>(t) * in_dim);
        std::vector<float> w(static_cast<std::size_t>(out_dim) * in_dim);
        std::vector<float> b(out_dim);
        std::vector<float> ref(static_cast<std::size_t>(t) * out_dim);
        fill_random(x);
        fill_random(w);
        fill_random(b);
        nn::linear_forward(x.data(), w.data(), b.data(), ref.data(), t,
                           in_dim, out_dim);
        diar::backend::GpuArena a;
        a.init(8u * 1024u * 1024u);
        float* dx = h2d(x, a, "lx");
        float* dw = h2d(w, a, "lw");
        float* db = h2d(b, a, "lb");
        float* dy = a.alloc("ly", (static_cast<std::size_t>(t)) * out_dim);
        diar::backend::bench_linear(cublas, stream, kBlock, dx, dw, db, dy, t,
                                    in_dim, out_dim);
        CUDA4_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> got(static_cast<std::size_t>(t) * out_dim);
        CUDA4_CHECK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
        const double m = max_abs(ref, got);
        report("parity_linear",
               "\"t\":" + std::to_string(t) +
                   ",\"in\":" + std::to_string(in_dim) +
                   ",\"out\":" + std::to_string(out_dim) +
                   ",\"max_abs\":" + fmt9(m) +
                   ",\"gate\":" + fmt9(parity_gate(in_dim)) +
                   ",\"pass\":" +
                   (m <= parity_gate(in_dim) ? "true" : "false") + "}\n");
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

    // ---- G-S4e: transformer block parity (t=12/20/26) ---------------------
    // (in --parity-only mode too: the PTX-JIT pass must gate every layer
    // family the production chain runs)
    {
        for (const TShape& s : kTShapes) {
            const int t = s.t;
            TfHost th;
            th.build(kTH, kTI);
            diar::backend::GpuArena a;
            a.init(8u * 1024u * 1024u);
            diar::backend::TransformerDevWeights dw = [&] {
                diar::backend::TransformerDevWeights w{};
                auto up = [&](const std::vector<float>& v, const char* n) {
                    return h2d(v, a, n);
                };
                w.q_w = up(th.q_w, "qw"); w.q_b = up(th.q_b, "qb");
                w.k_w = up(th.k_w, "kw"); w.k_b = up(th.k_b, "kb");
                w.v_w = up(th.v_w, "vw"); w.v_b = up(th.v_b, "vb");
                w.o_w = up(th.o_w, "ow"); w.o_b = up(th.o_b, "ob");
                w.ln1_g = up(th.ln1_g, "l1g"); w.ln1_b = up(th.ln1_b, "l1b");
                w.ln2_g = up(th.ln2_g, "l2g"); w.ln2_b = up(th.ln2_b, "l2b");
                w.f1_w = up(th.f1_w, "f1w"); w.f1_b = up(th.f1_b, "f1b");
                w.f2_w = up(th.f2_w, "f2w"); w.f2_b = up(th.f2_b, "f2b");
                return w;
            }();
            std::vector<float> x(static_cast<std::size_t>(t) * kTH);
            fill_random(x);
            float* dx = h2d(x, a, "x");
            float* dy = a.alloc("y", static_cast<std::size_t>(t) * kTH);
            const std::size_t ws_n =
                transformer_block_scratch_floats(t, kTH, kTI, kTHeads);
            float* dws = a.alloc("ws", ws_n);
            gpu_transformer_block(cublas, stream, kBlock, dw, dx, dy, dws, t,
                                  kTH, kTI, kTHeads);
            CUDA4_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> got(static_cast<std::size_t>(t) * kTH);
            CUDA4_CHECK(cudaMemcpy(got.data(), dy, got.size() * 4,
                                   cudaMemcpyDeviceToHost));
            std::vector<float> ref(got.size());
            diar::transformer_block_forward(x.data(), th.host_ref(),
                                            ref.data(), t, kTH, kTI, kTHeads);
            const double m = max_abs(ref, got);
            report("parity_tf", std::string("\"shape\":\"") + s.name +
                   "\",\"max_abs\":" + fmt9(m) +
                   ",\"gate\":" + fmt9(kTfGate) +
                   ",\"pass\":" + (m <= kTfGate ? "true" : "false") + "}\n");
        }
    }

    // ---- G-S5a: device fp32->fp16 cast bit-identity -----------------------
    // The device cast kernel must reproduce the host packer (RNE) exactly;
    // any drift here poisons every downstream fp16 parity number.
    {
        const std::size_t n = 1u << 20;
        std::vector<float> x(n);
        fill_random(x);
        for (std::size_t i = 0; i < 64; ++i) x[i] *= 1000.0f;  // range tails
        x[0] = 0.0f; x[1] = -0.0f; x[2] = 1e-8f; x[3] = 65504.0f;
        float* dx;
        __half* dh;
        CUDA4_CHECK(cudaMalloc(&dx, n * 4));
        CUDA4_CHECK(cudaMalloc(&dh, n * 2));
        CUDA4_CHECK(cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice));
        bench_cast_f32_f16(dx, dh, n, stream, kBlock);
        CUDA4_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> got(n);
        CUDA4_CHECK(cudaMemcpy(got.data(), dh, n * 2, cudaMemcpyDeviceToHost));
        std::size_t diff = 0;
        for (std::size_t i = 0; i < n; ++i)
            diff += got[i] != diar::float_to_half(x[i]);
        report("parity_cast",
               "\"n\":" + std::to_string(n) + ",\"diff\":" +
                   std::to_string(diff) +
                   ",\"pass\":" + (diff == 0 ? "true" : "false") + "}\n");
        cudaFree(dx);
        cudaFree(dh);
    }

    // ---- G-S5b: fp16 linear parity (5 engine shapes) ----------------------
    // Row 1 (parity_linear_h): GPU fp16 route vs CPU fp32 with the ORIGINAL
    //   fp32 weights — the real divergence a production switch introduces.
    // Row 2 (parity_linear_hq): CPU ref with dequantized-half weights —
    //   isolates activation-cast + accumulation noise from weight quant.
    for (const HShape& s : kHShapes) {
        std::vector<float> x(static_cast<std::size_t>(s.t) * s.in);
        std::vector<float> w(static_cast<std::size_t>(s.out) * s.in);
        std::vector<float> b(s.out);
        fill_random(x);
        fill_random(w);
        fill_random(b);
        std::vector<float> ref(static_cast<std::size_t>(s.t) * s.out);
        nn::linear_forward(x.data(), w.data(), b.data(), ref.data(), s.t,
                           s.in, s.out);
        std::vector<std::uint16_t> wh(w.size());
        diar::pack_weights_f16(w.data(), wh.data(), s.out, s.in);
        std::vector<float> wd(w.size());
        diar::unpack_weights_f16(wh.data(), wd.data(), s.out, s.in);
        std::vector<float> refq(ref.size());
        nn::linear_forward(x.data(), wd.data(), b.data(), refq.data(), s.t,
                           s.in, s.out);
        diar::backend::GpuArena a;
        a.init(64u * 1024u * 1024u);
        float* dx = h2d(x, a, "x");
        __half* dwh = h2h(w, a, "w");
        float* db = h2d(b, a, "b");
        float* dy = a.alloc("y", static_cast<std::size_t>(s.t) * s.out);
        __half* cast16 = reinterpret_cast<__half*>(
            a.alloc("c16", (static_cast<std::size_t>(s.t) * s.in + 1) / 2));
        bench_linear_h(cublas, stream, kBlock, dx, dwh, db, dy, s.t, s.in,
                       s.out, cast16);
        CUDA4_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> got(ref.size());
        CUDA4_CHECK(cudaMemcpy(got.data(), dy, got.size() * 4,
                               cudaMemcpyDeviceToHost));
        const double m = max_abs(ref, got);
        const double mq = max_abs(refq, got);
        const double g = fp16_gate(s.in);
        report("parity_linear_h",
               std::string("\"shape\":\"") + s.name +
                   "\",\"max_abs\":" + fmt9(m) + ",\"gate\":" + fmt9(g) +
                   ",\"pass\":" + (m <= g ? "true" : "false") + "}\n");
        const double gq = std::max(1.5e-3, 0.7 * g);  // measured hq ~0.7x
        report("parity_linear_hq",
               std::string("\"shape\":\"") + s.name +
                   "\",\"max_abs\":" + fmt9(mq) + ",\"gate\":" + fmt9(gq) +
                   ",\"pass\":" + (mq <= gq ? "true" : "false") + "}\n");
    }

    // ---- G-S5c: conformer layer fp16 parity (t=20) ------------------------
    {
        const int t = kLayerParityT, p = 2 * t - 1;
        LayerHost lh;
        lh.build(kC, kF, kK);
        diar::backend::GpuArena a;
        a.init(64u * 1024u * 1024u);
        diar::backend::LayerDevWeightsH dlw = upload_layer_h(a, lh, "HL");
        std::vector<float> x(static_cast<std::size_t>(t) * kC);
        std::vector<float> pos(static_cast<std::size_t>(p) * kC);
        fill_random(x);
        diar::relpos_table_forward(pos.data(), t, kC);
        float* dx = h2d(x, a, "x");
        float* dpos = h2d(pos, a, "pos");
        float* dy = a.alloc("y", static_cast<std::size_t>(t) * kC);
        const std::size_t ws_n = conformer_layer_scratch_floats(t, kC, kF, kH);
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
        report("parity_layer_h",
               "\"max_abs\":" + fmt9(m) + ",\"gate\":" + fmt9(kLayerHGate) +
                   ",\"pass\":" + (m <= kLayerHGate ? "true" : "false") +
                   "}\n");
    }

    // ---- G-S5d: transformer block fp16 parity (t=12/20/26) ----------------
    {
        for (const TShape& s : kTShapes) {
            const int t = s.t;
            TfHost th;
            th.build(kTH, kTI);
            diar::backend::GpuArena a;
            a.init(16u * 1024u * 1024u);
            diar::backend::TransformerDevWeightsH dw = upload_tf_h(a, th,
                                                                   "HU");
            std::vector<float> x(static_cast<std::size_t>(t) * kTH);
            fill_random(x);
            float* dx = h2d(x, a, "x");
            float* dy = a.alloc("y", static_cast<std::size_t>(t) * kTH);
            const std::size_t ws_n =
                transformer_block_scratch_floats(t, kTH, kTI, kTHeads);
            float* dws = a.alloc("ws", ws_n);
            gpu_transformer_block(cublas, stream, kBlock, dw, dx, dy, dws, t,
                                  kTH, kTI, kTHeads);
            CUDA4_CHECK(cudaStreamSynchronize(stream));
            std::vector<float> got(static_cast<std::size_t>(t) * kTH);
            CUDA4_CHECK(cudaMemcpy(got.data(), dy, got.size() * 4,
                                   cudaMemcpyDeviceToHost));
            std::vector<float> ref(got.size());
            diar::transformer_block_forward(x.data(), th.host_ref(),
                                            ref.data(), t, kTH, kTI, kTHeads);
            const double m = max_abs(ref, got);
            report("parity_tf_h",
                   std::string("\"shape\":\"") + s.name +
                       "\",\"max_abs\":" + fmt9(m) +
                       ",\"gate\":" + fmt9(kTfHGate) + ",\"pass\":" +
                       (m <= kTfHGate ? "true" : "false") + "}\n");
        }
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
        struct TState {
            int t;
            float *dpos, *dx, *dy, *dws;
        };
        std::vector<TState> states;
        for (const int t : {20, 26}) {
            const int p = 2 * t - 1;
            const std::size_t tc = static_cast<std::size_t>(t) * kC;
            std::vector<float> x(tc);
            std::vector<float> pos(static_cast<std::size_t>(p) * kC);
            fill_random(x);
            diar::relpos_table_forward(pos.data(), t, kC);
            const std::string tag = std::to_string(t);
            states.push_back(TState{t, h2d(pos, a, "pos" + tag),
                                    h2d(x, a, "x" + tag),
                                    a.alloc(("y" + tag).c_str(), tc),
                                    a.alloc(("ws" + tag).c_str(),
                                            conformer_layer_scratch_floats(
                                                t, kC, kF, kH))});
        }
        // per-chunk chain = 17 layers, alternating input buffers
        auto run_chain = [&](const TState& st) {
            float* in = st.dx;
            float* out = st.dy;
            for (int l = 0; l < kTimelineLayers; ++l) {
                gpu_conformer_layer(cublas, stream, kBlock, layers[l], in,
                                    st.dpos, out, st.dws, st.t, kC, kF, kH);
                std::swap(in, out);
            }
        };
        // symmetric warmup for BOTH configs first — v1 timed t=20 cold
        // (first cuBLAS autotune + clock ramp) and it read 35% slower than
        // t=26 which was plain wrong.
        for (const TState& st : states)
            for (int c = 0; c < 8; ++c) run_chain(st);
        for (const TState& st : states) {
            cudaEvent_t e0, e1;
            CUDA4_CHECK(cudaEventCreate(&e0));
            CUDA4_CHECK(cudaEventCreate(&e1));
            CUDA4_CHECK(cudaEventRecord(e0, stream));
            for (int c = 0; c < kTimelineChunks; ++c) run_chain(st);
            CUDA4_CHECK(cudaEventRecord(e1, stream));
            CUDA4_CHECK(cudaEventSynchronize(e1));
            float total_ms = 0.0f;
            CUDA4_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));
            const double per_chunk = total_ms / kTimelineChunks;
            report("timeline",
                   "\"layers\":" + std::to_string(kTimelineLayers) +
                       ",\"t\":" + std::to_string(st.t) +
                       ",\"ms_per_chunk\":" + fmt9(per_chunk) +
                       ",\"us_per_layer\":" +
                       fmt9(per_chunk * 1000.0 / kTimelineLayers) + "}\n");
            CUDA4_CHECK(cudaEventDestroy(e0));
            CUDA4_CHECK(cudaEventDestroy(e1));
        }
    }

    // ---- G-S4d2/G-S4d3: transformer stack + full-encoder timelines --------
    // G-S4d2: 18 transformer layers alone. G-S4d3: the production chunk
    // chain end-to-end — conformer x17 -> encoder proj (512->192) ->
    // transformer x18, all device-resident. This is the number the G-D
    // gate (< 25 ms/chunk, stretch < 20) is judged against.
    if (!parity_only) {
        const std::size_t per_layer =
            10u * kC + 4u * static_cast<std::size_t>(kF) * kC + 4u * kF +
            4u * kC + 5u * kC * kC + 6u * kC + 3u * kC * kC +
            static_cast<std::size_t>(kC) * kK + 8u * kC;
        const std::size_t tf_per_layer =
            4u * static_cast<std::size_t>(kTH) * kTH + 4u * kTH + 4u * kTH +
            static_cast<std::size_t>(kTI) * kTH + kTI +
            static_cast<std::size_t>(kTH) * kTI + kTH;
        diar::backend::GpuArena a;
        a.init(kTimelineLayers * per_layer +
               kTimelineTfLayers * tf_per_layer +
               static_cast<std::size_t>(kTH) * kC + kTH +  // proj w+b
               16u * 1024u * 1024u);                       // activations+ws
        std::vector<diar::backend::LayerDevWeights> conf;
        for (int i = 0; i < kTimelineLayers; ++i) {
            LayerHost lh;
            lh.build(kC, kF, kK);
            conf.push_back(upload_layer(a, lh, "F" + std::to_string(i)));
        }
        std::vector<diar::backend::TransformerDevWeights> tf;
        for (int i = 0; i < kTimelineTfLayers; ++i) {
            TfHost th;
            th.build(kTH, kTI);
            tf.push_back(upload_tf(a, th, "U" + std::to_string(i)));
        }
        // encoder proj [kTH,kC] + bias
        std::vector<float> proj_w(static_cast<std::size_t>(kTH) * kC);
        std::vector<float> proj_b(kTH);
        fill_random(proj_w);
        fill_random(proj_b);
        float* dproj_w = h2d(proj_w, a, "pjw");
        float* dproj_b = h2d(proj_b, a, "pjb");

        struct FState {
            int t;
            float *dpos, *xc, *yc, *ws_c, *xt, *yt, *ws_t;
        };
        std::vector<FState> fstates;
        for (const int t : {20, 26}) {
            const int p = 2 * t - 1;
            std::vector<float> x(static_cast<std::size_t>(t) * kC);
            std::vector<float> pos(static_cast<std::size_t>(p) * kC);
            fill_random(x);
            diar::relpos_table_forward(pos.data(), t, kC);
            const std::string tag = std::to_string(t);
            fstates.push_back(FState{
                t, h2d(pos, a, "pos" + tag), h2d(x, a, "xc" + tag),
                a.alloc(("yc" + tag).c_str(),
                        static_cast<std::size_t>(t) * kC),
                a.alloc(("wsc" + tag).c_str(),
                        conformer_layer_scratch_floats(t, kC, kF, kH)),
                a.alloc(("xt" + tag).c_str(),
                        static_cast<std::size_t>(t) * kTH),
                a.alloc(("yt" + tag).c_str(),
                        static_cast<std::size_t>(t) * kTH),
                a.alloc(("wst" + tag).c_str(),
                        transformer_block_scratch_floats(t, kTH, kTI,
                                                          kTHeads))});
        }
        auto run_tf_chain = [&](const FState& st) {
            float* in = st.xt;
            float* out = st.yt;
            for (int l = 0; l < kTimelineTfLayers; ++l) {
                gpu_transformer_block(cublas, stream, kBlock, tf[l], in, out,
                                      st.ws_t, st.t, kTH, kTI, kTHeads);
                std::swap(in, out);
            }
        };
        auto run_full_chain = [&](const FState& st) {
            float* in = st.xc;
            float* out = st.yc;
            for (int l = 0; l < kTimelineLayers; ++l) {
                gpu_conformer_layer(cublas, stream, kBlock, conf[l], in,
                                    st.dpos, out, st.ws_c, st.t, kC, kF, kH);
                std::swap(in, out);
            }
            diar::backend::bench_linear(cublas, stream, kBlock, in, dproj_w,
                                        dproj_b, st.xt, st.t, kC, kTH);
            run_tf_chain(st);
        };
        for (const FState& st : fstates)
            for (int c = 0; c < 8; ++c) run_full_chain(st);
        for (const FState& st : fstates) {
            for (int which = 0; which < 2; ++which) {
                cudaEvent_t e0, e1;
                CUDA4_CHECK(cudaEventCreate(&e0));
                CUDA4_CHECK(cudaEventCreate(&e1));
                CUDA4_CHECK(cudaEventRecord(e0, stream));
                for (int c = 0; c < kTimelineChunks; ++c)
                    which == 0 ? run_tf_chain(st) : run_full_chain(st);
                CUDA4_CHECK(cudaEventRecord(e1, stream));
                CUDA4_CHECK(cudaEventSynchronize(e1));
                float total_ms = 0.0f;
                CUDA4_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));
                const double per_chunk = total_ms / kTimelineChunks;
                const int nl =
                    which == 0 ? kTimelineTfLayers
                               : kTimelineLayers + kTimelineTfLayers + 1;
                report(which == 0 ? "timeline_tf" : "timeline_full",
                       "\"layers\":" + std::to_string(nl) +
                           ",\"t\":" + std::to_string(st.t) +
                           ",\"ms_per_chunk\":" + fmt9(per_chunk) +
                           ",\"us_per_layer\":" +
                           fmt9(per_chunk * 1000.0 / nl) + "}\n");
                CUDA4_CHECK(cudaEventDestroy(e0));
                CUDA4_CHECK(cudaEventDestroy(e1));
            }
        }
    }

    // ---- G-S5e: GEMM p50 fp32 vs fp16 (bandwidth evidence) ----------------
    if (!parity_only) {
        for (const HShape& s : kHShapes) {
            std::vector<float> x(static_cast<std::size_t>(s.t) * s.in);
            std::vector<float> w(static_cast<std::size_t>(s.out) * s.in);
            std::vector<float> b(s.out);
            fill_random(x);
            fill_random(w);
            fill_random(b);
            diar::backend::GpuArena a;
            a.init(64u * 1024u * 1024u);
            float* dx = h2d(x, a, "x");
            float* dw32 = h2d(w, a, "w32");
            __half* dw16 = h2h(w, a, "w16");
            float* db = h2d(b, a, "b");
            float* dy = a.alloc("y", static_cast<std::size_t>(s.t) * s.out);
            __half* cast16 = reinterpret_cast<__half*>(
                a.alloc("c16", (static_cast<std::size_t>(s.t) * s.in + 1) / 2));
            auto time_route = [&](bool fp16) {
                std::vector<double> ms;
                for (int r = 0; r < 23; ++r) {
                    const auto c0 = std::chrono::steady_clock::now();
                    if (fp16)
                        bench_linear_h(cublas, stream, kBlock, dx, dw16, db,
                                       dy, s.t, s.in, s.out, cast16);
                    else
                        diar::backend::bench_linear(cublas, stream, kBlock,
                                                    dx, dw32, db, dy, s.t,
                                                    s.in, s.out);
                    CUDA4_CHECK(cudaStreamSynchronize(stream));
                    ms.push_back(std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - c0)
                                     .count());
                }
                return median_of(ms);
            };
            (void)time_route(false);  // warm both routes before timing
            (void)time_route(true);
            const double m32 = time_route(false);
            const double m16 = time_route(true);
            report("gemm_p50",
                   std::string("\"shape\":\"") + s.name +
                       "\",\"ms_fp32\":" + fmt9(m32) + ",\"ms_fp16\":" +
                       fmt9(m16) + ",\"speedup\":" + fmt9(m32 / m16) +
                       ",\"pass\":" + (m16 <= m32 ? "true" : "false") +
                       "}\n");
        }
    }

    // ---- G-S5f: fp16 timelines (conf x17 and full chain) ------------------
    // Same geometry/timing protocol as the fp32 timelines; the framework
    // gate compares these against timeline/timeline_full per t.
    if (!parity_only) {
        const std::size_t conf_gemm =
            4u * static_cast<std::size_t>(kF) * kC + 8u * kC * kC;
        const std::size_t conf_vec = 10u * kC + 4u * kF + 4u * kC + 6u * kC +
                                     static_cast<std::size_t>(kC) * kK +
                                     8u * kC;
        const std::size_t per_layer_h = conf_gemm / 2 + conf_vec;
        const std::size_t tf_gemm = 4u * static_cast<std::size_t>(kTH) * kTH +
                                    static_cast<std::size_t>(kTI) * kTH +
                                    static_cast<std::size_t>(kTH) * kTI;
        const std::size_t tf_vec = 4u * kTH + 4u * kTH + kTI + kTH;
        const std::size_t tf_per_layer_h = tf_gemm / 2 + tf_vec;
        diar::backend::GpuArena a;
        a.init(kTimelineLayers * per_layer_h +
               kTimelineTfLayers * tf_per_layer_h +
               (static_cast<std::size_t>(kTH) * kC) / 2 + kTH +  // proj
               16u * 1024u * 1024u);
        std::vector<diar::backend::LayerDevWeightsH> conf;
        for (int i = 0; i < kTimelineLayers; ++i) {
            LayerHost lh;
            lh.build(kC, kF, kK);
            conf.push_back(upload_layer_h(a, lh, "HF" + std::to_string(i)));
        }
        std::vector<diar::backend::TransformerDevWeightsH> tf;
        for (int i = 0; i < kTimelineTfLayers; ++i) {
            TfHost th;
            th.build(kTH, kTI);
            tf.push_back(upload_tf_h(a, th, "HU" + std::to_string(i)));
        }
        std::vector<float> proj_w(static_cast<std::size_t>(kTH) * kC);
        std::vector<float> proj_b(kTH);
        fill_random(proj_w);
        fill_random(proj_b);
        __half* dproj_w = h2h(proj_w, a, "hpjw");
        float* dproj_b = h2d(proj_b, a, "hpjb");
        __half* proj_cast = reinterpret_cast<__half*>(
            a.alloc("hpc16", (static_cast<std::size_t>(26) * kC + 1) / 2));

        struct FHState {
            int t;
            float *dpos, *xc, *yc, *ws_c, *xt, *yt, *ws_t;
        };
        std::vector<FHState> fstates;
        for (const int t : {20, 26}) {
            const int p = 2 * t - 1;
            std::vector<float> x(static_cast<std::size_t>(t) * kC);
            std::vector<float> pos(static_cast<std::size_t>(p) * kC);
            fill_random(x);
            diar::relpos_table_forward(pos.data(), t, kC);
            const std::string tag = std::to_string(t);
            fstates.push_back(FHState{
                t, h2d(pos, a, "hpos" + tag), h2d(x, a, "hxc" + tag),
                a.alloc(("hyc" + tag).c_str(),
                        static_cast<std::size_t>(t) * kC),
                a.alloc(("hwsc" + tag).c_str(),
                        conformer_layer_scratch_floats(t, kC, kF, kH)),
                a.alloc(("hxt" + tag).c_str(),
                        static_cast<std::size_t>(t) * kTH),
                a.alloc(("hyt" + tag).c_str(),
                        static_cast<std::size_t>(t) * kTH),
                a.alloc(("hwst" + tag).c_str(),
                        transformer_block_scratch_floats(t, kTH, kTI,
                                                          kTHeads))});
        }
        auto run_h_chain = [&](const FHState& st) {
            float* in = st.xc;
            float* out = st.yc;
            for (int l = 0; l < kTimelineLayers; ++l) {
                gpu_conformer_layer(cublas, stream, kBlock, conf[l], in,
                                    st.dpos, out, st.ws_c, st.t, kC, kF, kH);
                std::swap(in, out);
            }
        };
        auto run_h_full = [&](const FHState& st) {
            run_h_chain(st);
            bench_linear_h(cublas, stream, kBlock, st.yc, dproj_w, dproj_b,
                           st.xt, st.t, kC, kTH, proj_cast);
            float* in = st.xt;
            float* out = st.yt;
            for (int l = 0; l < kTimelineTfLayers; ++l) {
                gpu_transformer_block(cublas, stream, kBlock, tf[l], in, out,
                                      st.ws_t, st.t, kTH, kTI, kTHeads);
                std::swap(in, out);
            }
        };
        for (const FHState& st : fstates)
            for (int c = 0; c < 8; ++c) run_h_full(st);
        for (const FHState& st : fstates) {
            for (int which = 0; which < 2; ++which) {
                cudaEvent_t e0, e1;
                CUDA4_CHECK(cudaEventCreate(&e0));
                CUDA4_CHECK(cudaEventCreate(&e1));
                CUDA4_CHECK(cudaEventRecord(e0, stream));
                for (int c = 0; c < kTimelineChunks; ++c)
                    which == 0 ? run_h_chain(st) : run_h_full(st);
                CUDA4_CHECK(cudaEventRecord(e1, stream));
                CUDA4_CHECK(cudaEventSynchronize(e1));
                float total_ms = 0.0f;
                CUDA4_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));
                const double per_chunk = total_ms / kTimelineChunks;
                const int nl = which == 0 ? kTimelineLayers
                                          : kTimelineLayers +
                                                kTimelineTfLayers + 1;
                report(which == 0 ? "timeline_h" : "timeline_h_full",
                       "\"layers\":" + std::to_string(nl) +
                           ",\"t\":" + std::to_string(st.t) +
                           ",\"ms_per_chunk\":" + fmt9(per_chunk) +
                           ",\"us_per_layer\":" +
                           fmt9(per_chunk * 1000.0 / nl) + "}\n");
                CUDA4_CHECK(cudaEventDestroy(e0));
                CUDA4_CHECK(cudaEventDestroy(e1));
            }
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
    // fp16 pack/unpack round-trip smoke (deep 65536-value oracle lives in
    // tests/test_cublas_layout.cpp; this pins the linkage + engine range)
    {
        std::vector<float> v(4096);
        fill_random(v);
        std::vector<std::uint16_t> h(4096);
        diar::pack_weights_f16(v.data(), h.data(), 4096, 1);
        std::vector<float> back(4096);
        diar::unpack_weights_f16(h.data(), back.data(), 4096, 1);
        for (std::size_t i = 0; i < v.size(); ++i) {
            const float rel = std::fabs(back[i] - v[i]) /
                              std::max(1e-6f, std::fabs(v[i]));
            expect(rel < 1.1e-3f, "half round-trip rel error");
        }
    }
    // transformer block reference: finite + post-LN rows are standardized
    // (gamma=1/beta=0 host set => mean~0, biased var~1 per row) — the same
    // reference the G-S4e GPU gate compares against.
    {
        const int tt = 6;
        TfHost th;
        th.build(kTH, kTI);
        std::vector<float> tx(static_cast<std::size_t>(tt) * kTH);
        fill_random(tx);
        std::vector<float> ty(tx.size());
        diar::transformer_block_forward(tx.data(), th.host_ref(), ty.data(),
                                        tt, kTH, kTI, kTHeads);
        for (int r = 0; r < tt; ++r) {
            double mean = 0.0, var = 0.0;
            for (int i = 0; i < kTH; ++i)
                mean += ty[static_cast<std::size_t>(r) * kTH + i];
            mean /= kTH;
            for (int i = 0; i < kTH; ++i) {
                const double d =
                    ty[static_cast<std::size_t>(r) * kTH + i] - mean;
                var += d * d;
            }
            var /= kTH;
            expect(std::isfinite(mean) && std::isfinite(var),
                   "tf row stats finite");
            expect(std::fabs(mean) < 1e-4, "tf post-LN row mean ~ 0");
            expect(std::fabs(var - 1.0) < 1e-3, "tf post-LN row var ~ 1");
        }
    }
    report("verdict", "\"mode\":\"cpu-selftest-complete\"}\n");
    return 0;
}

#endif  // DIAR_WITH_CUDA
