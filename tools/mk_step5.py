#!/usr/bin/env python3
"""Derive tools/step5_bench_main.cpp from tools/step4_bench_main.cpp.

Step 5 = fp16 weight storage (GemmEx 16F-in / 32F-acc) behind opt-in H
weight structs. Keeps every Step-4 gate as regression (the layer bodies are
now templates; the fp32 instantiation must reproduce the same numbers) and
adds the G-S5 gate family.
"""
import re

SRC = 'tools/step4_bench_main.cpp'
DST = 'tools/step5_bench_main.cpp'
s = open(SRC).read()

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:70])
    s = s.replace(old, new)

# ---- 1) header comment ----------------------------------------------------
rep('''// tools/step4_bench_main.cpp — Step 4 bench harness (MHA + conv + layer).
//
// Same two-mode pattern as step3_bench_main.cpp:''',
'''// tools/step5_bench_main.cpp — Step 5 bench harness (fp16 weight storage).
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
// Same two-mode pattern as step3/step4_bench_main.cpp:''')

# gate list lines in the header comment: keep as-is (they document S4) but
# rename the file reference on the last comment line.
rep('''//      G-S4d3 full-encoder timeline (conf x17 + proj + tf x18) vs G-D gate
//  Reports [s4] JSON lines.''',
'''//      G-S4d3 full-encoder timeline (conf x17 + proj + tf x18) vs G-D gate
//  Reports [s5] JSON lines.''')

# ---- 2) using declarations ------------------------------------------------
rep('''using diar::backend::bench_dwconv_bn_silu;
using diar::backend::bench_glu;
using diar::backend::bench_softmax_rows;''',
'''using diar::backend::bench_cast_f32_f16;
using diar::backend::bench_dwconv_bn_silu;
using diar::backend::bench_glu;
using diar::backend::bench_linear_h;
using diar::backend::bench_softmax_rows;''')

# ---- 3) report prefix -----------------------------------------------------
rep('''void report(const std::string& key, const std::string& json) {
    std::printf("[s4] {\\"k\\":\\"%s\\",%s", key.c_str(), json.c_str());
    std::fflush(stdout);
}''',
'''void report(const std::string& key, const std::string& json) {
    std::printf("[s5] {\\"k\\":\\"%s\\",%s", key.c_str(), json.c_str());
    std::fflush(stdout);
}''')

# fatal markers mention [s4] too (cosmetic, keep consistent)
s = s.replace('[s4] {\\"k\\":\\"fatal', '[s5] {\\"k\\":\\"fatal')

# ---- 4) fp16 gate constants ------------------------------------------------
rep('''const double kTfGate = 2.0 * parity_gate(768);
''',
'''const double kTfGate = 2.0 * parity_gate(768);

// ---- Step 5 (fp16-storage route) gate tiers -------------------------------
// Unlike the fp32 tiers, the fp16 parity error is dominated by INPUT
// QUANTIZATION (2^-11 RNE on the weights once at upload + on every GEMM
// activation on device), accumulated over k products: expected ~sqrt(k)
// random-walk of per-term ~|x||w|*2^-10 relative. Provisional tier
// 2.5e-6*k (floor 2e-3, cap 4e-2) — recalibrated against measured values
// exactly like every gate in this harness lineage (Step 4 v2 precedent).
inline double fp16_gate(int k) {
    return std::min(4e-2, std::max(2e-3, 2.5e-6 * static_cast<double>(k)));
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
''')

# ---- 5) host half-pack helpers --------------------------------------------
rep('''double median_of(std::vector<double>& v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}''',
'''double median_of(std::vector<double>& v) {
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
}''')

# ---- 6) G-S5 parity sections (before the G-S4d timeline) -------------------
REP5 = r'''
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
        const double gq = std::max(1e-3, g / 2);
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

    // ---- G-S4d: 17-layer device-resident timeline ---'''
rep('''
    // ---- G-S4d: 17-layer device-resident timeline ---''', REP5)

# ---- 7) G-S5e/G-S5f timing sections (before the gpu-bench verdict) ---------
REP6 = r'''
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

    report("verdict", "\"mode\":\"gpu-bench-complete\"}\n");'''
rep('''
    report("verdict", "\\"mode\\":\\"gpu-bench-complete\\"}\\n");''', REP6)

# ---- 8) CPU selftest: half pack/unpack round-trip smoke --------------------
rep('''    // transformer block reference: finite + post-LN rows are standardized''',
'''    // fp16 pack/unpack round-trip smoke (deep 65536-value oracle lives in
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
    // transformer block reference: finite + post-LN rows are standardized''')

open(DST, 'w').write(s)
print('wrote', DST, len(s), 'bytes')
