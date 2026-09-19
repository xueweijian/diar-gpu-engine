// M3 Step 6 — EncoderRoute hook tests (CPU-only; the CUDA implementation
// is src/encoder_cuda.cpp and is gated on Kaggle/CI, these tests pin the
// INTERFACE contract the device route implements):
//
//   1. delegating mock (recomputes steps 5-7 with the same CPU reference
//      calls) routed through the interface reproduces the route=nullptr
//      engine run BIT-IDENTICALLY — the x/pe/px domains of the hook are
//      exactly the conformer chain + encoder_proj + transformer chain.
//   2. refusing mock (returns false) falls back to the CPU chain
//      bit-identically (the L-budget escape hatch).
//   3. constant mock (px = c) changes values but no geometry: frame count,
//      ledger and [0,1] prob validity survive — head/AOSC/gate keep running
//      on whatever the route produces.
//   4. taps force the CPU chain even with a route attached (K5 contract).
#include "diar/engine.hpp"
#include "diar/layers.hpp"
#include "diar/nn.hpp"
#include "diar/conformer.hpp"
#include "tiny_fixture.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

diar::SortformerWeights load_tiny() {
    Tiny tiny;
    tiny.pe_max = 600;
    const std::vector<TSpec> tensors = tiny_tensors(true, tiny.pe_max);
    std::map<std::string, std::vector<float>> scaled;
    for (std::size_t i = 0; i < tensors.size(); i++) {
        std::vector<float> v(numel(tensors[i]));
        for (std::size_t j = 0; j < v.size(); j++) v[j] = pattern(i, j) * 0.003F;
        scaled.emplace(tensors[i].name, std::move(v));
    }
    return diar::SortformerWeights::load(
        write_temp(build_gguf(tiny_kv_typed(tiny), tensors, &scaled)));
}

std::vector<float> make_audio(int samples, unsigned seed) {
    std::vector<float> a(static_cast<std::size_t>(samples));
    unsigned lcg = seed;
    for (auto& v : a) {
        lcg = lcg * 1103515245u + 12345u;
        v = static_cast<float>((lcg >> 16) & 0xFF) / 512.0F - 0.25F;
    }
    return a;
}

bool ledger_equal(const std::vector<diar::ChunkLedgerEntry>& a,
                  const std::vector<diar::ChunkLedgerEntry>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); i++) {
        const diar::ChunkLedgerEntry &x = a[i], &y = b[i];
        if (x.chunk_index != y.chunk_index || x.t_mel != y.t_mel ||
            x.t3 != y.t3 || x.lc_enc != y.lc_enc || x.rc_enc != y.rc_enc ||
            x.emitted != y.emitted || x.spkcache_frames != y.spkcache_frames ||
            x.fifo_frames != y.fifo_frames ||
            x.window_frames != y.window_frames ||
            x.tail_feat_len != y.tail_feat_len)
            return false;
    }
    return true;
}

diar::EngineConfig base_cfg() {
    diar::EngineConfig c;
    c.geometry = diar::StreamGeometry::streaming();
    return c;
}

// Recomputes steps 5-7 exactly like the CPU chain in sortformer_run_chunk.
struct DelegatingRoute : diar::EncoderRoute {
    const diar::SortformerWeights* w = nullptr;
    int calls = 0;
    std::vector<int> seen_L;
    std::vector<diar::EncoderRouteConfig> seen_cfg;

    bool encoder_forward(const float* x, const float* pe, float* px, int L,
        const diar::EncoderRouteConfig& cfg) override {
        calls++;
        seen_L.push_back(L);
        seen_cfg.push_back(cfg);
        const int D = cfg.d_model, X = cfg.transformer_hidden;
        const std::size_t ld = static_cast<std::size_t>(L) * D;
        const std::size_t lx = static_cast<std::size_t>(L) * X;
        // conformer chain: ping-pong between two scratch buffers
        std::vector<float> cy(ld * 2);
        float* cb[2] = {cy.data(), cy.data() + ld};
        int c = 0;
        for (int i = 0; i < cfg.encoder_layers; ++i) {
            const float* in = (i == 0) ? x : cb[c];
            diar::conformer_layer_forward(in, pe, w->conformer(i), cb[c ^ 1], L,
                D, cfg.encoder_d_ff, cfg.encoder_heads, cfg.conv_kernel);
            c ^= 1;
        }
        const float* conf_out = cb[c];
        // encoder_proj into px (also the layer-0 input of the tf chain)
        diar::nn::linear_forward(conf_out, w->proj_w(), w->proj_b(), px,
            static_cast<std::size_t>(L), static_cast<std::size_t>(D),
            static_cast<std::size_t>(X));
        std::vector<float> ty(lx * 2);
        float* tb[2] = {ty.data(), ty.data() + lx};
        int t = 0;
        for (int i = 0; i < cfg.transformer_layers; ++i) {
            const float* in = (i == 0) ? px : tb[t];
            diar::transformer_block_forward(in, w->transformer(i), tb[t ^ 1], L,
                X, cfg.transformer_inner, cfg.transformer_heads);
            t ^= 1;
        }
        if (cfg.transformer_layers >= 1 && tb[t] != px)
            std::memcpy(px, tb[t], sizeof(float) * lx);
        return true;
    }
};

struct RefusingRoute : diar::EncoderRoute {
    int calls = 0;
    bool encoder_forward(const float*, const float*, float*, int,
        const diar::EncoderRouteConfig&) override {
        calls++;
        return false;
    }
};

struct ConstantRoute : diar::EncoderRoute {
    int calls = 0;
    bool encoder_forward(const float*, const float*, float* px, int L,
        const diar::EncoderRouteConfig& cfg) override {
        calls++;
        const std::size_t n = static_cast<std::size_t>(L) * cfg.transformer_hidden;
        for (std::size_t i = 0; i < n; i++) px[i] = 5.0F;
        return true;
    }
};

struct TapRecorder : diar::TapSink {
    int count = 0;
    bool saw_conformer0 = false;
    void tap(const char* name, const float*, int, int) override {
        count++;
        if (std::strcmp(name, "conformer.0") == 0) saw_conformer0 = true;
    }
};

struct RunResult {
    std::int64_t frames = 0;
    std::vector<float> pre, post;
    std::vector<diar::ChunkLedgerEntry> ledger;
};

// SortformerWeights is move-only, so every engine gets its own load; the
// mock keeps a pointer to a separate, value-identical load (deterministic
// fixture fill) that outlives the engine.
RunResult run_stream(const std::vector<float>& audio, diar::EncoderRoute* route,
                     diar::TapSink* taps = nullptr) {
    diar::DiarEngine engine(load_tiny(), base_cfg());
    if (route) engine.set_encoder_route(route);
    if (taps) engine.set_tap_sink(taps);
    engine.feed_audio(audio.data(), audio.size());
    engine.finish();
    RunResult r;
    r.frames = engine.n_frames();
    r.pre = engine.pre_gate_probs();
    r.post = engine.post_gate_probs();
    r.ledger = engine.chunk_ledger();
    return r;
}

void test_delegating_route_bit_identity() {
    const std::vector<float> audio = make_audio(16000 * 5, 7);
    const diar::SortformerWeights w = load_tiny();

    const RunResult base = run_stream(audio, nullptr);
    DelegatingRoute route;
    route.w = &w;
    const RunResult routed = run_stream(audio, &route);

    expect(route.calls > 0, "delegating: route was called");
    expect(route.calls == static_cast<int>(base.ledger.size()),
        "delegating: one call per chunk (" + std::to_string(route.calls) + " vs " +
            std::to_string(base.ledger.size()) + ")");
    for (std::size_t i = 0; i < base.ledger.size(); i++)
        expect(route.seen_L[i] == base.ledger[i].window_frames,
            "delegating: L == ledger window_frames at chunk " + std::to_string(i));
    const diar::SortformerConfig& c = w.config();
    for (const diar::EncoderRouteConfig& rc : route.seen_cfg) {
        expect(rc.d_model == c.d_model && rc.encoder_layers == c.encoder_layers &&
                   rc.encoder_heads == c.encoder_heads &&
                   rc.encoder_d_ff == c.encoder_d_ff &&
                   rc.conv_kernel == c.conv_kernel &&
                   rc.transformer_layers == c.transformer_layers &&
                   rc.transformer_hidden == c.transformer_hidden &&
                   rc.transformer_inner == c.transformer_inner &&
                   rc.transformer_heads == c.transformer_heads,
            "delegating: cfg echoes the model config");
    }
    expect(base.frames == routed.frames, "delegating: identical frame counts");
    expect(base.pre == routed.pre, "delegating: bit-identical pre-gate");
    expect(base.post == routed.post, "delegating: bit-identical post-gate");
    expect(ledger_equal(base.ledger, routed.ledger), "delegating: identical ledger");
}

void test_refusing_route_falls_back() {
    const std::vector<float> audio = make_audio(16000 * 4, 11);

    const RunResult base = run_stream(audio, nullptr);
    RefusingRoute route;
    const RunResult fell = run_stream(audio, &route);

    expect(route.calls > 0, "refusing: route was called");
    expect(base.pre == fell.pre && base.post == fell.post,
        "refusing: CPU fallback bit-identical");
}

void test_constant_route_changes_values_not_geometry() {
    const std::vector<float> audio = make_audio(16000 * 4, 13);

    const RunResult base = run_stream(audio, nullptr);
    ConstantRoute route;
    const RunResult routed = run_stream(audio, &route);

    expect(route.calls > 0, "constant: route was called");
    expect(base.frames == routed.frames, "constant: identical frame counts");
    expect(ledger_equal(base.ledger, routed.ledger), "constant: identical ledger");
    expect(base.post != routed.post, "constant: values differ (route really bypassed)");
    for (float p : routed.post)
        expect(std::isfinite(p) && p >= 0.0F && p <= 1.0F, "constant: probs in [0,1]");
}

void test_taps_force_cpu_chain() {
    const std::vector<float> audio = make_audio(16000 * 4, 17);

    DelegatingRoute route;
    TapRecorder rec;
    const RunResult routed = run_stream(audio, &route, &rec);
    const RunResult base = run_stream(audio, nullptr);

    expect(route.calls == 0, "taps: route bypassed when a tap sink is set");
    expect(rec.saw_conformer0, "taps: CPU conformer chain fired");
    expect(base.pre == routed.pre && base.post == routed.post,
        "taps: tapped routed run == CPU baseline");
}

}  // namespace

int main() {
    test_delegating_route_bit_identity();
    test_refusing_route_falls_back();
    test_constant_route_changes_values_not_geometry();
    test_taps_force_cpu_chain();
    std::cout << "test_encoder_route: OK\n";
    return 0;
}
