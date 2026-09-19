// M3 Step 6b — StemRoute + fused-head hook tests (CPU-only; the CUDA
// implementations are src/stem_cuda.cpp / encoder_cuda.cpp, gated on
// Kaggle/CI — these tests pin the INTERFACE contracts):
//
//   1. delegating stem mock (recomputes subsampling_forward with the same
//      weights) routed through the hook reproduces the stem_route=nullptr
//      run BIT-IDENTICALLY.
//   2. refusing stem mock (returns false) falls back to the CPU stem
//      bit-identically (the t_mel-budget escape hatch).
//   3. constant stem mock changes values but no geometry — frame count,
//      ledger, [0,1] prob validity survive.
//   4. taps force the CPU stem even with a stem route attached.
//   5. headed encoder route (encoder_forward_headed computing the chain +
//      diar_head_forward on "device") reproduces the baseline
//      bit-identically; the CPU head is skipped.
//   6. headed route refusing (returns false) falls back to plain
//      encoder_forward + CPU head bit-identically.
#include "diar/engine.hpp"
#include "diar/layers.hpp"
#include "diar/nn.hpp"
#include "diar/conformer.hpp"
#include "diar/subsampling.hpp"
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

diar::EngineConfig base_cfg() {
    diar::EngineConfig c;
    c.geometry = diar::StreamGeometry::streaming();
    return c;
}

struct RunResult {
    std::int64_t frames = 0;
    std::vector<float> pre, post;
};

RunResult run_stream(const std::vector<float>& audio,
                     diar::EncoderRoute* enc_route = nullptr,
                     diar::StemRoute* stem_route = nullptr,
                     diar::TapSink* taps = nullptr) {
    diar::DiarEngine engine(load_tiny(), base_cfg());
    if (enc_route) engine.set_encoder_route(enc_route);
    if (stem_route) engine.set_stem_route(stem_route);
    if (taps) engine.set_tap_sink(taps);
    engine.feed_audio(audio.data(), audio.size());
    engine.finish();
    RunResult r;
    r.frames = engine.n_frames();
    r.pre = engine.pre_gate_probs();
    r.post = engine.post_gate_probs();
    return r;
}

// ---- mocks -----------------------------------------------------------------

struct DelegatingStem : diar::StemRoute {
    const diar::SortformerWeights* w = nullptr;
    int calls = 0;
    std::vector<int> seen_t_mel, seen_t3, seen_feat_len;

    bool stem_forward(const float* mel, int t_mel, int feat_len, float* y,
                      int t3, const diar::StemRouteConfig& cfg) override {
        const diar::SortformerConfig& c = w->config();
        expect(cfg.feat_in == c.feat_in &&
                   cfg.conv_channels == c.subsampling_conv_channels &&
                   cfg.d_model == c.d_model &&
                   cfg.subsampling_factor == c.subsampling_factor,
               "stem mock: cfg echoes the model config");
        expect(t3 == diar::sortformer_subsampled_len(t_mel, c.subsampling_factor),
               "stem mock: t3 == subsampled_len(t_mel)");
        calls++;
        seen_t_mel.push_back(t_mel);
        seen_t3.push_back(t3);
        seen_feat_len.push_back(feat_len);
        subsampling_forward(mel, w->stem(), y, t_mel, c.feat_in,
                            c.subsampling_conv_channels, c.d_model, feat_len);
        return true;
    }
};

struct RefusingStem : diar::StemRoute {
    int calls = 0;
    bool stem_forward(const float*, int, int, float*, int,
                      const diar::StemRouteConfig&) override {
        calls++;
        return false;
    }
};

struct ConstantStem : diar::StemRoute {
    int calls = 0;
    bool stem_forward(const float*, int, int, float* y, int t3,
                      const diar::StemRouteConfig& cfg) override {
        calls++;
        const std::size_t n = static_cast<std::size_t>(t3) * cfg.d_model;
        for (std::size_t i = 0; i < n; i++) y[i] = 1.0F;
        return true;
    }
};

// headed encoder mock: runs the CPU chain + diar_head_forward, i.e. exactly
// what CudaEncoderRoute::encoder_forward_headed does on device.
struct HeadedRoute : diar::EncoderRoute {
    const diar::SortformerWeights* w = nullptr;
    int plain_calls = 0, headed_calls = 0;

    bool provides_head() const override { return true; }

    void run_chain(const float* x, const float* pe, float* px, int L,
                   const diar::EncoderRouteConfig& cfg) const {
        const int D = cfg.d_model, X = cfg.transformer_hidden;
        const std::size_t ld = static_cast<std::size_t>(L) * D;
        const std::size_t lx = static_cast<std::size_t>(L) * X;
        std::vector<float> cy(ld * 2);
        float* cb[2] = {cy.data(), cy.data() + ld};
        int c = 0;
        for (int i = 0; i < cfg.encoder_layers; ++i) {
            const float* in = (i == 0) ? x : cb[c];
            diar::conformer_layer_forward(in, pe, w->conformer(i), cb[c ^ 1], L,
                D, cfg.encoder_d_ff, cfg.encoder_heads, cfg.conv_kernel);
            c ^= 1;
        }
        diar::nn::linear_forward(cb[c], w->proj_w(), w->proj_b(), px,
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
    }

    bool encoder_forward(const float* x, const float* pe, float* px, int L,
        const diar::EncoderRouteConfig& cfg) override {
        plain_calls++;
        run_chain(x, pe, px, L, cfg);
        return true;
    }

    bool encoder_forward_headed(const float* x, const float* pe, float* preds,
        int L, const diar::EncoderRouteConfig& cfg) override {
        headed_calls++;
        const int X = cfg.transformer_hidden;
        const int S = w->config().num_speakers;  // tiny fixtures vary (3 here)
        std::vector<float> px(static_cast<std::size_t>(L) * X);
        run_chain(x, pe, px.data(), L, cfg);
        diar::diar_head_forward(px.data(), w->head_hidden_w(), w->head_hidden_b(),
            w->head_spks_w(), w->head_spks_b(), preds, L, X, S);
        return true;
    }
};

// refuses the headed path only — must fall back to plain encoder_forward
// (which the same mock serves) + the CPU head.
struct HeadedRefusingRoute : public HeadedRoute {
    bool encoder_forward_headed(const float*, const float*, float*, int,
        const diar::EncoderRouteConfig&) override {
        headed_calls++;
        return false;
    }
};

struct StemTapRecorder : diar::TapSink {
    int count = 0;
    bool saw_stem_out = false;
    bool saw_preds = false;
    void tap(const char* name, const float*, int, int) override {
        count++;
        if (std::strcmp(name, "stem.out") == 0) saw_stem_out = true;
        if (std::strcmp(name, "preds") == 0) saw_preds = true;
    }
};

// ---- tests -----------------------------------------------------------------

void test_delegating_stem_bit_identity() {
    const std::vector<float> audio = make_audio(16000 * 5, 7);
    const diar::SortformerWeights w = load_tiny();

    const RunResult base = run_stream(audio);
    DelegatingStem stem;
    stem.w = &w;
    const RunResult routed = run_stream(audio, nullptr, &stem);

    expect(stem.calls > 0, "delegating stem: route was called");
    expect(base.frames == routed.frames, "delegating stem: identical frames");
    expect(base.pre == routed.pre, "delegating stem: bit-identical pre-gate");
    expect(base.post == routed.post, "delegating stem: bit-identical post-gate");
    // streaming windows run full-grid EXCEPT the final flush window (the
    // tailfix route pads to a 32-multiple and passes feat_len = valid rows)
    int masked = 0;
    for (int fl : stem.seen_feat_len)
        if (fl > 0) masked++;
    expect(masked <= 1,
           "delegating stem: at most one masked tail window (got " +
               std::to_string(masked) + ")");
}

void test_refusing_stem_falls_back() {
    const std::vector<float> audio = make_audio(16000 * 4, 11);

    const RunResult base = run_stream(audio);
    RefusingStem stem;
    const RunResult routed = run_stream(audio, nullptr, &stem);

    expect(stem.calls > 0, "refusing stem: route was called");
    expect(base.pre == routed.pre && base.post == routed.post,
        "refusing stem: CPU fallback bit-identical");
}

void test_constant_stem_geometry_survives() {
    const std::vector<float> audio = make_audio(16000 * 3, 13);

    const RunResult base = run_stream(audio);
    ConstantStem stem;
    const RunResult routed = run_stream(audio, nullptr, &stem);

    expect(stem.calls > 0, "constant stem: route was called");
    expect(base.frames == routed.frames, "constant stem: identical frames");
    expect(base.pre != routed.pre, "constant stem: values changed");
    for (float v : routed.post)
        expect(v >= 0.0F && v <= 1.0F, "constant stem: probs stay in [0,1]");
}

void test_taps_force_cpu_stem() {
    const std::vector<float> audio = make_audio(16000 * 4, 17);
    const diar::SortformerWeights w = load_tiny();

    DelegatingStem stem;
    stem.w = &w;
    StemTapRecorder rec;
    const RunResult routed = run_stream(audio, nullptr, &stem, &rec);

    expect(rec.saw_stem_out && rec.saw_preds, "taps: stem.out + preds fired");
    expect(stem.calls == 0, "taps: stem route bypassed (CPU authoritative)");
    const RunResult base = run_stream(audio);
    expect(base.pre == routed.pre && base.post == routed.post,
        "taps: routed-but-tapped run == pure CPU run");
}

void test_headed_route_bit_identity() {
    const std::vector<float> audio = make_audio(16000 * 5, 19);
    const diar::SortformerWeights w = load_tiny();

    const RunResult base = run_stream(audio);
    HeadedRoute route;
    route.w = &w;
    const RunResult routed = run_stream(audio, &route);

    expect(route.headed_calls > 0, "headed: headed path was called");
    expect(route.plain_calls == 0, "headed: plain path never called");
    expect(base.frames == routed.frames, "headed: identical frames");
    expect(base.pre == routed.pre, "headed: bit-identical pre-gate");
    expect(base.post == routed.post, "headed: bit-identical post-gate");
}

void test_headed_refusal_falls_back() {
    const std::vector<float> audio = make_audio(16000 * 4, 23);
    const diar::SortformerWeights w = load_tiny();

    const RunResult base = run_stream(audio);
    HeadedRefusingRoute route;
    route.w = &w;
    const RunResult routed = run_stream(audio, &route);

    expect(route.headed_calls > 0, "headed-refusal: headed path was tried");
    expect(route.plain_calls > 0, "headed-refusal: plain fallback served");
    expect(base.pre == routed.pre && base.post == routed.post,
        "headed-refusal: fallback bit-identical");
}

}  // namespace

int main() {
    test_delegating_stem_bit_identity();
    test_refusing_stem_falls_back();
    test_constant_stem_geometry_survives();
    test_taps_force_cpu_stem();
    test_headed_route_bit_identity();
    test_headed_refusal_falls_back();
    std::cout << "test_stem_route: all tests passed\n";
    return 0;
}
