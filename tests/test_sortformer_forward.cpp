// M2 Stage 3.2b — sortformer_run_chunk assembly tests (C++-only).
// The numpy-mirror comparison lives in tests/test_stage32_mirror.py (drives
// tools/dump_forward + reference/mirror_sortformer.py); this file pins the
// properties a same-language double-run cannot catch:
//   - xscale timing (chunk_embs are RAW; state prefix re-scaled each chunk)
//   - concat ordering + bit-exact passthrough of AOSC state
//   - stored-PE slice == formula table (bit-exact when store built by the
//     same pinned formula; row semantics pos = (L-1) - r on both sides)
//   - tail-mode divergence: feat_len>0 masked rows == out.bias bit-exact,
//     feat_len<=0 production rows differ
//   - determinism (double run bit-equal), PE budget guard, non-degeneracy
// Weights come from the shared tiny fixture (tests/tiny_fixture.hpp):
// pattern values are all-positive and varied, so ReLU stacks stay live.

#include "tiny_fixture.hpp"

#include "diar/sortformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// Records every tap into owned buffers (pointers into run_chunk temporaries
// die at return, so copy immediately).
struct TapRecorder : diar::TapSink {
    std::map<std::string, std::vector<float>> taps;
    std::map<std::string, std::pair<int, int>> dims;

    void tap(const char* name, const float* data, int rows, int cols) override {
        std::vector<float> copy(data, data + static_cast<std::size_t>(rows) * cols);
        dims[name] = {rows, cols};
        auto it = taps.emplace(name, std::vector<float>{});
        bool inserted = it.second;
        (void)inserted;
        it.first->second = std::move(copy);
    }
};

diar::SortformerWeights load_tiny(bool with_pe = true) {
    const Tiny tiny;
    const std::vector<TSpec> tensors = tiny_tensors(with_pe);
    // The raw index-pattern values (up to ~0.5*130) saturate the head's
    // sigmoid to 1.0f in fp32 — which would mask real divergence (e.g. the
    // tail-mode test). Scale the whole table down; determinism and
    // positivity preserved, activations land in a sane range.
    std::map<std::string, std::vector<float>> scaled;
    for (std::size_t i = 0; i < tensors.size(); i++) {
        std::vector<float> v(numel(tensors[i]));
        for (std::size_t j = 0; j < v.size(); j++) v[j] = pattern(i, j) * 0.003F;
        scaled.emplace(tensors[i].name, std::move(v));
    }
    return diar::SortformerWeights::load(
        write_temp(build_gguf(tiny_kv_typed(tiny), tensors, &scaled)));
}

// distinctive pseudo-state: spkcache/fifo frames of D floats, value = seed
// per region so concat order is observable.
std::vector<float> make_state(int frames, int d, float base) {
    std::vector<float> v(static_cast<std::size_t>(frames) * d);
    for (std::size_t i = 0; i < v.size(); i++)
        v[i] = base + 0.0001F * static_cast<float>(i % 97);
    return v;
}

std::vector<float> make_mel(int frames, int mels) {
    std::vector<float> v(static_cast<std::size_t>(frames) * mels);
    unsigned lcg = 12345u;
    for (std::size_t i = 0; i < v.size(); i++) {
        lcg = lcg * 1103515245u + 12345u;
        v[i] = static_cast<float>((lcg >> 16) & 0xFF) / 512.0F - 0.25F;  // [-0.25, 0.25)
    }
    return v;
}

void test_geometry_and_ledger() {
    const diar::SortformerWeights w = load_tiny();
    expect(diar::sortformer_subsampled_len(160, 8) == 20, "160 -> 20");
    expect(diar::sortformer_subsampled_len(86, 8) == 11, "86 -> 11");
    expect(diar::sortformer_subsampled_len(159, 8) == 20, "159 -> 20 (ceil chain == ceil(t/8))");
    expect(diar::sortformer_subsampled_len(157, 8) == 20 && diar::sortformer_subsampled_len(153, 8) == 20,
        "non-multiples round up");
    expect(diar::sortformer_subsampled_len(1, 8) == 1, "1 -> 1");

    const std::vector<float> mel = make_mel(160, 128);
    const std::vector<float> sc = make_state(12, 16, 0.5F);
    const std::vector<float> fifo = make_state(6, 16, 0.75F);
    TapRecorder rec;
    const diar::SortformerChunkOutput out = diar::sortformer_run_chunk(
        mel.data(), 160, -1, sc.data(), 12, fifo.data(), 6, w, &rec);
    expect(out.chunk_frames == 20 && out.total_frames == 38, "L ledger (12+6+20)");
    expect(out.preds.size() == 38u * 3u && out.chunk_embs.size() == 20u * 16u,
        "output shapes");
    expect(rec.dims["stem.out"].first == 20 && rec.dims["pos_emb"].first == 2 * 38 - 1,
        "tap dims");
    expect(rec.dims["conformer.1"].first == 38 && rec.dims["proj.out"].second == 12,
        "layer tap dims");
    expect(rec.dims["transformer.1"].second == 12 && rec.dims["preds"].second == 3,
        "head tap dims");

    // empty state + zero-length mel window: no frames at all
    const diar::SortformerChunkOutput empty = diar::sortformer_run_chunk(
        nullptr, 0, -1, nullptr, 0, nullptr, 0, w, nullptr);
    expect(empty.total_frames == 0 && empty.preds.empty(), "empty run is a no-op");
}

void test_xscale_timing_and_concat() {
    const diar::SortformerWeights w = load_tiny();
    const int D = 16;
    const std::vector<float> mel = make_mel(160, 128);
    const std::vector<float> sc = make_state(10, D, 0.5F);
    const std::vector<float> fifo = make_state(5, D, 0.75F);
    TapRecorder rec;
    const diar::SortformerChunkOutput out = diar::sortformer_run_chunk(
        mel.data(), 160, -1, sc.data(), 10, fifo.data(), 5, w, &rec);

    const std::vector<float>& stem = rec.taps["stem.out"];
    expect(stem.size() == out.chunk_embs.size() &&
               std::equal(stem.begin(), stem.end(), out.chunk_embs.begin()),
        "chunk_embs bit-equal stem.out tap (copied BEFORE xscale)");

    const std::vector<float>& raw = rec.taps["concat.raw"];
    const std::vector<float>& xs = rec.taps["xscaled"];
    const float scale = std::sqrt(static_cast<float>(D));
    // concat order + passthrough: rows [0,10) = spkcache, [10,15) = fifo,
    // [15,35) = chunk (raw, unscaled)
    bool raw_ok = true, xs_ok = true;
    for (int r = 0; r < 10; r++)
        for (int c = 0; c < D; c++) {
            raw_ok &= raw[r * D + c] == sc[r * D + c];
            xs_ok &= xs[r * D + c] == sc[r * D + c] * scale;  // state re-scaled
        }
    for (int r = 0; r < 5; r++)
        for (int c = 0; c < D; c++)
            raw_ok &= raw[(10 + r) * D + c] == fifo[r * D + c];
    for (int r = 0; r < 20; r++)
        for (int c = 0; c < D; c++)
            raw_ok &= raw[(15 + r) * D + c] == out.chunk_embs[r * D + c];
    expect(raw_ok, "concat.raw = [spkcache|fifo|chunk] bit-exact");
    expect(xs_ok, "xscale covers the state prefix too (AOSC raw-emb contract)");

    // stem-only equivalence: run the stem kernel directly, compare bit-exact
    std::vector<float> direct(20 * D);
    diar::subsampling_forward(mel.data(), w.stem(), direct.data(), 160, 128, 8, D, -1);
    expect(std::equal(direct.begin(), direct.end(), out.chunk_embs.begin()),
        "stem output == direct subsampling_forward call");
}

void test_pe_slice_vs_formula() {
    // Custom GGUF whose stored pe IS the pinned formula table for L=64:
    // the slice for any L<=64 must then be bit-equal to formula(L).
    const Tiny tiny;
    std::vector<float> pe_store(static_cast<std::size_t>(2 * 64 - 1) * 16);
    diar::relpos_table_forward(pe_store.data(), 64, 16);
    const std::map<std::string, std::vector<float>> ov = {{"encoder.pos_enc.pe", pe_store}};
    const diar::SortformerWeights w = diar::SortformerWeights::load(write_temp(
        build_gguf(tiny_kv_typed(tiny), tiny_tensors(true), &ov)));
    const std::vector<float> mel = make_mel(160, 128);
    const std::vector<float> sc = make_state(12, 16, 0.5F);
    TapRecorder rec;
    (void)diar::sortformer_run_chunk(
        mel.data(), 160, -1, sc.data(), 12, nullptr, 0, w, &rec);
    const int L = 32;  // 12 + 20
    const std::vector<float>& sliced = rec.taps["pos_emb"];
    std::vector<float> formula(static_cast<std::size_t>(2 * L - 1) * 16);
    diar::relpos_table_forward(formula.data(), L, 16);
    expect(sliced.size() == formula.size() &&
               std::equal(sliced.begin(), sliced.end(), formula.begin()),
        "stored-table slice bit-equal formula table (row semantics pin)");

    // DFW1 route (no stored pe) builds the formula table itself:
    // build a DFW1 without pe via the fixture table
    std::vector<TSpec> tensors = tiny_tensors(false);
    const std::vector<CfgPair> cfg = {
        {"sortformer.encoder.d_model", "16"}, {"sortformer.encoder.n_layers", "2"},
        {"sortformer.encoder.n_heads", "2"}, {"sortformer.encoder.d_ff", "24"},
        {"sortformer.encoder.conv_kernel_size", "3"},
        {"sortformer.encoder.subsampling_conv_channels", "8"},
        {"sortformer.encoder.pos_emb_max_len", "64"},
        {"sortformer.transformer.n_layers", "2"},
        {"sortformer.transformer.hidden_size", "12"},
        {"sortformer.transformer.inner_size", "24"},
        {"sortformer.transformer.n_heads", "2"},
        {"sortformer.num_speakers", "3"},
    };
    const diar::SortformerWeights wf = diar::SortformerWeights::load(
        write_temp(build_dfw1(cfg, tensors)));
    TapRecorder rec2;
    (void)diar::sortformer_run_chunk(
        mel.data(), 160, -1, sc.data(), 12, nullptr, 0, wf, &rec2);
    expect(rec2.taps["pos_emb"] == formula, "formula route matches too");

    // PE budget guard: L > pos_emb_max_len must throw
    const std::vector<float> sc_big = make_state(64, 16, 0.5F);  // 64+20 > 64
    bool threw = false;
    try {
        (void)diar::sortformer_run_chunk(
            mel.data(), 160, -1, sc_big.data(), 64, nullptr, 0, w, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "L exceeding pos_emb_max_len throws");
}

void test_tail_modes_diverge() {
    const diar::SortformerWeights w = load_tiny();
    const int D = 16;
    // 86 mel frames: T3 = 11 full-window rows. feat_len=80 -> valid L3 = 10
    // (80 -> 40 -> 20 -> 10): rows >= 10 are masked fill == out.bias.
    const std::vector<float> mel = make_mel(86, 128);
    const std::vector<float> sc = make_state(8, D, 0.5F);

    TapRecorder masked, production;
    diar::SortformerChunkOutput a = diar::sortformer_run_chunk(
        mel.data(), 86, 80, sc.data(), 8, nullptr, 0, w, &masked);
    const diar::SortformerChunkOutput b = diar::sortformer_run_chunk(
        mel.data(), 86, -1, sc.data(), 8, nullptr, 0, w, &production);
    expect(a.chunk_frames == 11 && b.chunk_frames == 11,
        "geometry identical across tail modes");

    bool tail_fill = true;
    for (int c = 0; c < D; c++)
        tail_fill &= a.chunk_embs[10 * D + c] == w.stem().out_b[c];
    expect(tail_fill, "masked tail row == out.bias (bit-exact)");
    bool differs = false;
    for (int c = 0; c < D; c++)
        differs |= b.chunk_embs[10 * D + c] != w.stem().out_b[c];
    expect(differs, "production tail row is a real conv row, not the bias fill");
    double tail_diff = 0;
    for (std::size_t i = 0; i < a.preds.size(); i++)
        tail_diff = std::max(tail_diff, (double)std::fabs(a.preds[i] - b.preds[i]));
    expect(tail_diff > 1e-7, "preds diverge across tail modes");
    // valid rows identical in both modes
    bool head_same = true;
    for (int r = 0; r < 10; r++)
        for (int c = 0; c < D; c++)
            head_same &= a.chunk_embs[r * D + c] == b.chunk_embs[r * D + c];
    expect(head_same, "valid rows unaffected by the mask");
}

void test_determinism_and_nondegeneracy() {
    const diar::SortformerWeights w = load_tiny();
    const std::vector<float> mel = make_mel(160, 128);
    const std::vector<float> sc = make_state(20, 16, 0.5F);
    const std::vector<float> fifo = make_state(10, 16, 0.75F);
    TapRecorder r1, r2;
    const diar::SortformerChunkOutput a = diar::sortformer_run_chunk(
        mel.data(), 160, -1, sc.data(), 20, fifo.data(), 10, w, &r1);
    const diar::SortformerChunkOutput b = diar::sortformer_run_chunk(
        mel.data(), 160, -1, sc.data(), 20, fifo.data(), 10, w, &r2);
    expect(a.preds == b.preds && a.chunk_embs == b.chunk_embs,
        "double run bit-equal");
    expect(r1.taps == r2.taps, "all taps bit-equal");

    // non-degeneracy (v13 lesson: assert the fixture produces live signal)
    float mn = a.preds[0], mx = a.preds[0];
    for (float p : a.preds) {
        mn = std::min(mn, p);
        mx = std::max(mx, p);
        expect(std::isfinite(p), "preds finite");
    }
    expect(mx > mn + 1e-6F, "preds carry signal (not exact-constant collapsed)");
    float emn = a.chunk_embs[0], emx = a.chunk_embs[0];
    for (float e : a.chunk_embs) {
        emn = std::min(emn, e);
        emx = std::max(emx, e);
    }
    expect(emx > emn, "chunk_embs carry signal");
}

}  // namespace

int main() {
    test_geometry_and_ledger();
    test_xscale_timing_and_concat();
    test_pe_slice_vs_formula();
    test_tail_modes_diverge();
    test_determinism_and_nondegeneracy();
    std::cout << "PASS: sortformer forward assembly tests\n";
    return 0;
}
