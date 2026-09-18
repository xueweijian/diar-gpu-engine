// Stage 3.2c engine tests: frame-ledger conservation, feed-sharding
// invariance, finish-tail handling, offline entries, determinism, ledger/
// state geometry and tap plumbing. Tiny weights from the shared fixture
// (independent byte writers).
#include "diar/engine.hpp"
#include "tiny_fixture.hpp"

#include <cmath>
#include <cstdio>
#include <map>
#include <cstdlib>
#include <iostream>
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
    tiny.pe_max = 600;  // streaming window 280 + offline-preset 512 must fit
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

struct TapRecorder : diar::TapSink {
    struct Rec {
        std::string name;
        int rows;
        int cols;
    };
    std::vector<Rec> recs;

    void tap(const char* name, const float* data, int rows, int cols) override {
        (void)data;
        recs.push_back(Rec{name, rows, cols});
    }
    std::vector<Rec> named(const std::string& name) const {
        std::vector<Rec> out;
        for (const Rec& r : recs)
            if (r.name == name) out.push_back(r);
        return out;
    }
};

void test_ledger_and_sharding_invariance() {
    const int n = 16000 * 6;  // 6 s
    const std::vector<float> audio = make_audio(n, 42);
    const diar::SortformerWeights w = load_tiny();

    diar::DiarEngine whole(load_tiny(), base_cfg());
    whole.feed_audio(audio.data(), audio.size());
    whole.finish();

    diar::DiarEngine shards(load_tiny(), base_cfg());
    const int s1 = 4000, s2 = 7000;  // deliberately off-hop boundaries
    shards.feed_audio(audio.data(), s1);
    shards.feed_audio(audio.data() + s1, s2 - s1);
    shards.feed_audio(audio.data() + s2, n - s2);
    shards.finish();

    expect(whole.n_frames() == shards.n_frames(), "sharding: identical frame counts");
    expect(whole.total_mel_frames() == shards.total_mel_frames(), "sharding: identical mel");
    expect(whole.post_gate_probs() == shards.post_gate_probs(),
        "sharding: bit-identical post-gate timeline");
    expect(whole.pre_gate_probs() == shards.pre_gate_probs(),
        "sharding: bit-identical pre-gate timeline");

    // Ledger: timeline frames == subsampled_len(total mel produced).
    const int expect_frames =
        diar::sortformer_subsampled_len(static_cast<int>(whole.total_mel_frames()), 8);
    expect(whole.n_frames() == expect_frames,
        "ledger: n_frames == subsampled_len(mel) (" + std::to_string(whole.n_frames()) +
            " vs " + std::to_string(expect_frames) + ")");
    // Pre-gate == post-gate in frame count.
    expect(whole.pre_gate_probs().size() == whole.post_gate_probs().size(),
        "ledger: pre/post gate same size");
    // Probs are valid probabilities.
    for (float p : whole.post_gate_probs())
        expect(std::isfinite(p) && p >= 0.0F && p <= 1.0F, "ledger: probs in [0,1]");

    // Chunk ledger: the free-running counterpart of the same identity.
    const auto& led = whole.chunk_ledger();
    expect(!led.empty(), "ledger: chunks recorded");
    std::int64_t sum_emitted = 0, sum_t3 = 0;
    for (std::size_t i = 0; i < led.size(); i++) {
        const diar::ChunkLedgerEntry& e = led[i];
        expect(e.chunk_index == static_cast<int>(i), "ledger: chunk indices are dense");
        expect(e.emitted == e.t3 - e.lc_enc - e.rc_enc, "ledger: emitted == t3 - lc - rc");
        expect(e.window_frames == e.spkcache_frames + e.fifo_frames + e.t3,
            "ledger: window == spkcache + fifo + t3");
        expect(e.t_mel > 0 && e.t3 > 0, "ledger: positive window geometry");
        sum_emitted += e.emitted;
        sum_t3 += e.t3;
    }
    expect(sum_emitted == whole.n_frames(), "ledger: emitted frames sum to the timeline");
    expect(sum_t3 == expect_frames, "ledger: chunk stem frames sum to subsampled_len(mel)");
    expect(shards.chunk_ledger().size() == led.size(), "ledger: sharding does not split chunks");
}

void test_state_geometry_and_taps() {
    // 26 s -> ~16 chunks of 1.6 s: long enough for the FIFO to overflow twice
    // and the speaker cache to hit its cap + run a compress pass.
    const int n = 16000 * 26;
    const std::vector<float> audio = make_audio(n, 11);
    const diar::SortformerWeights w = load_tiny();
    const diar::SortformerConfig& c = w.config();

    diar::DiarEngine eng(load_tiny(), base_cfg());
    TapRecorder taps;
    eng.set_tap_sink(&taps);
    eng.feed_audio(audio.data(), audio.size());
    eng.finish();

    const std::vector<diar::ChunkLedgerEntry>& led = eng.chunk_ledger();
    expect(led.size() >= 10, "taps: enough chunks for an evolution curve");

    // Streaming geometry: every full chunk emits exactly chunk_len frames and
    // the state sizes respect the AOSC caps at every chunk boundary.
    int max_spk = 0, max_fifo = 0;
    for (std::size_t i = 0; i + 1 < led.size(); i++) {
        const diar::ChunkLedgerEntry& e = led[i];
        expect(e.lc_enc == 0 && e.rc_enc == 0, "geometry: streaming preset has no lc/rc");
        expect(e.emitted == base_cfg().geometry.chunk_len,
            "geometry: full chunk emits chunk_len frames (got " + std::to_string(e.emitted) + ")");
        max_spk = std::max(max_spk, e.spkcache_frames);
        max_fifo = std::max(max_fifo, e.fifo_frames);
    }
    for (const diar::ChunkLedgerEntry& e : led) {
        expect(e.spkcache_frames <= base_cfg().geometry.spkcache_len,
            "geometry: spkcache within capacity");
        expect(e.fifo_frames <= base_cfg().geometry.fifo_len, "geometry: fifo within capacity");
    }
    expect(max_spk == base_cfg().geometry.spkcache_len,
        "geometry: spkcache saturates at its capacity (got " + std::to_string(max_spk) + ")");
    expect(max_fifo > base_cfg().geometry.fifo_len - 20,
        "geometry: fifo fills up to its capacity (got " + std::to_string(max_fifo) + ")");

    // Tap plumbing: one call per tap name per chunk, shapes straight from the
    // window geometry (the K5 stage-isolation face).
    const std::vector<TapRecorder::Rec> stems = taps.named("stem.out");
    expect(stems.size() == led.size(), "taps: one stem.out per chunk");
    for (std::size_t i = 0; i < stems.size(); i++) {
        expect(stems[i].rows == led[i].t3 && stems[i].cols == c.d_model,
            "taps: stem.out is [t3, d_model] of that chunk");
    }
    const std::vector<TapRecorder::Rec> preds = taps.named("preds");
    expect(preds.size() == led.size(), "taps: one preds per chunk");
    for (std::size_t i = 0; i < preds.size(); i++) {
        expect(preds[i].rows == led[i].window_frames && preds[i].cols == c.num_speakers,
            "taps: preds is [L, n_spk] of that chunk");
    }
    expect(taps.named("xscaled").size() == led.size(), "taps: xscale tap per chunk");
    expect(taps.named("pos_emb").size() == led.size(), "taps: pos table tap per chunk");
    const std::string first_name = taps.recs.empty() ? std::string() : taps.recs[0].name;
    expect(first_name == "stem.out", "taps: chunk order starts at the stem");
}

void test_one_sample_sharding() {
    const int n = 16000;  // 1 s
    const std::vector<float> audio = make_audio(n, 2024);

    diar::DiarEngine whole(load_tiny(), base_cfg());
    whole.feed_audio(audio.data(), audio.size());
    whole.finish();

    diar::DiarEngine drip(load_tiny(), base_cfg());
    for (int i = 0; i < n; i++) drip.feed_audio(audio.data() + i, 1);
    drip.finish();

    expect(drip.n_frames() == whole.n_frames(), "drip feed: identical frame count");
    expect(drip.total_mel_frames() == whole.total_mel_frames(), "drip feed: identical mel count");
    expect(drip.pre_gate_probs() == whole.pre_gate_probs(),
        "drip feed: bit-identical pre-gate timeline");
    expect(drip.post_gate_probs() == whole.post_gate_probs(),
        "drip feed: bit-identical post-gate timeline");
    expect(drip.chunk_ledger().size() == whole.chunk_ledger().size(),
        "drip feed: identical chunk count");
}

void test_offline_preset_geometry() {
    // 45 s at chunk_len=100 encoder frames (8 s/chunk): 5 full chunks + tail.
    const int n = 16000 * 45;
    const std::vector<float> audio = make_audio(n, 77);
    const diar::SortformerWeights w = load_tiny();

    diar::EngineConfig cfg = base_cfg();
    cfg.geometry = diar::StreamGeometry::offline_preset();  // {312, 100, 100, 100, 0, 0}
    diar::DiarEngine eng(load_tiny(), cfg);
    eng.feed_audio(audio.data(), audio.size());
    eng.finish();

    const std::vector<diar::ChunkLedgerEntry>& led = eng.chunk_ledger();
    expect(led.size() >= 6, "offline preset: whole chunks ran");    for (std::size_t i = 0; i + 1 < led.size(); i++) {
        expect(led[i].emitted == cfg.geometry.chunk_len,
            "offline preset: full chunk emits chunk_len frames");
        expect(led[i].spkcache_frames <= cfg.geometry.spkcache_len,
            "offline preset: spkcache within capacity");
    }
    expect(eng.n_frames() == diar::sortformer_subsampled_len(
               static_cast<int>(eng.total_mel_frames()), 8),
        "offline preset: frame ledger closed");
    // The rel-pos guard: a window that cannot fit the table is a loud ctor
    // failure, not a silent truncation.
    diar::EngineConfig too_wide = base_cfg();
    too_wide.geometry = diar::StreamGeometry::offline_preset();
    bool threw = false;
    try {
        // Tiny fixture table is 2*600-1 rows; the offline preset needs 512.
        Tiny small;
        small.pe_max = 400;  // < 512 window budget
        const std::vector<TSpec> tensors = tiny_tensors(true, small.pe_max);
        const std::string blob = build_gguf(tiny_kv_typed(small), tensors);
        diar::DiarEngine bad(diar::SortformerWeights::load(write_temp(blob)), too_wide);
        (void)bad;
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("pos-emb table") != std::string::npos ||
                std::string(e.what()).find("rel-pos") != std::string::npos ||
                std::string(e.what()).find("geometry") != std::string::npos;
    }
    expect(threw, "offline preset: window > rel-pos table is a loud ctor error");
}

void test_finish_tail_and_determinism() {
    const diar::SortformerWeights w = load_tiny();
    // Shorter than one hop: forced final chunk still labels whole frames.
    diar::DiarEngine short_run(load_tiny(), base_cfg());
    const std::vector<float> tiny_audio = make_audio(2000, 7);
    short_run.feed_audio(tiny_audio.data(), tiny_audio.size());
    short_run.finish();
    expect(short_run.n_frames() > 0, "tail: short audio still produces frames");
    expect(short_run.n_frames() ==
               diar::sortformer_subsampled_len(
                   static_cast<int>(short_run.total_mel_frames()), 8),
        "tail: short-audio ledger consistent");

    diar::DiarEngine a(load_tiny(), base_cfg());
    diar::DiarEngine b(load_tiny(), base_cfg());
    const std::vector<float> audio = make_audio(16000 * 3, 99);
    for (int rep = 0; rep < 2; rep++) (rep ? b : a).feed_audio(audio.data(), audio.size());
    a.finish();
    b.finish();
    expect(a.post_gate_probs() == b.post_gate_probs(), "determinism: double run bit-equal");

    // Post-finish feeds are silent no-ops.
    a.feed_audio(audio.data(), 100);
    expect(a.n_frames() == b.n_frames(), "tail: post-finish feed is a no-op");

    // finish() on an empty stream is a no-op, not a crash.
    diar::DiarEngine empty(load_tiny(), base_cfg());
    empty.finish();
    expect(empty.n_frames() == 0, "tail: empty stream finishes empty");
}

void test_nemo_tail_route_and_offline() {
    const diar::SortformerWeights w = load_tiny();
    diar::EngineConfig k5 = base_cfg();
    k5.nemo_tail_semantics = true;
    diar::DiarEngine k5_engine(load_tiny(), k5);
    const std::vector<float> audio = make_audio(16000 * 4, 5);
    k5_engine.feed_audio(audio.data(), audio.size());
    k5_engine.finish();

    diar::DiarEngine k6_engine(load_tiny(), base_cfg());
    k6_engine.feed_audio(audio.data(), audio.size());
    k6_engine.finish();

    // Same geometry (ledger), different tail values on non-hop-multiple
    // total mel. (16800+256 samples -> 17056/hop=1705+1? not multiple of 8.)
    expect(k5_engine.n_frames() == k6_engine.n_frames(),
        "tail routes share frame geometry");
    bool differs = k5_engine.post_gate_probs() != k6_engine.post_gate_probs();
    expect(differs, "tail routes diverge in values (final chunk mask)");
    // The routing must be visible in the ledger too (feat_len bookkeeping).
    bool masked = false, unmasked = false;
    for (const diar::ChunkLedgerEntry& e : k5_engine.chunk_ledger())
        if (e.feat_len >= 0) masked = true;
    for (const diar::ChunkLedgerEntry& e : k6_engine.chunk_ledger())
        if (e.feat_len < 0) unmasked = true;
    expect(masked && unmasked, "tail routes: feat_len routing recorded in the ledger");

    // Offline entries.
    const diar::OfflineDiarizationResult off =
        diar::diarize_offline(w, audio.data(), audio.size());
    expect(off.n_frames > 0 && off.preds.size() ==
                static_cast<std::size_t>(off.n_frames) * w.config().num_speakers,
        "offline: preds sized [n_frames, n_spk]");
    expect(off.n_frames == diar::sortformer_subsampled_len(off.n_mel, 8),
        "offline: frame ledger");
    for (float p : off.preds)
        expect(std::isfinite(p) && p >= 0.0F && p <= 1.0F, "offline: probs in [0,1]");

    // segments() smoke on the streaming engine.
    auto segs = k6_engine.segments();
    for (const auto& s : segs) expect(s.end_sec >= s.start_sec, "segments: ordered spans");
    const diar::FrameProbabilities fp = k6_engine.post_gate_frame_probabilities();
    expect(fp.frames() == static_cast<std::size_t>(k6_engine.n_frames()) &&
               fp.speakers() == static_cast<std::size_t>(w.config().num_speakers),
        "segments: frame probabilities face matches the timeline");
}

}  // namespace

int main() {
    test_ledger_and_sharding_invariance();
    test_state_geometry_and_taps();
    test_one_sample_sharding();
    test_offline_preset_geometry();
    test_finish_tail_and_determinism();
    test_nemo_tail_route_and_offline();
    std::cout << "PASS: diar engine tests\n";
    return 0;
}
