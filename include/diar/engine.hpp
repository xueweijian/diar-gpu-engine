#pragma once

// M2 Stage 3.2c — DiarEngine: the host state machine wired to the neural
// assembly (3.2b). Mirrors upstream DiarStream + DiarModel::diarize_offline
// (a5b6953 diar_pipeline.cpp): mel scheduling via produce_new_mel_frames,
// per-chunk sortformer_run_chunk, AOSC state update, BirthGate timeline.
//
// Deliberate omissions vs upstream (3.2 plan non-goals): maybe_compact /
// frozen segments (memory guard for hour-long streams; K5/K6 fixtures are
// <= 357 s and the timeline stays tiny) and flush_available/speaker_for_frames
// (riva-facing APIs).

#include "diar/diar.hpp"
#include "diar/sortformer.hpp"

#include <cstdint>
#include <vector>

namespace diar {

// Tail semantics (T9 + tail-fixture, settled 2026-09-18 against upstream
// a5b6953 source + m2-ref npz + tail_slice probe):
//   - PRODUCTION (a5b6953 ggml run_chunk): no feat_len, no window padding;
//     subsampled_len = whole-window ceil. The tail_slice probe (body 0.028,
//     tail row 0.9964) proves the final window's extra computed rows are
//     WRONG speech values where python NeMo has masked rows.
//   - PYTHON NeMo reference: streaming loader pads the tail window to a
//     32-mel multiple and masks pad rows (masked rows == out.bias fill,
//     all-zero probs); emission trims phantom rows.
// The engine follows production on full windows and takes the NeMo route
// (pad32 + feat_len>0 + trim) on the final flush window. tail_feat_len in
// the ledger marks which route a chunk took (0 = production). The K5 gate
// compares the real prefix and reports the phantom count.

struct EngineConfig {
    StreamGeometry geometry = StreamGeometry::streaming();
    SegmentationConfig segmentation{};
};

// One line of the chunk ledger — every run_one_chunk call appends one. This
// is the alignment table 3.3 K5 needs: NeMo's per-chunk state_lens dump
// (m2-ref npz) names the same quantities, so the free-running comparison can
// be walked chunk by chunk instead of by whole-output diffs.
struct ChunkLedgerEntry {
    int chunk_index = 0;
    int t_mel = 0;  // mel rows in the run_chunk window (incl. context)
    int t3 = 0;     // encoder frames the stem produced for it
    int lc_enc = 0, rc_enc = 0;  // context frames AOSC trims (lround/ceil)
    int emitted = 0;             // timeline frames appended (t3 - lc - rc)
    int spkcache_frames = 0, fifo_frames = 0;  // state sizes BEFORE this chunk
    int window_frames = 0;                     // spkcache + fifo + t3
    // Tailfix route (NeMo tail-window semantics): 0 = production path
    // (whole-window stem, feat_len=-1); >0 = padded masked tail window with
    // this many VALID mel rows (feat_len), emission trimmed to valid rows.
    int tail_feat_len = 0;
};

class DiarEngine {
public:
    // wires the weight arena's mel basis into the FE (mirrors DiarModel's
    // ctor); throws std::invalid_argument on cfg mismatches (feat_in vs FE
    // pins, geometry budget vs pos_emb_max_len).
    DiarEngine(SortformerWeights weights, EngineConfig cfg);

    // Appends samples and runs every chunk whose window is fully covered.
    // Silent no-op after finish() (upstream post-finalize contract).
    void feed_audio(const float* samples, std::size_t n_samples);

    // Appends the geometric preemphasis decay tail (n_fft/2 samples) so the
    // post-preemphasis pad is exactly zero — NeMo constant right-pad parity,
    // bit-for-bit (upstream DiarStream::finish pin) — then drains the tail
    // chunks.
    void finish();

    std::int64_t n_frames() const { return static_cast<std::int64_t>(probs_.size()) / n_spk_; }
    std::int64_t total_mel_frames() const { return mel_produced(); }

    // K5 comparison face: AOSC-emitted chain (pre-BirthGate, relabel-free).
    const std::vector<float>& pre_gate_probs() const { return pre_gate_; }
    // K6 comparison face: BirthGate timeline (establishment/relabel applied).
    const std::vector<float>& post_gate_probs() const { return probs_; }

    FrameProbabilities post_gate_frame_probabilities() const;
    std::vector<Segment> segments() const;

    // Forwarded verbatim into sortformer_run_chunk for every chunk (each tap
    // name fires once per chunk; the sink sees the chunk order in call
    // order). K5 stage-isolation diagnostics — production never sets one.
    void set_tap_sink(TapSink* sink) { tap_sink_ = sink; }

    const std::vector<ChunkLedgerEntry>& chunk_ledger() const { return ledger_; }

    // Full copy of the AOSC state — the closed-loop comparison face for 3.3
    // K5: the m2-ref npz stores spkcache_after / fifo_after / mean_sil_emb /
    // n_sil_frames per chunk, so the free-running kernel walks chunk by chunk
    // against the state itself, not just the emitted probs. Empty vectors are
    // legal (state empty). Diagnostic copies only; production never calls it.
    struct AoscSnapshot {
        int spk_frames = 0;
        int fifo_frames = 0;
        long long silence_frames = 0;
        std::vector<float> spkcache;  // spk_frames x emb_dim
        std::vector<float> fifo;      // fifo_frames x emb_dim
        std::vector<float> mean_sil;  // emb_dim
        std::vector<float> spkcache_preds;  // spk_frames x num_speakers (empty unless valid)
    };
    AoscSnapshot aosc_snapshot() const;

    // When set, the engine appends one AoscSnapshot after EVERY chunk's
    // aosc_.update inside run_one_chunk — index-aligned with chunk_ledger().
    // Ownership stays with the caller (kernel runner buffer). Production
    // leaves it null; the copy cost is then zero.
    void set_aosc_recorder(std::vector<AoscSnapshot>* recorder) { aosc_recorder_ = recorder; }

private:
    void ensure_mel();
    bool run_one_chunk(bool force, bool final_flush);
    void run_ready_chunks(bool end_of_stream);
    std::int64_t mel_produced() const {
        return mel_base_ + static_cast<std::int64_t>(mel_buf_.size()) / n_mels_;
    }

    SortformerWeights weights_;
    EngineConfig cfg_;
    MelSpectrogramExtractor fe_;
    AoscState aosc_;
    ChannelBirthGate gate_;

    int n_spk_ = 0;
    int sub_ = 8;
    int n_mels_ = 0;
    bool finished_ = false;

    std::vector<float> audio_buf_;
    std::size_t audio_base_ = 0;
    std::vector<float> mel_buf_;
    std::int64_t mel_base_ = 0;
    std::int64_t mel_consumed_ = 0;

    std::vector<float> probs_;     // post-gate timeline (frames x n_spk)
    std::vector<float> pre_gate_;  // pre-gate emitted chain (frames x n_spk)
    std::vector<ChunkLedgerEntry> ledger_;
    TapSink* tap_sink_ = nullptr;
    std::vector<AoscSnapshot>* aosc_recorder_ = nullptr;
};

struct OfflineDiarizationResult {
    std::vector<float> preds;  // [t_enc, n_spk]; no AOSC/BirthGate upstream
    int n_frames = 0;
    int n_mel = 0;
};

// Full-window offline path (upstream DiarModel::diarize_offline): peak
// normalize x * 1/(max(x)+1e-3), single FE compute (reflect-left), one
// run_chunk with empty state. Pre-gate == post-gate on this path.
OfflineDiarizationResult diarize_offline(const SortformerWeights& weights,
    const float* audio, std::size_t n_samples);

}  // namespace diar
