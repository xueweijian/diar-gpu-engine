#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace diar {

// Frame-major speaker activity probabilities. This is deliberately independent
// of any transcription or ASR representation.
class FrameProbabilities {
public:
    FrameProbabilities() = default;
    FrameProbabilities(std::size_t frames, std::size_t speakers);

    std::size_t frames() const noexcept { return frames_; }
    std::size_t speakers() const noexcept { return speakers_; }
    const std::vector<float>& values() const noexcept { return values_; }
    std::vector<float>& values() noexcept { return values_; }

    float& at(std::size_t frame, std::size_t speaker);
    float at(std::size_t frame, std::size_t speaker) const;
    void validate() const;

private:
    std::size_t frames_ = 0;
    std::size_t speakers_ = 0;
    std::vector<float> values_;
};

struct SegmentationConfig {
    float onset = 0.5F;
    float offset = 0.5F;
    double frame_duration_sec = 0.08;
    double pad_onset_sec = 0.0;
    double pad_offset_sec = 0.0;
    double min_gap_sec = 0.0;
    double min_duration_sec = 0.0;
};

struct Segment {
    double start_sec = 0.0;
    double end_sec = 0.0;
    std::uint32_t speaker = 0; // Public labels are 1-based.
};

// Sortformer v2 frontend configuration, mirroring the pinned upstream
// (NeMo-Speech.cpp a5b6953: sortformer_model.h SortformerModelConfig FE
// block + src/asr/features/fe.h MelSpecConfig + the diarizer's own wiring
// in src/asr/diar/diar_pipeline.cpp make_fe_cfg). Only the fields the M2
// engine needs to reproduce byte-compatible mel frames are listed.
// Diar-specific wiring (differs from MelSpecConfig defaults and from the
// ASR path — verified in diar_pipeline.cpp, not assumed):
//   stft_center_window=true + hann_periodic=false (NeMo torch.stft parity);
//   per-feature normalize=false (FE-internal); offline peak-normalizes the
//   waveform first (x * 1/(max(x)+1e-3)), streaming does NOT.
struct FrontendConfig {
    int sample_rate = 16000;
    float window_size_sec = 0.025F;
    float window_stride_sec = 0.01F;
    int n_fft = 512;
    int n_mels = 128;
    float preemph = 0.97F;
    // NeMo FilterbankFeatures "add"-type guard: silence bins land on the
    // log floor, so this exact 2^-24 value matters for parity, not just
    // any small epsilon.
    float log_zero_guard = 5.9604645e-8F;
    // Diar wiring: centered STFT + symmetric (non-periodic) Hann.
    bool hann_periodic = false;
    bool center_window = true;
    bool reflect_pad_left = true;
    // FE-internal per-feature normalization is off; normalization lives
    // outside the FE (offline peak gain only, streaming none).
    bool normalize_per_feature = false;
    // Offline-only peak normalization: x * 1/(max(x) + eps), eps = 1e-3.
    bool offline_peak_normalize = true;
    float offline_peak_eps = 1e-3F;
};

// Returns frame-major 0/1 activity after per-speaker hysteresis.
std::vector<std::uint8_t> hysteresis_activity(
    const FrameProbabilities& probabilities, const SegmentationConfig& config);

// Converts frame activity into independently tracked speaker segments. Overlap
// is preserved: two speakers may have segments covering the same time range.
std::vector<Segment> to_segments(
    const FrameProbabilities& probabilities, const SegmentationConfig& config);

// Faithful port of upstream diar_segments_from_probs
// (NeMo-Speech.cpp src/asr/diar/diar_pipeline.cpp @ a5b6953): strict
// inequalities on hysteresis edges, end clamped to the timeline total,
// strict < on gap merge, start-only sort. Use this for any upstream RTTM
// comparison; to_segments keeps the pre-existing local contract.
std::vector<Segment> upstream_segments_from_probs(
    const FrameProbabilities& probabilities, const SegmentationConfig& config);

struct ProbabilityMetrics {
    float max_abs_error = 0.0F;
    double mean_abs_error = 0.0;
    double frame_agreement = 0.0;
};

ProbabilityMetrics compare_probabilities(
    const FrameProbabilities& expected, const FrameProbabilities& actual,
    const SegmentationConfig& config);

// Streaming geometry in 80 ms encoder frames, mirroring upstream
// DiarGeometry (NeMo-Speech.cpp a5b6953 src/asr/diar/aosc_state.h).
// Defaults ARE the streaming preset (riva_streaming returns {}); the
// offline preset is still AOSC streaming with larger chunks/caches —
// it is NOT the full-attention --offline path (see diar_pipeline.h).
struct StreamGeometry {
    int spkcache_len = 160;
    int fifo_len = 80;
    int chunk_len = 20;
    int spkcache_update_period = 80;
    int chunk_left_context = 0;
    int chunk_right_context = 0;

    static StreamGeometry streaming();
    static StreamGeometry offline_preset();
};

// Throws std::invalid_argument on unknown names ("streaming" | "offline").
StreamGeometry stream_geometry_preset(const char* name);

// Mirrors DiarGeometry::validate: positive chunk/update sizes, non-negative
// fifo/contexts, speaker-cache budget (1+sil_frames_per_spk)*n_spk, and the
// rel-pos table limit. Throws std::invalid_argument with a "diar geometry: "
// prefix like upstream.
void validate_stream_geometry(
    const StreamGeometry& geometry, int num_speakers, int sil_frames_per_spk,
    int pos_emb_max_len);

// AOSC scoring constants, model-tied (from GGUF sortformer.scoring.*).
// Pinned against NeMo-Speech.cpp a5b6953 sortformer_model.h DiarScoringConfig.
struct AoscScoringConfig {
    int sil_frames_per_spk = 3;
    float pred_score_threshold = 0.25F;
    float scores_boost_latest = 0.05F;
    float sil_threshold = 0.2F;
    float strong_boost_rate = 0.75F;
    float weak_boost_rate = 1.5F;
    float min_pos_scores_rate = 0.5F;
};

// Arrival-Order Speaker Cache streaming state, mirroring upstream AoscState
// (NeMo-Speech.cpp a5b6953 src/asr/diar/aosc_state.h/.cpp — host-side port
// of NeMo's SortformerModules.streaming_update / _compress_spkcache).
// Pure host-side float logic, no model dependency: update/compress move
// embeddings and predictions between FIFO and speaker cache. Embeddings are
// opaque blobs here (emb_dim floats per frame); the engine fills them in M2.
class AoscState {
public:
    AoscState(
        const StreamGeometry& geometry, const AoscScoringConfig& scoring, int num_speakers,
        int emb_dim);

    // One streaming update after a model chunk. chunk_embs holds t3 frames of
    // emb_dim (including lc/rc, trimmed internally); preds holds
    // (spkcache_frames + fifo_frames + t3) x num_speakers. Returns the
    // emitted chunk predictions ((t3 - lc - rc) x num_speakers), empty when
    // the valid window is non-positive.
    std::vector<float> update(
        const float* chunk_embs, int t3, const float* preds, int lc, int rc);

    int spkcache_frames() const { return spk_frames_; }
    int fifo_frames() const { return fifo_frames_; }
    const std::vector<float>& spkcache() const { return spkcache_; }
    const std::vector<float>& fifo() const { return fifo_; }
    bool spkcache_preds_valid() const { return !spkcache_preds_.empty(); }
    const std::vector<float>& spkcache_preds() const { return spkcache_preds_; }
    const std::vector<float>& mean_sil_emb() const { return mean_sil_emb_; }
    long long silence_frames() const { return silence_frames_; }

private:
    void accumulate_silence(const float* embs, const float* preds, int n);
    void compress(const std::vector<float>& cache_preds);

    StreamGeometry geometry_;
    AoscScoringConfig scoring_;
    int num_speakers_;
    int emb_dim_;

    std::vector<float> spkcache_;
    std::vector<float> spkcache_preds_;
    int spk_frames_ = 0;
    std::vector<float> fifo_;
    int fifo_frames_ = 0;
    std::vector<float> mean_sil_emb_;
    long long silence_frames_ = 0;
};

// Transient-channel guard, mirroring upstream ChannelBirthGate
// (NeMo-Speech.cpp a5b6953 src/asr/diar/aosc_state.h/.cpp).
// A channel becomes established only after a short clean run
// (4 frames >= 0.95 while established channels stay <= 0.02) or a
// fading handoff (20 frames >= 0.90 while established stay <= 0.15);
// an episode gap of 25 frames resets the counters. Until established,
// a channel's probability is folded into the strongest established
// channel (relabel). When a channel newly establishes, the last
// 128 frames of the timeline are rewritten from raw and relabeled.
// Pure host-side float logic, no model dependency.
class ChannelBirthGate {
public:
    explicit ChannelBirthGate(int num_speakers);

    void reset();
    // Appends raw chunk probs (multiple of num_speakers) to timeline,
    // relabeling new frames and revising the tail on new establishments.
    // Throws std::invalid_argument on an incomplete probability frame.
    void append(const std::vector<float>& raw, std::vector<float>& timeline);
    bool is_established(int speaker) const;

private:
    bool observe(const float* probs);
    void relabel(float* probs) const;
    void push_raw(const float* probs);

    int num_speakers_;
    long long frame_ = 0;
    std::vector<std::uint8_t> established_;
    std::vector<int> clean_frames_;
    std::vector<int> fading_frames_;
    std::vector<long long> last_win_;
    std::vector<float> raw_ring_;
};

} // namespace diar
