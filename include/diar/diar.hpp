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

} // namespace diar
