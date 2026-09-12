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

// Returns frame-major 0/1 activity after per-speaker hysteresis.
std::vector<std::uint8_t> hysteresis_activity(
    const FrameProbabilities& probabilities, const SegmentationConfig& config);

// Converts frame activity into independently tracked speaker segments. Overlap
// is preserved: two speakers may have segments covering the same time range.
std::vector<Segment> to_segments(
    const FrameProbabilities& probabilities, const SegmentationConfig& config);

struct ProbabilityMetrics {
    float max_abs_error = 0.0F;
    double mean_abs_error = 0.0;
    double frame_agreement = 0.0;
};

ProbabilityMetrics compare_probabilities(
    const FrameProbabilities& expected, const FrameProbabilities& actual,
    const SegmentationConfig& config);

} // namespace diar
