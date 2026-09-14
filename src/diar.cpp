#include "diar/diar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace diar {
namespace {

std::size_t checked_size(std::size_t frames, std::size_t speakers) {
    if (frames != 0 && speakers > std::numeric_limits<std::size_t>::max() / frames) {
        throw std::length_error("frame probability matrix is too large");
    }
    return frames * speakers;
}

void validate_config(const SegmentationConfig& config) {
    if (!std::isfinite(config.onset) || !std::isfinite(config.offset) ||
        config.onset < 0.0F || config.onset > 1.0F || config.offset < 0.0F ||
        config.offset > 1.0F || config.offset > config.onset) {
        throw std::invalid_argument("invalid hysteresis thresholds");
    }
    if (!std::isfinite(config.frame_duration_sec) || config.frame_duration_sec <= 0.0 ||
        !std::isfinite(config.pad_onset_sec) || config.pad_onset_sec < 0.0 ||
        !std::isfinite(config.pad_offset_sec) || config.pad_offset_sec < 0.0 ||
        !std::isfinite(config.min_gap_sec) || config.min_gap_sec < 0.0 ||
        !std::isfinite(config.min_duration_sec) || config.min_duration_sec < 0.0) {
        throw std::invalid_argument("invalid segmentation timing configuration");
    }
}

std::size_t index_of(std::size_t frame, std::size_t speaker, std::size_t speakers) {
    return frame * speakers + speaker;
}

} // namespace

FrameProbabilities::FrameProbabilities(std::size_t frames, std::size_t speakers)
    : frames_(frames), speakers_(speakers), values_(checked_size(frames, speakers), 0.0F) {}

float& FrameProbabilities::at(std::size_t frame, std::size_t speaker) {
    if (frame >= frames_ || speaker >= speakers_) {
        throw std::out_of_range("frame probability index out of range");
    }
    return values_[index_of(frame, speaker, speakers_)];
}

float FrameProbabilities::at(std::size_t frame, std::size_t speaker) const {
    if (frame >= frames_ || speaker >= speakers_) {
        throw std::out_of_range("frame probability index out of range");
    }
    return values_[index_of(frame, speaker, speakers_)];
}

void FrameProbabilities::validate() const {
    if (values_.size() != checked_size(frames_, speakers_)) {
        throw std::invalid_argument("frame probability storage has the wrong size");
    }
    for (float value : values_) {
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
            throw std::invalid_argument("frame probability must be finite and in [0, 1]");
        }
    }
}

std::vector<std::uint8_t> hysteresis_activity(
    const FrameProbabilities& probabilities, const SegmentationConfig& config) {
    probabilities.validate();
    validate_config(config);

    std::vector<std::uint8_t> activity(probabilities.values().size(), 0);
    for (std::size_t speaker = 0; speaker < probabilities.speakers(); ++speaker) {
        bool active = false;
        for (std::size_t frame = 0; frame < probabilities.frames(); ++frame) {
            const float probability = probabilities.at(frame, speaker);
            if (!active && probability >= config.onset) {
                active = true;
            } else if (active && probability < config.offset) {
                active = false;
            }
            activity[index_of(frame, speaker, probabilities.speakers())] =
                static_cast<std::uint8_t>(active ? 1 : 0);
        }
    }
    return activity;
}

std::vector<Segment> to_segments(
    const FrameProbabilities& probabilities, const SegmentationConfig& config) {
    const auto activity = hysteresis_activity(probabilities, config);
    std::vector<Segment> segments;

    for (std::size_t speaker = 0; speaker < probabilities.speakers(); ++speaker) {
        struct Run {
            std::size_t begin;
            std::size_t end;
        };
        std::vector<Run> runs;
        std::size_t frame = 0;
        while (frame < probabilities.frames()) {
            while (frame < probabilities.frames() &&
                   activity[index_of(frame, speaker, probabilities.speakers())] == 0) {
                ++frame;
            }
            if (frame == probabilities.frames()) {
                break;
            }
            const std::size_t begin = frame;
            while (frame < probabilities.frames() &&
                   activity[index_of(frame, speaker, probabilities.speakers())] != 0) {
                ++frame;
            }
            runs.push_back({begin, frame});
        }

        std::vector<Run> merged;
        for (const Run& run : runs) {
            if (!merged.empty()) {
                const double gap = static_cast<double>(run.begin - merged.back().end) *
                    config.frame_duration_sec;
                if (gap <= config.min_gap_sec) {
                    merged.back().end = run.end;
                    continue;
                }
            }
            merged.push_back(run);
        }

        for (const Run& run : merged) {
            const double raw_start = static_cast<double>(run.begin) * config.frame_duration_sec;
            const double raw_end = static_cast<double>(run.end) * config.frame_duration_sec;
            double start = std::max(0.0, raw_start - config.pad_onset_sec);
            double end = raw_end + config.pad_offset_sec;
            if (end - start < config.min_duration_sec) {
                continue;
            }
            segments.push_back({start, end, static_cast<std::uint32_t>(speaker + 1)});
        }
    }

    std::sort(segments.begin(), segments.end(), [](const Segment& lhs, const Segment& rhs) {
        return lhs.start_sec < rhs.start_sec;
    });
    return segments;
}

// Upstream-faithful port of NeMo-Speech.cpp's diar_segments_from_probs
// (src/asr/diar/diar_pipeline.cpp @ a5b6953). Differences from to_segments:
//  - hysteresis edges are strict inequalities (p > onset opens, p < offset
//    closes; equality keeps the current state), while to_segments opens on
//    >= and closes on <;
//  - segment end clamps to the timeline total (n * frame_duration_sec);
//  - merge condition is strict (< min_gap_sec): a gap exactly equal to the
//    threshold does NOT merge, while to_segments merges on <=;
//  - final sort is by start time only (std::sort is not stable), matching
//    upstream's single-key comparator.
// New code must use this port when comparing against upstream RTTMs;
// to_segments stays for the existing local contract tests.
std::vector<Segment> upstream_segments_from_probs(
    const FrameProbabilities& probabilities, const SegmentationConfig& config) {
    probabilities.validate();
    validate_config(config);
    const std::size_t frames = probabilities.frames();
    const std::size_t speakers = probabilities.speakers();
    const double frame_sec = config.frame_duration_sec;
    const double total = static_cast<double>(frames) * frame_sec;
    std::vector<Segment> out;
    for (std::size_t speaker = 0; speaker < speakers; ++speaker) {
        std::vector<std::pair<double, double>> segs;
        bool active = false;
        std::size_t start = 0;
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const float p = probabilities.at(frame, speaker);
            if (!active && p > config.onset) {
                active = true;
                start = frame;
            } else if (active && p < config.offset) {
                active = false;
                segs.emplace_back(
                    std::max(0.0, static_cast<double>(start) * frame_sec - config.pad_onset_sec),
                    std::min(total, static_cast<double>(frame) * frame_sec + config.pad_offset_sec));
            }
        }
        if (active) {
            segs.emplace_back(
                std::max(0.0, static_cast<double>(start) * frame_sec - config.pad_onset_sec), total);
        }
        std::vector<std::pair<double, double>> merged;
        for (const auto& sg : segs) {
            if (!merged.empty() && sg.first - merged.back().second < config.min_gap_sec) {
                merged.back().second = std::max(merged.back().second, sg.second);
            } else {
                merged.push_back(sg);
            }
        }
        for (const auto& sg : merged) {
            if (sg.second - sg.first >= config.min_duration_sec) {
                out.push_back({sg.first, sg.second, static_cast<std::uint32_t>(speaker + 1)});
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const Segment& lhs, const Segment& rhs) {
        return lhs.start_sec < rhs.start_sec;
    });
    return out;
}

ProbabilityMetrics compare_probabilities(
    const FrameProbabilities& expected, const FrameProbabilities& actual,
    const SegmentationConfig& config) {
    expected.validate();
    actual.validate();
    validate_config(config);
    if (expected.frames() != actual.frames() || expected.speakers() != actual.speakers()) {
        throw std::invalid_argument("probability matrices have different shapes");
    }

    ProbabilityMetrics metrics;
    if (expected.values().empty()) {
        metrics.frame_agreement = 1.0;
        return metrics;
    }
    double total_error = 0.0;
    const auto expected_activity = hysteresis_activity(expected, config);
    const auto actual_activity = hysteresis_activity(actual, config);
    std::size_t equal = 0;
    for (std::size_t i = 0; i < expected.values().size(); ++i) {
        const float error = std::fabs(expected.values()[i] - actual.values()[i]);
        metrics.max_abs_error = std::max(metrics.max_abs_error, error);
        total_error += static_cast<double>(error);
        if (expected_activity[i] == actual_activity[i]) {
            ++equal;
        }
    }
    metrics.mean_abs_error = total_error / static_cast<double>(expected.values().size());
    metrics.frame_agreement = static_cast<double>(equal) /
        static_cast<double>(expected_activity.size());
    return metrics;
}

} // namespace diar
