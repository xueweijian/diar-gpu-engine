#include "diar/diar.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void expect_near(double actual, double expected, double tolerance, const std::string& message) {
    expect(std::fabs(actual - expected) <= tolerance,
           message + " actual=" + std::to_string(actual) +
               " expected=" + std::to_string(expected));
}

void test_hysteresis() {
    diar::FrameProbabilities probabilities(5, 1);
    probabilities.values() = {0.1F, 0.7F, 0.55F, 0.45F, 0.39F};
    diar::SegmentationConfig config;
    config.onset = 0.6F;
    config.offset = 0.4F;
    const auto activity = diar::hysteresis_activity(probabilities, config);
    expect(activity == std::vector<std::uint8_t>({0, 1, 1, 1, 0}),
           "hysteresis should keep activity between onset and offset");
    const auto segments = diar::to_segments(probabilities, config);
    expect(segments.size() == 1, "hysteresis should produce one segment");
    expect_near(segments[0].start_sec, 0.08, 1e-9, "segment start");
    expect_near(segments[0].end_sec, 0.32, 1e-9, "segment end");
    expect(segments[0].speaker == 1, "speaker labels are one-based");
}

void test_overlap_and_sorting() {
    diar::FrameProbabilities probabilities(4, 2);
    probabilities.values() = {
        0.0F, 0.0F,
        0.8F, 0.0F,
        0.7F, 0.9F,
        0.0F, 0.8F,
    };
    diar::SegmentationConfig config;
    config.onset = 0.5F;
    config.offset = 0.3F;
    const auto segments = diar::to_segments(probabilities, config);
    expect(segments.size() == 2, "overlap should retain one run per speaker");
    expect(segments[0].speaker == 1 && segments[1].speaker == 2,
           "segments should be sorted by time and speaker");
    expect_near(segments[0].start_sec, 0.08, 1e-9, "speaker one overlap start");
    expect_near(segments[0].end_sec, 0.24, 1e-9, "speaker one overlap end");
    expect_near(segments[1].start_sec, 0.16, 1e-9, "speaker two overlap start");
    expect_near(segments[1].end_sec, 0.32, 1e-9, "speaker two overlap end");
}

void test_gap_fill_and_min_duration() {
    diar::FrameProbabilities probabilities(3, 1);
    probabilities.values() = {0.9F, 0.0F, 0.9F};
    diar::SegmentationConfig config;
    config.onset = 0.5F;
    config.offset = 0.4F;
    config.min_gap_sec = 0.08;
    const auto merged = diar::to_segments(probabilities, config);
    expect(merged.size() == 1, "short inactive gap should be filled");
    expect_near(merged[0].end_sec, 0.24, 1e-9, "filled segment end");

    config.min_duration_sec = 0.25;
    expect(diar::to_segments(probabilities, config).empty(),
           "minimum duration should drop the short merged segment");
}

void test_metrics_and_validation() {
    diar::FrameProbabilities expected(2, 1);
    expected.values() = {0.0F, 1.0F};
    diar::FrameProbabilities actual(2, 1);
    actual.values() = {0.1F, 0.8F};
    diar::SegmentationConfig config;
    const auto metrics = diar::compare_probabilities(expected, actual, config);
    expect_near(metrics.max_abs_error, 0.2, 1e-6, "maximum probability error");
    expect_near(metrics.mean_abs_error, 0.15, 1e-6, "mean probability error");
    expect_near(metrics.frame_agreement, 1.0, 1e-12, "frame agreement");

    bool threw = false;
    try {
        diar::FrameProbabilities invalid(1, 1);
        invalid.values()[0] = 1.1F;
        invalid.validate();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "probabilities outside [0,1] must be rejected");

    threw = false;
    try {
        config.offset = 0.9F;
        config.onset = 0.5F;
        diar::hysteresis_activity(expected, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "offset above onset must be rejected");
}

} // namespace

int main() {
    test_hysteresis();
    test_overlap_and_sorting();
    test_gap_fill_and_min_duration();
    test_metrics_and_validation();
    std::cout << "PASS: pure diarization core tests\n";
    return 0;
}
