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

void test_metrics_and_validation() {    diar::FrameProbabilities expected(2, 1);
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

void test_upstream_port_matches_reference_vectors() {
    // Oracle vectors generated from upstream diar_segments_from_probs semantics
    // (NeMo-Speech.cpp src/asr/diar/diar_pipeline.cpp @ a5b6953) with the
    // checkpoint's published callhome postprocessing
    // (scripts/asr/score_diar_der.py POSTPROC_CALLHOME) and 80 ms frames.
    // probs: 10 frames x 2 speakers; speaker 0 opens late, speaker 1 opens
    // exactly on the onset boundary (tests strict >) and has a gap exactly
    // equal to min_duration_off (tests strict < merge).
    diar::FrameProbabilities probs(10, 2);
    probs.values() = {
        0.00F, 0.641F,  // f0: s1 exactly on onset -> must NOT open (strict >)
        0.00F, 0.642F,  // f1: s1 opens
        0.70F, 0.700F,  // f2: s0 opens
        0.70F, 0.000F,  // f3: s1 closes at f3 (0.0 < 0.561)
        0.00F, 0.900F,  // f4: s0 closes at f4; s1 reopens
        0.00F, 0.900F,  // f5
        0.00F, 0.000F,  // f6: s1 closes at f6
        0.00F, 0.000F,  // f7
        0.80F, 0.000F,  // f8: s0 reopens
        0.80F, 0.000F,  // f9: trailing active run -> end clamps to total 0.8
    };
    diar::SegmentationConfig cfg;
    cfg.onset = 0.641F;
    cfg.offset = 0.561F;
    cfg.frame_duration_sec = 0.08;
    cfg.pad_onset_sec = 0.229;
    cfg.pad_offset_sec = 0.079;
    cfg.min_gap_sec = 0.296;
    cfg.min_duration_sec = 0.511;

    const auto segs = diar::upstream_segments_from_probs(probs, cfg);
    // s0: run [f2,f4) -> (max(0,0.16-0.229), min(0.8,0.32+0.079)) = (0, 0.399);
    // run [f8,f10) active at end -> (0.64-0.229, 0.8) = (0.411, 0.8).
    // gap 0.411-0.399 = 0.012 < 0.296 -> merge -> (0, 0.8), kept (>= 0.511).
    // s1: run [f1,f3) -> (0.08-0.229 clamp 0, 0.24+0.079) = (0, 0.319);
    // run [f4,f6) -> (0.32-0.229, 0.48+0.079) = (0.091, 0.559).
    // gap 0.091-0.319 < 0 -> overlap -> merge -> (0, 0.559), kept.
    expect(segs.size() == 2, "upstream port should emit two merged segments");
    expect(segs[0].speaker == 1, "first segment is speaker one");
    expect_near(segs[0].start_sec, 0.0, 1e-9, "speaker one start");
    expect_near(segs[0].end_sec, 0.8, 1e-9, "speaker one end");
    expect(segs[1].speaker == 2, "second segment is speaker two");
    expect_near(segs[1].start_sec, 0.0, 1e-9, "speaker two start");
    expect_near(segs[1].end_sec, 0.559, 1e-9, "speaker two end");

    // Boundary contract, each in isolation:
    // (a) p == onset must not open; (b) gap == min_gap must not merge;
    // (c) trailing run ends at total, not past it.
    diar::FrameProbabilities edge(2, 1);
    edge.values() = {0.641F, 0.0F};
    expect(diar::upstream_segments_from_probs(edge, cfg).empty(),
           "probability exactly on onset must not open a segment");

    diar::FrameProbabilities gap(4, 1);
    gap.values() = {0.9F, 0.0F, 0.0F, 0.9F};
    diar::SegmentationConfig gap_cfg;
    gap_cfg.onset = 0.5F;
    gap_cfg.offset = 0.4F;
    gap_cfg.frame_duration_sec = 0.1;
    gap_cfg.min_gap_sec = 0.2;  // gap f1..f3 = 0.2 == threshold -> no merge
    const auto gap_segs = diar::upstream_segments_from_probs(gap, gap_cfg);
    expect(gap_segs.size() == 2, "gap equal to threshold must not merge");
}

void test_frontend_config_matches_upstream_diar_wiring() {
    // Pinned against NeMo-Speech.cpp a5b6953:
    // sortformer_model.h SortformerModelConfig (FE values) +
    // src/asr/diar/diar_pipeline.cpp make_fe_cfg (diar wiring, NOT the
    // MelSpecConfig defaults and NOT the ASR path's legacy placement).
    // If upstream changes any value, this test forces us to update the
    // struct instead of silently drifting out of parity.
    diar::FrontendConfig cfg;
    expect(cfg.sample_rate == 16000, "frontend sample rate");
    expect(cfg.n_fft == 512, "frontend n_fft");
    expect(cfg.n_mels == 128, "frontend n_mels (diar, not the 80 default)");
    expect_near(cfg.window_size_sec, 0.025, 1e-9, "frontend window size");
    expect_near(cfg.window_stride_sec, 0.01, 1e-9, "frontend window stride");
    expect_near(cfg.preemph, 0.97, 1e-6, "frontend preemphasis");
    expect_near(cfg.log_zero_guard, 5.9604645e-8, 1e-15, "frontend log guard 2^-24");
    // Diar wiring — the three fields that differ from naive defaults:
    expect(cfg.center_window, "diar uses centered STFT");
    expect(!cfg.hann_periodic, "diar uses symmetric (non-periodic) Hann");
    expect(!cfg.normalize_per_feature, "FE-internal normalize is off for diar");
    expect(cfg.offline_peak_normalize, "offline peak-normalizes the waveform");
    expect_near(cfg.offline_peak_eps, 1e-3, 1e-9, "offline peak eps");
}

} // namespace

int main() {
    test_hysteresis();
    test_overlap_and_sorting();
    test_gap_fill_and_min_duration();
    test_metrics_and_validation();
    test_upstream_port_matches_reference_vectors();
    test_frontend_config_matches_upstream_diar_wiring();
    std::cout << "PASS: pure diarization core tests\n";
    return 0;
}
