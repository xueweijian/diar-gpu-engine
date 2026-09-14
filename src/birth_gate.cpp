// ChannelBirthGate port — see include/diar/diar.hpp for the contract.
// Pinned against NeMo-Speech.cpp a5b6953 src/asr/diar/aosc_state.cpp
// (the ChannelBirthGate block only; AoscState lives in aosc.cpp).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "diar/diar.hpp"

namespace diar {
namespace {

// Birth-gate constants, model-tied (NeMo sortformer diarizer defaults).
constexpr float kBirthSpeech = 0.30F;
constexpr float kBirthClean = 0.95F;
constexpr float kEstablishedQuiet = 0.02F;
constexpr float kBirthFading = 0.90F;
constexpr float kEstablishedFading = 0.15F;
constexpr int kBirthCleanFrames = 4;
constexpr int kBirthFadingFrames = 20;
constexpr int kBirthEpisodeGapFrames = 25;
constexpr int kBirthRevisionFrames = 128;

}  // namespace

ChannelBirthGate::ChannelBirthGate(int num_speakers) : num_speakers_(num_speakers) {
    if (num_speakers_ <= 0) {
        throw std::invalid_argument("ChannelBirthGate: n_spk must be positive");
    }
    reset();
}

void ChannelBirthGate::reset() {
    frame_ = 0;
    established_.assign(static_cast<std::size_t>(num_speakers_), false);
    clean_frames_.assign(static_cast<std::size_t>(num_speakers_), 0);
    fading_frames_.assign(static_cast<std::size_t>(num_speakers_), 0);
    last_win_.assign(
        static_cast<std::size_t>(num_speakers_), std::numeric_limits<long long>::min() / 2);
    raw_ring_.clear();
}

bool ChannelBirthGate::observe(const float* probs) {
    int winner = 0;
    for (int s = 1; s < num_speakers_; s++) {
        if (probs[s] > probs[winner]) {
            winner = s;
        }
    }

    bool changed = false;
    if (probs[winner] >= kBirthSpeech && !established_[static_cast<std::size_t>(winner)]) {
        if (frame_ - last_win_[static_cast<std::size_t>(winner)] > kBirthEpisodeGapFrames) {
            clean_frames_[static_cast<std::size_t>(winner)] = 0;
            fading_frames_[static_cast<std::size_t>(winner)] = 0;
        }
        last_win_[static_cast<std::size_t>(winner)] = frame_;

        float established_prob = 0.0F;
        for (int s = 0; s < num_speakers_; s++) {
            if (established_[static_cast<std::size_t>(s)]) {
                established_prob = std::max(established_prob, probs[s]);
            }
        }
        if (probs[winner] >= kBirthClean && established_prob <= kEstablishedQuiet) {
            clean_frames_[static_cast<std::size_t>(winner)]++;
        }
        if (probs[winner] >= kBirthFading && established_prob <= kEstablishedFading) {
            fading_frames_[static_cast<std::size_t>(winner)]++;
        }
        if (clean_frames_[static_cast<std::size_t>(winner)] >= kBirthCleanFrames ||
            fading_frames_[static_cast<std::size_t>(winner)] >= kBirthFadingFrames) {
            established_[static_cast<std::size_t>(winner)] = true;
            changed = true;
        }
    }
    frame_++;
    return changed;
}

void ChannelBirthGate::relabel(float* probs) const {
    int target = -1;
    for (int s = 0; s < num_speakers_; s++) {
        if (established_[static_cast<std::size_t>(s)] &&
            (target < 0 || probs[s] > probs[target])) {
            target = s;
        }
    }
    if (target < 0) {
        return;
    }

    for (int s = 0; s < num_speakers_; s++) {
        if (!established_[static_cast<std::size_t>(s)] && probs[s] > 0.0F) {
            probs[target] = std::max(probs[target], probs[s]);
            probs[s] = 0.0F;
        }
    }
}

void ChannelBirthGate::push_raw(const float* probs) {
    raw_ring_.insert(raw_ring_.end(), probs, probs + num_speakers_);
    const std::size_t cap = static_cast<std::size_t>(kBirthRevisionFrames) * num_speakers_;
    if (raw_ring_.size() > cap) {
        raw_ring_.erase(raw_ring_.begin(), raw_ring_.begin() + num_speakers_);
    }
}

void ChannelBirthGate::append(const std::vector<float>& raw, std::vector<float>& timeline) {
    if (raw.size() % static_cast<std::size_t>(num_speakers_) != 0) {
        throw std::invalid_argument("ChannelBirthGate: incomplete probability frame");
    }

    bool changed = false;
    for (std::size_t i = 0; i < raw.size(); i += num_speakers_) {
        push_raw(raw.data() + i);
        changed |= observe(raw.data() + i);
    }

    const std::size_t old_size = timeline.size();
    timeline.insert(timeline.end(), raw.begin(), raw.end());
    for (std::size_t i = old_size; i < timeline.size(); i += num_speakers_) {
        relabel(timeline.data() + i);
    }

    if (!changed) {
        return;
    }
    const std::size_t window = std::min(raw_ring_.size(), timeline.size());
    const std::size_t ring_offset = raw_ring_.size() - window;
    const std::size_t timeline_offset = timeline.size() - window;
    std::copy(
        raw_ring_.begin() + ring_offset, raw_ring_.end(), timeline.begin() + timeline_offset);
    for (std::size_t i = timeline_offset; i < timeline.size(); i += num_speakers_) {
        relabel(timeline.data() + i);
    }
}

bool ChannelBirthGate::is_established(int speaker) const {
    return speaker >= 0 && speaker < num_speakers_ && established_[speaker];
}

}  // namespace diar
