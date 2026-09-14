// AOSC streaming state port — see include/diar/diar.hpp for the contract.
// Pinned against NeMo-Speech.cpp a5b6953 src/asr/diar/aosc_state.cpp
// (ChannelBirthGate excluded: it rewrites probabilities downstream and is
// a separate port; this file is only the embedding/prediction cache).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "diar/diar.hpp"

namespace diar {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();
// NeMo's placeholder for disabled top-k picks (sortformer_modules.max_index).
constexpr long long kMaxIndex = 99999;

// Indices of the k largest values in column `spk` of row-major `scores`
// (n x n_spk). Ties break toward the lower frame index.
std::vector<int> topk_column(const std::vector<float>& scores, int n, int n_spk, int spk, int k) {
    std::vector<int> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        const float sa = scores[static_cast<std::size_t>(a) * n_spk + spk];
        const float sb = scores[static_cast<std::size_t>(b) * n_spk + spk];
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });
    idx.resize(static_cast<std::size_t>(k));
    return idx;
}

}  // namespace

AoscState::AoscState(
    const StreamGeometry& geometry, const AoscScoringConfig& scoring, int num_speakers,
    int emb_dim)
    : geometry_(geometry),
      scoring_(scoring),
      num_speakers_(num_speakers),
      emb_dim_(emb_dim) {
    mean_sil_emb_.assign(static_cast<std::size_t>(emb_dim_), 0.0F);
}

void AoscState::accumulate_silence(const float* embs, const float* preds, int n) {
    int cnt = 0;
    std::vector<double> sum(static_cast<std::size_t>(emb_dim_), 0.0);
    for (int f = 0; f < n; f++) {
        float act = 0.0F;
        for (int s = 0; s < num_speakers_; s++) {
            act += preds[static_cast<std::size_t>(f) * num_speakers_ + s];
        }
        if (act < scoring_.sil_threshold) {
            cnt++;
            for (int d = 0; d < emb_dim_; d++) {
                sum[static_cast<std::size_t>(d)] += embs[static_cast<std::size_t>(f) * emb_dim_ + d];
            }
        }
    }
    if (cnt == 0) {
        return;
    }
    const long long n_new = silence_frames_ + cnt;
    const double denom = static_cast<double>(n_new > 0 ? n_new : 1);
    for (int d = 0; d < emb_dim_; d++) {
        mean_sil_emb_[static_cast<std::size_t>(d)] = static_cast<float>(
            (static_cast<double>(mean_sil_emb_[static_cast<std::size_t>(d)]) * silence_frames_ +
             sum[static_cast<std::size_t>(d)]) /
            denom);
    }
    silence_frames_ = n_new;
}

std::vector<float> AoscState::update(
    const float* chunk_embs, int t3, const float* preds, int lc, int rc) {
    const int chunk_valid = t3 - lc - rc;
    if (chunk_valid <= 0) {
        return {};
    }

    const int l1 = spk_frames_;
    const int l2 = fifo_frames_;

    const float* fifo_preds = preds + static_cast<std::size_t>(l1) * num_speakers_;
    const float* chunk_preds = preds + static_cast<std::size_t>(l1 + l2 + lc) * num_speakers_;
    const float* chunk_valid_embs = chunk_embs + static_cast<std::size_t>(lc) * emb_dim_;

    std::vector<float> emitted(
        chunk_preds, chunk_preds + static_cast<std::size_t>(chunk_valid) * num_speakers_);

    fifo_.insert(
        fifo_.end(), chunk_valid_embs,
        chunk_valid_embs + static_cast<std::size_t>(chunk_valid) * emb_dim_);
    std::vector<float> fifo_preds_full(static_cast<std::size_t>(l2 + chunk_valid) * num_speakers_);
    std::memcpy(
        fifo_preds_full.data(), fifo_preds, static_cast<std::size_t>(l2) * num_speakers_ * 4);
    std::memcpy(
        fifo_preds_full.data() + static_cast<std::size_t>(l2) * num_speakers_, chunk_preds,
        static_cast<std::size_t>(chunk_valid) * num_speakers_ * 4);
    fifo_frames_ = l2 + chunk_valid;

    if (fifo_frames_ > geometry_.fifo_len) {
        int pop = geometry_.spkcache_update_period;
        pop = std::max(pop, chunk_valid - geometry_.fifo_len + l2);
        pop = std::min(pop, fifo_frames_);

        const float* pop_embs = fifo_.data();
        const float* pop_preds = fifo_preds_full.data();
        accumulate_silence(pop_embs, pop_preds, pop);

        spkcache_.insert(
            spkcache_.end(), pop_embs, pop_embs + static_cast<std::size_t>(pop) * emb_dim_);
        if (spkcache_preds_valid()) {
            spkcache_preds_.insert(
                spkcache_preds_.end(), pop_preds,
                pop_preds + static_cast<std::size_t>(pop) * num_speakers_);
        }
        spk_frames_ += pop;

        if (spk_frames_ > geometry_.spkcache_len && !spkcache_preds_valid()) {
            spkcache_preds_.resize(static_cast<std::size_t>(spk_frames_) * num_speakers_);
            std::memcpy(
                spkcache_preds_.data(), preds, static_cast<std::size_t>(l1) * num_speakers_ * 4);
            std::memcpy(
                spkcache_preds_.data() + static_cast<std::size_t>(l1) * num_speakers_, pop_preds,
                static_cast<std::size_t>(pop) * num_speakers_ * 4);
        }
        fifo_.erase(fifo_.begin(), fifo_.begin() + static_cast<std::size_t>(pop) * emb_dim_);
        fifo_frames_ -= pop;
        if (spk_frames_ > geometry_.spkcache_len) {
            compress(spkcache_preds_);
        }
    }
    return emitted;
}

void AoscState::compress(const std::vector<float>& cache_preds) {
    const int n = spk_frames_;
    const int cap = geometry_.spkcache_len;
    const int per_spk = cap / num_speakers_ - scoring_.sil_frames_per_spk;
    const int strong_k = static_cast<int>(std::floor(per_spk * scoring_.strong_boost_rate));
    const int weak_k = static_cast<int>(std::floor(per_spk * scoring_.weak_boost_rate));
    const int min_pos = static_cast<int>(std::floor(per_spk * scoring_.min_pos_scores_rate));
    const float log_half = std::log(0.5F);

    std::vector<float> scores(static_cast<std::size_t>(n) * num_speakers_);
    for (int f = 0; f < n; f++) {
        float sum_log1p = 0.0F;
        for (int s = 0; s < num_speakers_; s++) {
            const float p = cache_preds[static_cast<std::size_t>(f) * num_speakers_ + s];
            sum_log1p += std::log(std::max(1.0F - p, scoring_.pred_score_threshold));
        }
        for (int s = 0; s < num_speakers_; s++) {
            const float p = cache_preds[static_cast<std::size_t>(f) * num_speakers_ + s];
            const float logp = std::log(std::max(p, scoring_.pred_score_threshold));
            const float log1p = std::log(std::max(1.0F - p, scoring_.pred_score_threshold));
            scores[static_cast<std::size_t>(f) * num_speakers_ + s] =
                logp - log1p + sum_log1p - log_half;
        }
    }

    std::vector<int> pos_count(static_cast<std::size_t>(num_speakers_), 0);
    for (int f = 0; f < n; f++) {
        for (int s = 0; s < num_speakers_; s++) {
            const std::size_t i = static_cast<std::size_t>(f) * num_speakers_ + s;
            const bool is_speech = cache_preds[i] > 0.5F;
            if (!is_speech) {
                scores[i] = kNegInf;
            }
            if (scores[i] > 0.0F) {
                pos_count[static_cast<std::size_t>(s)]++;
            }
        }
    }
    for (int s = 0; s < num_speakers_; s++) {
        if (pos_count[static_cast<std::size_t>(s)] < min_pos) {
            continue;
        }
        for (int f = 0; f < n; f++) {
            const std::size_t i = static_cast<std::size_t>(f) * num_speakers_ + s;
            const bool is_speech = cache_preds[i] > 0.5F;
            if (is_speech && !(scores[i] > 0.0F)) {
                scores[i] = kNegInf;
            }
        }
    }

    if (scoring_.scores_boost_latest > 0.0F) {
        for (int f = cap; f < n; f++) {
            for (int s = 0; s < num_speakers_; s++) {
                scores[static_cast<std::size_t>(f) * num_speakers_ + s] +=
                    scoring_.scores_boost_latest;
            }
        }
    }

    for (int s = 0; s < num_speakers_; s++) {
        for (int f : topk_column(scores, n, num_speakers_, s, strong_k)) {
            scores[static_cast<std::size_t>(f) * num_speakers_ + s] -= 2.0F * log_half;
        }
    }
    for (int s = 0; s < num_speakers_; s++) {
        for (int f : topk_column(scores, n, num_speakers_, s, weak_k)) {
            scores[static_cast<std::size_t>(f) * num_speakers_ + s] -= log_half;
        }
    }

    const int n_pad = n + scoring_.sil_frames_per_spk;

    std::vector<long long> flat(static_cast<std::size_t>(num_speakers_) * n_pad);
    std::iota(flat.begin(), flat.end(), 0);
    auto flat_score = [&](long long i) -> float {
        const int f = static_cast<int>(i % n_pad);
        if (f >= n) {
            return kPosInf;
        }
        const int s = static_cast<int>(i / n_pad);
        return scores[static_cast<std::size_t>(f) * num_speakers_ + s];
    };
    std::partial_sort(flat.begin(), flat.begin() + cap, flat.end(), [&](long long a, long long b) {
        const float sa = flat_score(a), sb = flat_score(b);
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });
    std::vector<long long> picked(flat.begin(), flat.begin() + cap);
    for (auto& i : picked) {
        if (flat_score(i) == kNegInf) {
            i = kMaxIndex * static_cast<long long>(n_pad) + kMaxIndex;
        }
    }
    std::sort(picked.begin(), picked.end());

    std::vector<float> new_cache(static_cast<std::size_t>(cap) * emb_dim_);
    std::vector<float> new_preds(static_cast<std::size_t>(cap) * num_speakers_, 0.0F);
    for (int j = 0; j < cap; j++) {
        const long long i = picked[static_cast<std::size_t>(j)];
        const int f = static_cast<int>(i % n_pad);
        const bool disabled =
            i >= static_cast<long long>(num_speakers_) * n_pad || f >= n;
        if (disabled) {
            std::memcpy(
                new_cache.data() + static_cast<std::size_t>(j) * emb_dim_, mean_sil_emb_.data(),
                static_cast<std::size_t>(emb_dim_) * 4);
        } else {
            std::memcpy(
                new_cache.data() + static_cast<std::size_t>(j) * emb_dim_,
                spkcache_.data() + static_cast<std::size_t>(f) * emb_dim_,
                static_cast<std::size_t>(emb_dim_) * 4);
            std::memcpy(
                new_preds.data() + static_cast<std::size_t>(j) * num_speakers_,
                cache_preds.data() + static_cast<std::size_t>(f) * num_speakers_,
                static_cast<std::size_t>(num_speakers_) * 4);
        }
    }
    spkcache_ = std::move(new_cache);
    spkcache_preds_ = std::move(new_preds);
    spk_frames_ = cap;
}

}  // namespace diar
