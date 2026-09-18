// M2 Stage 3.2c — engine implementation. Upstream pins per section.
#include "diar/engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace diar {

DiarEngine::DiarEngine(SortformerWeights weights, EngineConfig cfg)
    : weights_(std::move(weights)), cfg_(cfg),
      fe_(FrontendConfig{}), aosc_(cfg_.geometry, weights_.config().scoring,
          weights_.config().num_speakers, weights_.config().d_model),
      gate_(weights_.config().num_speakers) {
    const SortformerConfig& c = weights_.config();
    n_spk_ = c.num_speakers;
    sub_ = c.subsampling_factor;
    n_mels_ = fe_.n_mels();
    if (c.feat_in != n_mels_)
        throw std::invalid_argument(
            "DiarEngine: weights feat_in=" + std::to_string(c.feat_in) +
            " but the pinned FE produces " + std::to_string(n_mels_) +
            " mels (foreign model would silently desync the stem)");
    fe_.set_mel_basis(weights_.mel_basis(), n_mels_, fe_.n_fft() / 2 + 1);
    validate_stream_geometry(cfg_.geometry, n_spk_, c.scoring.sil_frames_per_spk,
        c.pos_emb_max_len);
}

void DiarEngine::ensure_mel() {
    std::vector<float> new_mel;
    const int n = produce_new_mel_frames(fe_, audio_buf_, audio_base_, mel_produced(), new_mel);
    if (n > 0)
        mel_buf_.insert(mel_buf_.end(), new_mel.begin(), new_mel.end());
}

void DiarEngine::feed_audio(const float* samples, std::size_t n_samples) {
    // Trailing push after finish() is a silent no-op (upstream contract).
    if (finished_)
        return;
    audio_buf_.insert(audio_buf_.end(), samples, samples + n_samples);
    ensure_mel();
    run_ready_chunks(/*end_of_stream=*/false);
}

void DiarEngine::finish() {
    if (finished_)
        return;
    finished_ = true;
    mel_real_ = mel_produced();
    // Geometric decay tail: y[k] = a^(k+1)*x[N-1] - a*a^k*x[N-1] = 0 — the
    // post-preemphasis pad is exactly zero, matching NeMo's constant right
    // pad bit-for-bit (upstream DiarStream::finish pin).
    const int half = fe_.n_fft() / 2;
    if (!audio_buf_.empty() || audio_base_ > 0) {
        const float a = fe_.config().preemph;
        float tail = audio_buf_.empty() ? 0.0F : audio_buf_.back();
        for (int k = 0; k < half; k++) {
            tail *= a;
            audio_buf_.push_back(tail);
        }
        ensure_mel();
    }
    run_ready_chunks(/*end_of_stream=*/true);
}

std::vector<Segment> DiarEngine::segments() const {
    return upstream_segments_from_probs(post_gate_frame_probabilities(), cfg_.segmentation);
}

FrameProbabilities DiarEngine::post_gate_frame_probabilities() const {
    FrameProbabilities fp(static_cast<std::size_t>(n_frames()),
        static_cast<std::size_t>(n_spk_));
    fp.values() = probs_;
    return fp;
}

// Chunk scheduler — the streaming counterpart of NeMo's streaming_feat_loader
// (upstream DiarStream::run_one_chunk pin): the next chunk covers mel
// [mel_consumed_, mel_consumed_ + hop) plus lc/rc context, clamped at the
// stream edges (riva feeds exact-length tails — no pad+mask). Forced chunks
// may be shorter than a hop but stay on the 80 ms encoder-frame grid so
// frames are labeled exactly once.
bool DiarEngine::run_one_chunk(bool force, bool final_flush) {
    const int hop_mel = cfg_.geometry.chunk_len * sub_;
    const int lc_mel_max = cfg_.geometry.chunk_left_context * sub_;
    const int rc_mel_max = cfg_.geometry.chunk_right_context * sub_;

    const std::int64_t stt = mel_consumed_;
    std::int64_t end = stt + hop_mel;
    if (!force) {
        if (end + rc_mel_max > mel_produced())
            return false;
    } else {
        end = std::min(end, mel_produced());
        if (!final_flush)
            end = stt + ((end - stt) / sub_) * sub_;  // whole encoder frames only
        if (end <= stt)
            return false;
    }
    const std::int64_t lc_mel = std::min<std::int64_t>(lc_mel_max, stt);
    const std::int64_t rc_mel = std::min<std::int64_t>(rc_mel_max, mel_produced() - end);
    const std::int64_t w0 = stt - lc_mel;
    const int t_mel = static_cast<int>(end + rc_mel - w0);
    if (t_mel <= 0)
        return false;

    if (w0 < mel_base_)
        throw std::runtime_error("DiarEngine: mel window trimmed too aggressively");
    const float* mel = mel_buf_.data() + (w0 - mel_base_) * n_mels_;

    // Tail-semantics routing (Step 0): K5-A masks with the real-audio row
    // count inside the window; production runs the full window unmasked.
    int feat_len = -1;
    if (cfg_.nemo_tail_semantics) {
        const std::int64_t real = std::min<std::int64_t>(mel_real_, end + rc_mel) - w0;
        if (real >= t_mel)
            feat_len = -1;  // mask is a no-op — keep the production bit-path
        else
            feat_len = static_cast<int>(std::max<std::int64_t>(real, 0));
    }

    SortformerChunkOutput out = sortformer_run_chunk(mel, t_mel, feat_len,
        aosc_.spkcache_frames() ? aosc_.spkcache().data() : nullptr, aosc_.spkcache_frames(),
        aosc_.fifo_frames() ? aosc_.fifo().data() : nullptr, aosc_.fifo_frames(), weights_,
        tap_sink_);

    const int lc_enc = static_cast<int>(std::lround(lc_mel / static_cast<double>(sub_)));
    const int rc_enc = static_cast<int>(std::ceil(rc_mel / static_cast<double>(sub_)));
    ChunkLedgerEntry entry;
    entry.chunk_index = static_cast<int>(ledger_.size());
    entry.t_mel = t_mel;
    entry.t3 = out.chunk_frames;
    entry.lc_enc = lc_enc;
    entry.rc_enc = rc_enc;
    entry.spkcache_frames = aosc_.spkcache_frames();
    entry.fifo_frames = aosc_.fifo_frames();
    entry.window_frames = out.total_frames;
    entry.feat_len = feat_len;
    std::vector<float> emitted =
        aosc_.update(out.chunk_embs.data(), out.chunk_frames, out.preds.data(), lc_enc, rc_enc);
    entry.emitted = static_cast<int>(emitted.size()) / n_spk_;
    ledger_.push_back(entry);
    // BirthGate consumes a relabeled COPY; the raw emitted chain is the K5
    // pre-gate face.
    pre_gate_.insert(pre_gate_.end(), emitted.begin(), emitted.end());
    gate_.append(emitted, probs_);
    mel_consumed_ = end;

    // Trim consumed buffers; the next window starts at mel_consumed_ -
    // lc_mel_max, its first FE sample n_fft/2 before that frame's hop slot.
    const std::int64_t keep_mel = std::max<std::int64_t>(mel_consumed_ - lc_mel_max, 0);
    if (keep_mel > mel_base_) {
        mel_buf_.erase(mel_buf_.begin(), mel_buf_.begin() + (keep_mel - mel_base_) * n_mels_);
        mel_base_ = keep_mel;
    }
    const std::int64_t keep_sample =
        std::max<std::int64_t>(keep_mel * fe_.hop_length() - fe_.n_fft() / 2, 0);
    if (keep_sample > static_cast<std::int64_t>(audio_base_)) {
        audio_buf_.erase(audio_buf_.begin(), audio_buf_.begin() + (keep_sample - audio_base_));
        audio_base_ = static_cast<std::size_t>(keep_sample);
    }
    return true;
}

void DiarEngine::run_ready_chunks(bool end_of_stream) {
    while (run_one_chunk(/*force=*/end_of_stream, /*final_flush=*/end_of_stream)) {
    }
}

OfflineDiarizationResult diarize_offline(const SortformerWeights& weights, const float* audio,
    std::size_t n_samples) {
    // NeMo offline (streaming_mode=False, process_signal): peak-normalize,
    // whole-file mel, one forward with empty state (upstream pin).
    MelSpectrogramExtractor fe(FrontendConfig{});
    const SortformerConfig& c = weights.config();
    if (c.feat_in != fe.n_mels())
        throw std::invalid_argument("diarize_offline: feat_in != FE n_mels");
    fe.set_mel_basis(weights.mel_basis(), fe.n_mels(), fe.n_fft() / 2 + 1);

    float peak = n_samples ? audio[0] : 0.0F;
    for (std::size_t i = 1; i < n_samples; i++) peak = std::max(peak, audio[i]);
    const float gain = 1.0F / (peak + 1e-3F);
    std::vector<float> scaled(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) scaled[i] = audio[i] * gain;

    std::vector<float> mel;
    int t_mel = 0;
    fe.compute(n_samples ? scaled.data() : nullptr, n_samples, mel, t_mel,
        /*reflect_left=*/true, /*normalize=*/false);

    OfflineDiarizationResult r;
    r.n_mel = t_mel;
    r.n_frames = sortformer_subsampled_len(t_mel, c.subsampling_factor);
    if (r.n_frames > c.pos_emb_max_len)
        throw std::invalid_argument("diarize_offline: " + std::to_string(r.n_frames) +
                                    " encoder frames exceeds pos_emb_max_len=" +
                                    std::to_string(c.pos_emb_max_len));
    SortformerChunkOutput out = sortformer_run_chunk(
        mel.data(), t_mel, -1, nullptr, 0, nullptr, 0, weights);
    r.preds = std::move(out.preds);
    return r;
}

}  // namespace diar
