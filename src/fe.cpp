// Faithful port of the upstream NeMo-compatible log-mel frontend, CPU path.
// Upstream: NeMo-Speech.cpp a5b6953 src/asr/features/fe.cpp — every
// arithmetic line below is kept line-identical (naming adapted to this
// repo's FrontendConfig fields; class/method names kept upstream for 1:1
// review). Bit-parity with upstream is enforced by tests/fe_oracle.cpp,
// which compiles a mechanically reskinned copy of the upstream code next to
// this file and diffs outputs byte-for-byte.
// What was NOT ported (all unreachable on the diar path, which constructs
// the FE with a null backend manager): the ggml GPU graph (FeModule), the
// GPU micro-batcher (GpuBatcher, compute_gpu_*), the NVTX timing block and
// read_wav_mono_16k (I/O, not DSP).
// Degenerate-domain note: with reflect_left=false and n_samples < n_fft/2
// the upstream frame count computes a size_t underflow before the int cast;
// the same expression is kept verbatim so behavior matches upstream even
// there, but callers stay in-domain (streaming always feeds >= hop samples).

#include "diar/diar.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace diar {
namespace {

constexpr float kPi = 3.14159265358979323846F;

float hz_to_mel(float f)  // slaney (HTK-like used by NeMo default)
{
    return 1127.0F * std::log(1.0F + f / 700.0F);
}

float mel_to_hz(float m) {
    return 700.0F * (std::exp(m / 1127.0F) - 1.0F);
}

// Iterative radix-2 FFT over real and imaginary arrays of equal length.
void fft_radix2_inplace(std::vector<float>& re, std::vector<float>& im) {
    const int n = static_cast<int>(re.size());
    if (n <= 1)
        return;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0F * kPi / len;
        const float wre = std::cos(ang);
        const float wim = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0F, cur_im = 0.0F;
            for (int k = 0; k < len / 2; k++) {
                const float ur = re[i + k];
                const float ui = im[i + k];
                const float vr = re[i + k + len / 2] * cur_re - im[i + k + len / 2] * cur_im;
                const float vi = re[i + k + len / 2] * cur_im + im[i + k + len / 2] * cur_re;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                const float nre = cur_re * wre - cur_im * wim;
                const float nim = cur_re * wim + cur_im * wre;
                cur_re = nre;
                cur_im = nim;
            }
        }
    }
}

} // namespace

MelSpectrogramExtractor::MelSpectrogramExtractor(const FrontendConfig& cfg)
    : config_(cfg) {
    if (config_.fmax <= 0.0F)
        config_.fmax = 0.5F * config_.sample_rate;
    int v = config_.n_fft;
    while (v > 1) {
        if (v & 1)
            throw std::runtime_error("n_fft must be power of 2 for this FE");
        v >>= 1;
    }
    init_window();
    init_mel_basis();
}

void
MelSpectrogramExtractor::init_window() {
    const int win = win_length();
    window_.resize(win);
    // Legacy ASR path: periodic Hann. NeMo-parity path (hann_periodic=false):
    // symmetric Hann, matching torch.hann_window(periodic=False) which is
    // what FilterbankFeatures registers.
    const float denom = config_.hann_periodic ? static_cast<float>(win)
                                              : static_cast<float>(win - 1);
    for (int i = 0; i < win; i++) {
        window_[i] = 0.5F * (1.0F - std::cos(2.0F * kPi * i / denom));
    }
}

void
MelSpectrogramExtractor::init_mel_basis() {
    // Fallback Slaney-style filterbank for models without a serialized basis.
    // The diar wiring overrides this via set_mel_basis with the GGUF
    // "preprocessor.fb" tensor, but the fallback math is ported verbatim and
    // oracle-covered so non-diar consumers match upstream too.
    const int n_bins = config_.n_fft / 2 + 1;
    const int n_mels = config_.n_mels;
    mel_basis_.assign(static_cast<std::size_t>(n_mels) * n_bins, 0.0F);

    const float mel_min = hz_to_mel(config_.fmin);
    const float mel_max = hz_to_mel(config_.fmax);
    std::vector<float> mel_points(static_cast<std::size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        const float m = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
        mel_points[static_cast<std::size_t>(i)] = mel_to_hz(m);
    }
    std::vector<float> bin_freqs(static_cast<std::size_t>(n_bins));
    for (int i = 0; i < n_bins; i++) {
        bin_freqs[static_cast<std::size_t>(i)] =
            static_cast<float>(i) * config_.sample_rate / config_.n_fft;
    }
    for (int m = 0; m < n_mels; m++) {
        const float f_left = mel_points[static_cast<std::size_t>(m)];
        const float f_center = mel_points[static_cast<std::size_t>(m) + 1];
        const float f_right = mel_points[static_cast<std::size_t>(m) + 2];
        for (int k = 0; k < n_bins; k++) {
            const float f = bin_freqs[static_cast<std::size_t>(k)];
            float w = 0.0F;
            if (f >= f_left && f <= f_center)
                w = (f - f_left) / (f_center - f_left + 1e-12F);
            else if (f >= f_center && f <= f_right)
                w = (f_right - f) / (f_right - f_center + 1e-12F);
            mel_basis_[static_cast<std::size_t>(m) * n_bins + k] = std::max(0.0F, w);
        }
    }
}

void
MelSpectrogramExtractor::set_mel_basis(const float* fb, int n_mels, int n_bins) {
    const int expected_bins = config_.n_fft / 2 + 1;
    if (n_mels != config_.n_mels || n_bins != expected_bins) {
        throw std::runtime_error(
            "MelSpectrogramExtractor::set_mel_basis: shape mismatch - got (" +
            std::to_string(n_mels) + ", " + std::to_string(n_bins) + "), expected (" +
            std::to_string(config_.n_mels) + ", " + std::to_string(expected_bins) + ")");
    }
    mel_basis_.assign(fb, fb + static_cast<std::size_t>(n_mels) *
                                        static_cast<std::size_t>(n_bins));
}

void
MelSpectrogramExtractor::compute(const float* audio, std::size_t n_samples,
    std::vector<float>& features, int& n_frames, bool reflect_left,
    bool normalize) const {
    compute_padded(audio, n_samples, n_samples, features, n_frames, reflect_left,
        normalize);
}

void
MelSpectrogramExtractor::compute_padded(const float* audio, std::size_t n_samples,
    std::size_t valid_samples, std::vector<float>& features, int& n_frames,
    bool reflect_left, bool normalize) const {
    if (valid_samples > n_samples)
        throw std::invalid_argument("valid audio length exceeds padded length");
    // `normalize` is a call-site request (streaming can explicitly suppress
    // utterance normalization); the config is authoritative about whether
    // normalization exists at all.
    normalize = normalize && config_.normalize_per_feature;

    const int hop = hop_length();
    const int win = win_length();
    const int n_fft_ = config_.n_fft;

    // Pre-emphasis y[t] = x[t] - α·x[t-1] on the RAW signal, before STFT
    // framing. Matches NeMo's order (`preemph(audio)` then `torch.stft`);
    // applying it after padding biases the reflected/zero edges. First sample
    // kept as `y[0] = x[0]`.
    std::vector<float> pre(n_samples, 0.0F);
    std::copy_n(audio, valid_samples, pre.data());
    if (config_.preemph != 0.0F && valid_samples >= 2) {
        for (std::size_t i = valid_samples - 1; i > 0; --i) {
            pre[i] = pre[i] - config_.preemph * pre[i - 1];
        }
    }
    const float* preemphed = pre.data();

    // Zero-pad by n_fft/2 on each side to match NeMo's
    // `torch.stft(center=True, pad_mode='constant')`.
    //
    // `reflect_left=false` is the mid-stream streaming case where the caller
    // has already prepended n_fft/2 real left-context samples - we still emit
    // the right-side zero pad so frame indexing is consistent; the streaming
    // runner only consumes frames whose right edge is real, so the synthetic
    // right samples are never actually read.
    std::vector<float> padded;
    padded.reserve(n_samples + static_cast<std::size_t>(n_fft_));
    if (reflect_left) {
        for (int i = 0; i < n_fft_ / 2; i++) padded.push_back(0.0F);
    }
    for (std::size_t i = 0; i < n_samples; i++) padded.push_back(preemphed[i]);
    for (int i = 0; i < n_fft_ / 2; i++) padded.push_back(0.0F);

    n_frames = static_cast<int>(
        (padded.size() - static_cast<std::size_t>(n_fft_)) / static_cast<std::size_t>(hop) + 1);
    if (n_frames <= 0) {
        features.clear();
        n_frames = 0;
        return;
    }

    const int n_bins = n_fft_ / 2 + 1;
    features.assign(static_cast<std::size_t>(config_.n_mels) *
                        static_cast<std::size_t>(n_frames),
        0.0F);

    std::vector<float> re(static_cast<std::size_t>(n_fft_), 0.0F);
    std::vector<float> im(static_cast<std::size_t>(n_fft_), 0.0F);

    for (int f = 0; f < n_frames; f++) {
        const int offset = f * hop;
        std::fill(re.begin(), re.end(), 0.0F);
        std::fill(im.begin(), im.end(), 0.0F);
        // Window placement: legacy ASR path right-aligns (woff=0); the
        // NeMo-parity path centers the window inside the n_fft frame the way
        // torch.stft pads it on both sides (stft_center_window=true).
        const int woff = config_.center_window ? (n_fft_ - win) / 2 : 0;
        for (int i = 0; i < win; i++) {
            float sample = padded[static_cast<std::size_t>(offset + woff + i)];
            re[static_cast<std::size_t>(woff + i)] = sample * window_[static_cast<std::size_t>(i)];
        }

        fft_radix2_inplace(re, im);

        // Power spectrum -> mel
        for (int m = 0; m < config_.n_mels; m++) {
            float acc = 0.0F;
            for (int k = 0; k < n_bins; k++) {
                const float power =
                    re[static_cast<std::size_t>(k)] * re[static_cast<std::size_t>(k)] +
                    im[static_cast<std::size_t>(k)] * im[static_cast<std::size_t>(k)];
                acc += mel_basis_[static_cast<std::size_t>(m) * n_bins + k] * power;
            }
            features[static_cast<std::size_t>(m) +
                     static_cast<std::size_t>(f) * config_.n_mels] =
                std::log(acc + config_.log_zero_guard);
        }
    }

    const int valid = std::min(static_cast<int>(valid_samples) / hop, n_frames);
    if (config_.mask_invalid_frames) {
        for (int f = valid; f < n_frames; ++f) {
            std::fill_n(features.data() + static_cast<std::size_t>(f) * config_.n_mels,
                config_.n_mels, 0.0F);
        }
    }

    // Per-feature normalization over the valid frames in this call. Streaming
    // mode skips it and maintains incremental statistics in the runner.
    if (!normalize)
        return;

    // NeMo normalizes only the valid `floor(samples / hop)` frames, uses an
    // unbiased variance, and masks the final centered/padded frame. This is
    // observably different from normalizing every STFT frame.
    for (int m = 0; m < config_.n_mels; m++) {
        double sum = 0.0;
        for (int f = 0; f < valid; f++)
            sum += features[static_cast<std::size_t>(m) +
                            static_cast<std::size_t>(f) * config_.n_mels];
        const double mean = valid > 0 ? sum / valid : 0.0;
        double var = 0.0;
        for (int f = 0; f < valid; f++) {
            const double d =
                features[static_cast<std::size_t>(m) +
                         static_cast<std::size_t>(f) * config_.n_mels] -
                mean;
            var += d * d;
        }
        if (valid > 1)
            var /= valid - 1;
        const double inv_std = 1.0 / (std::sqrt(var) + 1e-5);
        for (int f = 0; f < n_frames; f++) {
            const std::size_t idx = static_cast<std::size_t>(m) +
                                    static_cast<std::size_t>(f) * config_.n_mels;
            features[idx] = f < valid ? static_cast<float>((features[idx] - mean) * inv_std)
                                      : 0.0F;
        }
    }
}

std::vector<float>
offline_peak_normalize(const float* audio, std::size_t n_samples, float eps) {
    // NeMo's offline path (streaming_mode=False, process_signal) peak-
    // normalizes the waveform before FE: x * 1/(max(x) + eps), eps=1e-3.
    // The streaming path does NOT - this is offline-only.
    float peak = n_samples ? audio[0] : 0.0F;
    for (std::size_t i = 1; i < n_samples; i++) peak = std::max(peak, audio[i]);
    const float gain = 1.0F / (peak + eps);
    std::vector<float> scaled(n_samples);
    for (std::size_t i = 0; i < n_samples; i++) scaled[i] = audio[i] * gain;
    return scaled;
}

} // namespace diar
