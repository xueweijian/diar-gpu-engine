#pragma once

// M2 Stage 3.2a — Sortformer weight arena.
//
// Owns the parsed weight container (GGUF v2/v3 or our DFW1 fp32 dump, see
// diar/gguf.hpp) and binds every tensor the assembled forward (3.2b) consumes
// into the const-pointer views the per-layer reference kernels already take
// (include/diar/{subsampling,conformer,mha,conv,layers}.hpp). One allocation
// per container; all views point into it — no copies, single lifetime.
//
// Naming pins (upstream a5b6953):
//   * GGUF schema = conversion/diarization.py remap:
//       sortformer_modules.encoder_proj.* -> encoder_proj.*
//       sortformer_modules.*              -> head.*  (first_hidden_to_hidden,
//                                                   single_hidden_to_spks)
//       transformer_encoder.*             -> transformer.*
//       encoder.*                         -> unchanged
//     synthesized at conversion: encoder.pos_enc.pe (analytical rel-pos table,
//     F32) and preprocessor.fb (mel filterbank, squeezed [n_mels, n_fft/2+1]).
//   * Conformer conv module prefix is "conv", NOT torch's conv_module
//     (fastconformer.cpp:649 ConformerLayer ctor; _POINTWISE_PATTERN in the
//     converter matches encoder.layers.{i}.conv.pointwise_conv{1,2}).
//   * Conformer self-attn projections are stored SEPARATELY in the GGUF
//     (linear_q/k/v/pos/out); upstream rel_pos_attention.cpp:808 stacks them
//     into a fused linear_qkv at load. We bind the separate tensors directly
//     — RelPosMhaWeights takes separate pointers.
//   * Stem tensor indices conv.{0,2,3,5,6} (1 and 4 are ReLUs, no tensors),
//     pinned from the K3 kernel loader (encoder.pre_encode.conv.*).
//
// Layout/shape pins:
//   * GGUF ne[] is fastest-first; DFW1 dims[] is outermost-first. Both are
//     normalized to canonical row-major outermost-first with size-1 dims
//     stripped on BOTH expected and actual sides before comparison: the
//     converter squeezes conformer pointwise [C,C,1] -> [C,C] but keeps stem
//     pointwise [C,C,1,1] and depthwise [C,1,K]; the underlying bytes are
//     identical for all of these. Square transposes cannot be caught by
//     shape checks — K5/K6 numeric gates own that failure mode.
//   * Linear weights are [out,in] row-major (nn.hpp contract == torch
//     contiguous == GGUF row-major bytes).
//
// PE route pin: production loads encoder.pos_enc.pe from the GGUF
// (fastconformer.cpp:1159, rel_pos_attention.cpp:119) — required on the GGUF
// route so K6 slices the exact production table. DFW1 (K5-A truth anchor)
// omits it: the forward then builds the table with the pinned formula
// (relpos_table_forward; v12 fingerprint mse 1.08e-13 vs NeMo live window).
// A shape-valid pe in a DFW1 file is still bound (accepted, documented).
//
// Config pins:
//   * GGUF route parses sortformer.* KVs with upstream parse_config defaults
//     (sortformer_model.cpp:17-63; missing key -> default). Rejected loudly:
//     conv_norm != "batch_norm", transformer pre_ln, use_bias false — the
//     reference kernels implement exactly these variants.
//   * sortformer.preprocessor.* KVs are validated against the compiled-in
//     FrontendConfig pins (the FE is not configurable in this engine); any
//     mismatch aborts so a foreign GGUF cannot silently feed a wrong mel.
//   * DFW1 carries no KV metadata: pure defaults + shape-driven validation
//     (every expected shape is derived from the config, so a wrong default
//     fails the shape check rather than misbinding).
//   * sortformer.streaming.* geometry KVs are deliberately NOT parsed:
//     production presets override them unconditionally (diarizer.cpp
//     riva_streaming/riva_offline) and the engine takes explicit
//     StreamGeometry (3.2c).
//
// Coverage is bidirectional and fail-loud: every expected tensor must exist
// with the derived shape, and every tensor in the file must be expected. A
// typo on either side aborts with the offender named (WeightFileError).

#include "diar/conformer.hpp"
#include "diar/diar.hpp"  // AoscScoringConfig, FrontendConfig pins
#include "diar/gguf.hpp"
#include "diar/layers.hpp"
#include "diar/nn.hpp"
#include "diar/posenc.hpp"
#include "diar/subsampling.hpp"

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace diar {

struct SortformerConfig {
    // Defaults = upstream parse_config (sortformer_model.cpp:17-63) for the
    // v2 diar model. GGUF KVs override; DFW1 uses them verbatim.
    int d_model = 512;
    int encoder_layers = 17;
    int encoder_heads = 8;
    int encoder_d_ff = 2048;
    int conv_kernel = 9;
    int subsampling_factor = 8;          // 3 striding stages (log2)
    int subsampling_conv_channels = 256;
    int feat_in = 128;                   // mel bins
    bool xscaling = true;
    int pos_emb_max_len = 5000;

    int transformer_layers = 18;
    int transformer_hidden = 192;
    int transformer_inner = 768;
    int transformer_heads = 8;

    int num_speakers = 4;

    AoscScoringConfig scoring{};
};

// Move-only: the bound pointers point into the owned container. Moving is
// safe (heap buffers move with the vectors); copying is deleted because a
// copy would leave the source's views dangling after its destruction.
class SortformerWeights {
public:
    static SortformerWeights load(const std::string& path);

    SortformerWeights(SortformerWeights&&) noexcept = default;
    SortformerWeights& operator=(SortformerWeights&&) noexcept = default;
    SortformerWeights(const SortformerWeights&) = delete;
    SortformerWeights& operator=(const SortformerWeights&) = delete;

    const SortformerConfig& config() const { return cfg_; }

    // ---- bound views (consumed by 3.2b sortformer_run_chunk) ----
    const SubsamplingWeights& stem() const { return stem_; }
    const ConformerLayerWeights& conformer(int i) const;  // throws out_of_range
    const TransformerBlockWeights& transformer(int i) const;  // ditto

    // encoder_proj [transformer_hidden, d_model] + bias.
    const float* proj_w() const { return proj_w_; }
    const float* proj_b() const { return proj_b_; }

    // head.first_hidden_to_hidden [H,H]; head.single_hidden_to_spks [spks,H].
    const float* head_hidden_w() const { return head_w_; }
    const float* head_hidden_b() const { return head_b_; }
    const float* head_spks_w() const { return spks_w_; }
    const float* head_spks_b() const { return spks_b_; }

    // preprocessor.fb [n_mels, n_fft/2+1] row-major (mel basis).
    const float* mel_basis() const { return fb_; }

    // encoder.pos_enc.pe [2*pos_emb_max_len-1, d_model] row-major, or nullptr
    // when absent (DFW1 route -> forward uses relpos_table_forward). When
    // present the forward slices it exactly like upstream
    // (row i of window L = pe[center - L + i], center = rows/2 + 1).
    const float* pe_table() const { return pe_; }
    int pe_rows() const { return pe_rows_; }

    // Number of tensors bound (expected-table size hit).
    std::size_t bound_tensor_count() const { return bound_; }

    // Full expected name list for the given config (test oracle / coverage
    // accounting). Same table load() enforces.
    static std::vector<std::string> expected_tensor_names(const SortformerConfig& cfg,
        bool require_pe);

private:
    SortformerWeights() = default;

    // 3.2a binding internals — defined in sortformer.cpp.
    struct Slot;  // {name, canonical shape, target pointer}
    static std::vector<Slot> collect_slots(SortformerWeights* w, const SortformerConfig& cfg);

    SortformerConfig cfg_{};
    std::variant<GgufFile, F32WeightFile> file_;

    SubsamplingWeights stem_{};
    std::vector<ConformerLayerWeights> conf_;
    std::vector<TransformerBlockWeights> xf_;
    const float* proj_w_ = nullptr;
    const float* proj_b_ = nullptr;
    const float* head_w_ = nullptr;
    const float* head_b_ = nullptr;
    const float* spks_w_ = nullptr;
    const float* spks_b_ = nullptr;
    const float* fb_ = nullptr;
    const float* pe_ = nullptr;
    int pe_rows_ = 0;
    std::size_t bound_ = 0;
};

// Surviving frequency bins after log2(subsampling_factor) striding stages of
// the stem (subsampling.hpp geometry: L' = (L + 2 - 3)/2 + 1). 128 -> 16.
int subsampling_freq_bins(int feat_in, int subsampling_factor);

// ---------------------------------------------------------------------------
// Stage 3.2b — per-chunk forward assembly.
// ---------------------------------------------------------------------------
// Upstream pin (sortformer_model.cpp build_graph + build_graph_from_embeddings,
// a5b6953), in order:
//   mel -> pre_encode stem                      [T3, D]   (NO xscale here)
//   concat [spkcache | fifo | chunk_embs]       [L, D]    (AOSC caches store
//                                                          RAW pre-encode
//                                                          embeddings; the
//                                                          state prefix is
//                                                          re-scaled every
//                                                          chunk)
//   xscale by sqrt(D) on the WHOLE concat      (only if cfg.xscaling)
//   rel-pos table for L                        [2L-1, D] (stored GGUF table
//                                                          sliced like
//                                                          production, else
//                                                          formula)
//   conformer layers x N                        [L, D]
//   encoder_proj Linear D -> X                  [L, X]
//   transformer blocks x M (post-LN)            [L, X]
//   head relu->Linear->relu->Linear->sigmoid    [L, n_spk] pre-gate preds
// Scalar run (B=1): no attention/valid masks exist on this path — masks are
// a batching artifact only (upstream run_chunk takes none).
//
// Tail-semantics switch (Step 0 verdict, machine-enforced by tests):
//   feat_len > 0  : NeMo reference — masked stem; rows >= valid become
//                   out.bias fill rows (K5-A route)
//   feat_len <= 0 : production — full-window stem, no mask (K6 route)
// Frame GEOMETRY (T3, L) is identical in both modes; only tail-row VALUES
// differ, so the host state machine (3.2c) is shared.

struct TapSink {
    virtual ~TapSink() = default;
    // data is rows*cols row-major. Names emitted by sortformer_run_chunk:
    //   "stem.out" [T3,D] pre-xscale (== chunk_embs, bit-exact)
    //   "concat.raw" [L,D] (pre-xscale), "xscaled" [L,D] (post)
    //   "pos_emb" [2L-1,D], "conformer.{i}" [L,D], "proj.out" [L,X],
    //   "transformer.{i}" [L,X], "preds" [L,n_spk]
    virtual void tap(const char* name, const float* data, int rows, int cols) = 0;
};

struct SortformerChunkOutput {
    std::vector<float> preds;       // [L, n_spk] frame-major, pre-gate sigmoid
    std::vector<float> chunk_embs;  // [T3, D] raw pre-encode (no xscale — AOSC
                                    // cache contract; bit-equal to "stem.out")
    int total_frames = 0;           // L = spkcache + fifo + T3
    int chunk_frames = 0;           // T3 (full-window geometry both modes)
};

// Full-window encoder frame count for a mel window (3-stage formula
// (L-1)/2+1 applied log2(subampling_factor) times; 160 -> 20, 86 -> 11).
int sortformer_subsampled_len(int t_mel, int subsampling_factor);

// Throws std::invalid_argument on L exceeding the PE budget.
SortformerChunkOutput sortformer_run_chunk(const float* mel, int t_mel, int feat_len,
    const float* spkcache, int spkcache_frames, const float* fifo, int fifo_frames,
    const SortformerWeights& w, TapSink* taps = nullptr);

}  // namespace diar
