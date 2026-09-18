// M2 tailfix: NeMo tail-window semantics (pad-to-32 + masked stem + trim).
//
// Upstream pins (python NeMo, tail-fixture verified 2026-09-18):
//   - streaming loader pads the final mel window with ZERO rows up to a
//     multiple of 32; feat_length = valid mel rows before padding
//     (tail_slice: 80 valid + 16 zero = 96).
//   - stem applies the multiplicative per-stage mask from feat_len
//     (subsampling_forward feat_len>0 branch); masked rows == out.bias fill.
//   - emission trims phantom rows: preds_full[lc+l2 : lc+l2+valid_T3].
//
// Production (current engine.cpp) does none of the three: whole-window stem
// (feat_len=-1), full-T3 emission — the tail11 probe shows the last row
// diverging 0.9964 from NeMo's zero phantom row.
//
// Contract (M2-TAILFIX-TASK.md §2):
//   - row count NEVER changes: emit exactly valid_T3 rows (NeMo's phantom
//     rows are gate-invisible zeros; production's phantom rows are wrong
//     speech values — trimming them is the fix, not a geometry change).
//   - active speech rows (8 of 12 in the probe) are untouched; the masked
//     tail value-rows (2 of 12) move toward NeMo; phantom rows (2) disappear.
//   - four legacy fixtures are all-silence tails: trimmed rows are zeros
//     either way, so their wire bytes are bit-identical pre/post patch.
#pragma once

namespace diar {

// NeMo streaming loader pads mel windows to this multiple (upstream pin).
inline constexpr int kNemoWindowPadMultiple = 32;

// Padded window length for a tail chunk: round t_mel up to a multiple of 32.
// Full windows (already a multiple) are returned unchanged.
inline int nemo_padded_window(int t_mel) {
    if (t_mel <= 0) return 0;
    return (t_mel + kNemoWindowPadMultiple - 1) / kNemoWindowPadMultiple *
           kNemoWindowPadMultiple;
}

// Valid encoder rows for a masked tail window: subsampled length of the
// VALID mel rows (not of the padded window). E.g. 80 -> 10, 86 -> 11.
int tail_valid_frames(int feat_len, int subsampling_factor);

}  // namespace diar
