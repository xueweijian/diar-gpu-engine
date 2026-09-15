# Parity fixtures (M1/M2 ground truth)

A fixture pins **one observation of the upstream binary** as ground truth
for the M2 engine to reproduce. Schema v1 covers the `frame_probs` layer
(per-frame speaker probabilities, the hysteresis input — the layer the v9
probdump made observable).

## Truth discipline

- Fixtures are generated **only** from kernel output by
  `scripts/fill_parity_fixture.py` (added with the v9 verdict). The
  generator records kernel version, repo SRC commit, upstream commit,
  capture time and GPU in `manifest.json`.
- Payload files are byte-copies of the kernel probdump (`<qi>` header +
  f32 row-major), sha256-pinned in the manifest. Any post-capture edit
  — even a recompression — fails `load_case()`.
- Nothing in this package generates fixtures from engine output. A gate
  comparing an engine against truth derived from that same engine is
  self-certification; the orals (AOSC, BirthGate, FE) exist precisely
  because that pattern ships bugs.
- `nbytes` in the manifest counts the logical tensor (`shape` product × 4);
  `frame_probs` files carry the extra 12-byte header on disk and the loader
  validates the file size as `nbytes + 12`.

## Gates

- `exact`: float-level equality. For host layers compiled from the same
  source on both sides (hysteresis, segmentation).
- `tolerance`: the NN core cannot be expected to bit-match ggml; thresholds
  (`FRAME_PROBS_TOLERANCE`) start conservative (`frame_agreement >= 0.999`,
  `max_abs <= 0.05`, `mean_abs <= 0.005`) and tighten as M2 lands. They are
  pinned constants, not env-tunable, so a red gate always means the same
  thing in CI.

## Layout

```
parity/fixtures/<name>/manifest.json   # schema v1 (see parity/manifest.py)
parity/fixtures/<name>/probs.f32       # kernel probdump bytes, hash-pinned
```

Fixture directories are created by the fill script only; the directory is
absent until the v9 verdict lands.
