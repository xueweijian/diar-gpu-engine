"""Parity gates: compare an M2 engine layer output against a fixture.

Two verdict modes, mirroring how the two engine families can be held to
truth:

- ``exact``: byte equality (host-side re-implementations compiled from the
  same source, e.g. hysteresis/segmentation consuming identical probs).
- ``tolerance``: float distance metrics with per-layer thresholds. The NN
  core cannot be expected to bit-match ggml; the thresholds are calibrated
  from the upstream binary's own run-to-run spread (v9 verdict) so the gate
  fires on deviations beyond upstream's nondeterminism band, not on noise.

A gate returns a verdict dict; callers (pytest / kernel reports) decide
severity. Gates never mutate fixtures and never read files — loaders do.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from parity.loader import FrameProbsTensor

#: Tolerance profile for the frame_probs layer. ``frame_agreement_min`` is
#: the argmax-frame agreement (what hysteresis consumes); ``max_abs_max`` is
#: the largest tolerated per-element abs diff. Values start conservative and
#: are tightened as M2 lands; they are pinned here (not env-tunable) so a
#: CI failure always means the same thing.
FRAME_PROBS_TOLERANCE: dict[str, float] = {
    "frame_agreement_min": 0.999,   # argmax frames identical fraction
    "max_abs_max": 0.05,            # probabilities are [0,1]-ish
    "mean_abs_max": 0.005,
}


def frame_probs_gate(
    actual: Sequence[float],
    fixture: FrameProbsTensor,
    mode: str = "tolerance",
    tolerance: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare a row-major candidate against a pinned frame_probs tensor.

    mode="exact" demands identical bytes at the float level; mode=
    "tolerance" (default) applies FRAME_PROBS_TOLERANCE (or an explicit
    override dict). Returns {"verdict": "PASS"|"FAIL", ...metrics}.
    """
    if mode not in ("exact", "tolerance"):
        raise ValueError(f"unknown gate mode {mode!r}")
    if len(actual) != len(fixture.values):
        return {
            "verdict": "FAIL",
            "reason": f"size mismatch: {len(actual)} vs {len(fixture.values)}",
            "expected_frames": fixture.n_frames,
            "expected_spk": fixture.n_spk,
        }

    if mode == "exact":
        exact = all(a == b for a, b in zip(actual, fixture.values))
        return {"verdict": "PASS" if exact else "FAIL", "bit_identical": exact}

    tol = dict(FRAME_PROBS_TOLERANCE if tolerance is None else tolerance)
    max_abs = 0.0
    acc = 0.0
    agree = 0
    for f in range(fixture.n_frames):
        base = f * fixture.n_spk
        a_row = actual[base:base + fixture.n_spk]
        g_row = fixture.values[base:base + fixture.n_spk]
        if _argmax(a_row) == _argmax(g_row):
            agree += 1
        for a, g in zip(a_row, g_row):
            d = abs(a - g)
            if d > max_abs:
                max_abs = d
            acc += d
    n = len(fixture.values)
    metrics = {
        "frame_agreement": agree / fixture.n_frames,
        "max_abs": max_abs,
        "mean_abs": acc / n if n else 0.0,
        "bit_identical": max_abs == 0.0,
    }
    failed: list[str] = []
    if metrics["frame_agreement"] < tol["frame_agreement_min"]:
        failed.append(
            f"frame_agreement={metrics['frame_agreement']:.6g} < "
            f"{tol['frame_agreement_min']}")
    if metrics["max_abs"] > tol["max_abs_max"]:
        failed.append(
            f"max_abs={metrics['max_abs']:.6g} > {tol['max_abs_max']}")
    if metrics["mean_abs"] > tol["mean_abs_max"]:
        failed.append(
            f"mean_abs={metrics['mean_abs']:.6g} > {tol['mean_abs_max']}")

    return {"verdict": "FAIL" if failed else "PASS", "failures": failed, **metrics}


def _argmax(row: Sequence[float]) -> int:
    best = 0
    best_v = -math.inf
    for i, v in enumerate(row):
        if v > best_v:
            best_v = v
            best = i
    return best
