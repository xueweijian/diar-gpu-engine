#!/usr/bin/env python3
"""G-E cross-card diff (M3 plan §5, acceptance protocol §6).

Compares two or more diar-bench probs dumps (probdump wire format: 12-byte
little-endian <qi> header + f32 payload) produced on DIFFERENT cards with
the SAME route (fp32) and input. Gate: max_abs <= 1e-6 (fp32 route) — three
independent driver/cuBLAS stacks agreeing at that tier makes a silent
kernel bug vanishingly unlikely (plan wording).

Usage:
  ge_cross_diff.py DUMP_A DUMP_B [DUMP_C ...] [--gate 1e-6]

Exit 0 = green (all pairs within gate), 1 = red, 2 = usage/error.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<qi")


def load(path: Path) -> tuple[np.ndarray, dict]:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: shorter than the 12-byte header")
    n, spk = HEADER.unpack(raw[: HEADER.size])
    if n <= 0 or spk <= 0:
        raise ValueError(f"{path}: bad header ({n}, {spk})")
    expect = HEADER.size + n * spk * 4
    if len(raw) != expect:
        raise ValueError(f"{path}: {len(raw)} bytes != {expect} (shape {(n, spk)})")
    vals = np.frombuffer(raw, dtype="<f4", count=n * spk, offset=HEADER.size)
    return vals.reshape(n, spk), {"rows": int(n), "spk": int(spk)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--gate", type=float, default=1e-6)
    args = ap.parse_args()
    if len(args.dumps) < 2:
        print("need at least two dumps", file=sys.stderr)
        return 2

    loaded = []
    for p in args.dumps:
        try:
            arr, meta = load(p)
        except Exception as e:  # noqa: BLE001 — report and exit red
            print(f"{p}: {e}", file=sys.stderr)
            return 2
        loaded.append((p, arr, meta))
        print(f"{p.name}: {meta['rows']} rows x {meta['spk']} spk")

    rows0 = loaded[0][2]["rows"]
    if any(m["rows"] != rows0 for _, _, m in loaded):
        print("row count mismatch across dumps — cross-card geometry drift",
              file=sys.stderr)
        return 2

    report = {"gate": args.gate, "pairs": []}
    worst = 0.0
    green = True
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            a, b = loaded[i][1], loaded[j][1]
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            mx = float(d.max())
            worst = max(worst, mx)
            ok = mx <= args.gate
            green &= ok
            pair = {
                "a": loaded[i][0].name,
                "b": loaded[j][0].name,
                "max_abs": mx,
                "mean_abs": float(d.mean()),
                "bit_identical": bool(np.array_equal(a, b)),
                "ok": ok,
            }
            report["pairs"].append(pair)
            print(f"  {pair['a']} vs {pair['b']}: max_abs={mx:.3g} "
                  f"mean={pair['mean_abs']:.3g} "
                  f"{'GREEN' if ok else 'RED'}")
    report["worst_max_abs"] = worst
    report["verdict"] = "ge-green" if green else "ge-red"
    out = Path("ge_cross_diff.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"{report['verdict']} (worst {worst:.3g} vs gate {args.gate:g}) "
          f"-> {out}")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
