"""Analyze the tail-fixture outputs (production probdump vs NeMo fp32).

    python3 scripts/analyze_tail_result.py tail_slice.probs.f32 tail_slice.nemo.npz

Prints the row-aligned diff, with emphasis on the tail (the last chunk's
emitted rows — where the production binary lacks the MaskedConvSequential
feat_len semantics). Exit code is 0 regardless: a red result is data.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_probdump(path: Path) -> tuple[np.ndarray, int, int]:
    raw = path.read_bytes()
    n_frames, n_spk = struct.unpack("<qi", raw[:12])
    expect = 12 + n_frames * n_spk * 4
    assert len(raw) == expect, f"{path}: {len(raw)} != {expect}"
    vals = np.frombuffer(raw, "<f4", count=n_frames * n_spk, offset=12)
    return vals.reshape(n_frames, n_spk).astype(np.float64), n_frames, n_spk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probs", help="production probdump (12B header + f32)")
    ap.add_argument("npz", help="NeMo reference npz (m2_tail_fixture output)")
    ap.add_argument("--tail-rows", type=int, default=24)
    args = ap.parse_args()

    prod, nf, ns = load_probdump(Path(args.probs))
    z = np.load(args.npz, allow_pickle=True)
    tp = np.asarray(z["total_preds"], dtype=np.float64)
    print(f"production : {nf} rows x {ns} spk")
    print(f"nemo       : {tp.shape[0]} rows x {tp.shape[1]} spk")
    if "n_chunks" in z:
        print(f"nemo chunks: {np.asarray(z['n_chunks']).ravel().tolist()}")

    n = min(nf, tp.shape[0])
    d = np.abs(tp[:n] - prod[:n])
    row_max = d.max(axis=1)
    print(f"\nrows compared: {n}   overall max_abs={d.max():.6g} "
          f"mean_abs={d.mean():.6g}")
    for frac in (0.5, 0.9, 0.99):
        k = max(1, int(n * (1 - frac)))
        print(f"  last {frac:.0%} of rows: max_abs={row_max[-k:].max():.6g}")

    print(f"\nlast {args.tail_rows} rows (row: prod_max nemo_max diff):")
    start = n - args.tail_rows
    for i in range(max(0, start), n):
        pm = prod[i].max()
        nm = tp[i].max()
        flag = "  <-- TAIL" if i >= n - 11 else ""
        print(f"  {i:5d}: {pm:8.5f} {nm:8.5f} {row_max[i]:9.5f}{flag}")

    tail11 = row_max[max(0, n - 11):]
    summary = {
        "rows_prod": int(nf), "rows_nemo": int(tp.shape[0]),
        "overall_max_abs": float(d.max()),
        "overall_mean_abs": float(d.mean()),
        "tail11_max_abs": float(tail11.max()) if tail11.size else 0.0,
        "tail11_mean_abs": float(tail11.mean()) if tail11.size else 0.0,
        "body_max_abs": float(row_max[:max(0, n - 11)].max()) if n > 11 else None,
    }
    print("\nsummary: " + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
