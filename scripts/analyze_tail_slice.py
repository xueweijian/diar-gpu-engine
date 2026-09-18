"""Reproduce / re-derive the tail-slice choice from an m2-ref npz.

The kernel's SLICE_START_SEC/LENGTH_SEC were chosen with this analysis
(mid audio = video2_audio.wav; the npz holds the same 16 kHz mono audio):

    python3 scripts/analyze_tail_slice.py \
        /var/minis/shared/diar-gpu-engine/m2-ref/m2_ref_mid.npz

It reports, for the pinned slice and a few alternates, the tail loudness and
(optionally, via the local tiny engine) the final chunk geometry. The
geometry check needs a built k5_runner + tiny DFW1 (see
tests/test_m2_stage3_k5a.py for the builder).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CANDIDATES = [("a", 244.5), ("b", 328.9), ("c", 343.4)]
PINNED = ("c", 343.4, 60.0)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean())) if x.size else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="m2_ref_{short,mid}.npz (has 'audio')")
    ap.add_argument("--runner", default=str(REPO / ".local-build" / "k5_runner"))
    ap.add_argument("--tiny-weights", default="/tmp/k5_selftest.dfw1")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    audio = np.asarray(z["audio"], dtype=np.float32)
    print(f"audio: {len(audio)} samples = {len(audio)/16000:.2f}s")

    runner = Path(args.runner)
    weights = Path(args.tiny_weights)
    have_engine = runner.exists() and weights.exists()
    if not have_engine:
        print("(tiny engine not available — geometry column skipped)")

    sr = 16000
    for tag, T in CANDIDATES:
        seg = audio[int((T - 60.0) * sr):int(T * sr)]
        last150 = rms(seg[-int(0.15 * sr):])
        last880 = rms(seg[-int(0.88 * sr):])
        line = (f"[{tag}] T={T:6.1f}s  last0.15s rms={last150:.3f}  "
                f"last0.88s rms={last880:.3f}")
        if have_engine:
            out = Path("/tmp/tail-analysis"); out.mkdir(exist_ok=True)
            f32 = out / f"{tag}.f32"
            f32.write_bytes(np.ascontiguousarray(seg, "<f4").tobytes())
            p = subprocess.run([str(runner), "--run", "--weights", str(weights),
                                "--audio", str(f32), "--out", str(out / tag)],
                               capture_output=True, text=True, timeout=900)
            if p.returncode == 0:
                led = json.loads((out / f"{tag}.ledger.json").read_text())
                last = led[-1]
                line += (f"  chunks={len(led)} t_mel={last['t_mel']} "
                         f"t_mel%32={last['t_mel'] % 32} t3={last['t3']}")
        print(line)

    tag, T, L = PINNED
    seg = audio[int((T - L) * sr):int(T * sr)]
    print(f"\npinned in kaggle/m2_tailfix: start={T - L}s length={L}s "
          f"(tag {tag}) — last1.5s rms={rms(seg[-int(1.5*sr):]):.3f}, "
          f"last0.15s rms={rms(seg[-int(0.15*sr):]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
