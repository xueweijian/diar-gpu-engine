"""M2 Stage 2 combined gate runner (single kernel, three sub-gates).

Kaggle script kernels upload ONLY code_file, so K1/K2/K3 are concatenated
into this file by embed_stage2.py before push (same mechanism as Stage 0
embed_dump.py). Locally each gate script is developed and mechanics-tested
separately; the kernel runs all three that fit in one ~50min budget:

  K1 head + transformer x18 (pure python, T<=232, fast).
  K2 conformer x17 full-layer (pure python, T<=60 deep chunks, slower).
  K3 pre_encode stem (pure python convs on mel windows, slowest per frame
      but only 36+224 small windows).

Each gate reads /kaggle/input/diar-m2-ref/*.npz + the .nemo checkpoint
(downloaded from HF at runtime, same as Stage 0 — no new dataset needed),
emits its own verdict json to /kaggle/working, and NEVER fails the kernel
on a gate FAIL (a FAIL is data: it measures the spread we pin thresholds
from). Only infra errors (missing ref/ckpt) fail loudly.

Usage locally (mechanics only): python3 kaggle/m2_stage2/embed_stage2.py
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    t0 = time.time()
    results = {}
    for gate in ("m2_stage2_k1.py", "m2_stage2_k2.py", "m2_stage2_k3.py"):
        path = HERE / gate
        if not path.exists():
            # Embedded-concat fallback: the gate code lives below in this
            # file (see embed_stage2.py); nothing to do here locally.
            results[gate] = "embedded-at-push"
            continue
        print(f"[stage2] running {gate}", flush=True)
        try:
            runpy.run_path(str(path), run_name="__main__")
            results[gate] = "ran"
        except SystemExit as exc:
            results[gate] = f"exit-{exc.code}"
        except Exception as exc:  # noqa: BLE001 — gate bugs must not kill siblings
            results[gate] = f"error: {exc!r}"
    print(f"[stage2] done in {time.time() - t0:.0f}s: {results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
