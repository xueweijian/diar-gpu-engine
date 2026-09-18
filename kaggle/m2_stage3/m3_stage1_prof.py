"""M3 Stage 1: per-stage profile of the REAL d=512 engine (profile-first).

Not a parity gate — a measurement. Runs the compiled (with
-DDIAR_PROFILE_STAGE) k5_runner over the short reference audio in the
free-running streaming path and aggregates the stage taxonomy of
profile.hpp into the decision input for M3-P100-CUDA-PLAN §2: which 3-5
operators cover >=80% of chunk time.

Report fields:
  per_stage   {stage: {ns, calls, ms_mean, pct_of_total}} sorted by ns
  chunks      ledger summary: n_chunks, emitted histogram, L rows first/
              steady (conformer/transformer cost scales with L)
  runner      wall seconds vs summed stage ns (coverage check)
  verdict     "profile-collected" | "infra-error"

Local mechanics: tests/test_m3_stage1.py exercises the aggregation on a
hand-built profile dict; no torch/weights needed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPORT: dict[str, object] = {
    "schema_version": 1,
    "job": "m3_stage1_profile",
    "plan_ref": "docs/M3-P100-CUDA-PLAN.md §2",
}

OUT = Path("/kaggle/working")
WORK = Path("/tmp/m3-stage1")


def aggregate(profile: dict) -> list[dict]:
    """Stage rows sorted by ns desc, with pct_of_total.

    >>> aggregate({"stem": {"ns": 900, "calls": 9, "ms_mean": 0.1},
    ...            "head": {"ns": 100, "calls": 9, "ms_mean": 0.011},
    ...            "total": {"ns": 1000}})[0]["stage"]
    'stem'
    """
    total = profile.get("total", {}).get("ns", 0)
    rows = []
    for stage, stat in profile.items():
        if stage == "total":
            continue
        rows.append({
            "stage": stage,
            "ns": stat["ns"],
            "calls": stat["calls"],
            "ms_mean": stat["ms_mean"],
            "pct_of_total": round(100.0 * stat["ns"] / total, 2) if total else 0.0,
        })
    rows.sort(key=lambda r: -r["ns"])
    return rows


def coverage(rows: list[dict], top_n: int) -> float:
    """Cumulative pct of the top-N stages (the 3-5 operator decision)."""
    tot = sum(r["ns"] for r in rows)
    return round(100.0 * sum(r["ns"] for r in rows[:top_n]) / tot, 2) if tot else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--dfw1", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--label", default="short")
    ap.add_argument("--out", default=str(OUT / "m3_stage1_profile.json"))
    args = ap.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    runner = Path(args.runner)
    dfw1 = Path(args.dfw1)
    z = np.load(str(Path(args.ref_dir) / f"m2_ref_{args.label}.npz"), allow_pickle=True)
    audio = np.ascontiguousarray(np.asarray(z["audio"], dtype="<f4"))
    pcm = WORK / f"{args.label}.f32"
    pcm.write_bytes(audio.tobytes())

    prefix = WORK / f"{args.label}_whole"
    prof_path = WORK / "m3_prof_stage.json"
    t0 = time.time()
    p = subprocess.run(
        [str(runner), "--run", "--weights", str(dfw1), "--audio", str(pcm),
         "--out", str(prefix), "--feed", "whole", "--profile-out", str(prof_path)],
        capture_output=True, text=True, timeout=4 * 3600)
    wall = time.time() - t0
    REPORT["runner_rc"] = p.returncode
    REPORT["runner_wall_seconds"] = round(wall, 1)
    if p.returncode != 0:
        REPORT["verdict"] = "infra-error"
        REPORT["stderr_tail"] = (p.stderr or p.stdout)[-2000:]
        _emit(args.out)
        print(json.dumps(REPORT, indent=1)[:3000], flush=True)
        return 0

    profile = json.loads(prof_path.read_text())
    rows = aggregate(profile)
    REPORT["per_stage"] = rows
    REPORT["top3_pct"] = coverage(rows, 3)
    REPORT["top5_pct"] = coverage(rows, 5)

    ledger = json.loads(Path(str(prefix) + ".ledger.json").read_text())
    emitted = [int(e["emitted"]) for e in ledger]
    window = [int(e["window_frames"]) for e in ledger]
    REPORT["chunks"] = {
        "n_chunks": len(ledger),
        "emitted_total": sum(emitted),
        "window_first": window[0] if window else 0,
        "window_last": window[-1] if window else 0,
        "window_max": max(window) if window else 0,
        "t_mel_last": int(ledger[-1]["t_mel"]) if ledger else 0,
    }
    summed_ns = sum(r["ns"] for r in rows)
    REPORT["profile_vs_wall"] = {
        "summed_stage_ns": summed_ns,
        "wall_ns": int(wall * 1e9),
        "coverage_pct": round(100.0 * summed_ns / (wall * 1e9), 2),
        "note": "wall includes weight load, FE scheduling and host IO not "
                "attributed to any stage; coverage >=70% is expected",
    }
    REPORT["verdict"] = "profile-collected"
    _emit(args.out)

    print(f"[m3s1] {len(ledger)} chunks, wall {wall:.0f}s, "
          f"top3={REPORT['top3_pct']}% top5={REPORT['top5_pct']}%", flush=True)
    for r in rows:
        print(f"[m3s1]   {r['stage']:<12} {r['pct_of_total']:6.2f}%  "
              f"{r['ms_mean']:10.2f} ms/call x{r['calls']}", flush=True)
    return 0


def _emit(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
