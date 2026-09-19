#!/usr/bin/env python3
"""Harvest diar-m3-step6 kernel outputs: download session outputs, parse the
verdict, print a gate table, and archive everything under
shared/diar-gpu-engine/m3-step6/.

Usage:
    python3 scripts/harvest_step6.py            # download + harvest
    python3 scripts/harvest_step6.py --skip-download --dir /tmp/foo
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REF = "weijianxue/diar-m3-step6"
DEFAULT_DIR = f"/tmp/kernels_weijianxue_diar-m3-step6_output"
ARCHIVE = Path("/var/minis/shared/diar-gpu-engine/m3-step6")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def download(out_dir: str) -> None:
    repo = Path(__file__).resolve().parent.parent
    p = run([sys.executable, str(repo / "scripts/kaggle_via_ip.py"),
             "kernels-log", "--ref", REF, "--download", "--out", out_dir],
            cwd=repo)
    tail = (p.stdout or "")[-1500:]
    print(tail)
    if p.returncode != 0:
        print("download rc:", p.returncode, file=sys.stderr)
        print((p.stderr or "")[-800:], file=sys.stderr)


def fmt(x, nd=4):
    return f"{x:.{nd}g}" if isinstance(x, (int, float)) else str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        download(args.dir)

    d = Path(args.dir)
    verdict_path = None
    for cand in [d / "step6_verdict.json", d / "m6_step6_verdict.json"]:
        if cand.exists():
            verdict_path = cand
            break
    if verdict_path is None:
        hits = list(d.glob("*verdict*.json"))
        if hits:
            verdict_path = hits[0]
    if verdict_path is None:
        print("NO VERDICT FILE FOUND in", d, file=sys.stderr)
        for f in sorted(d.glob("*")):
            print("  ", f.name, f.stat().st_size, file=sys.stderr)
        return 2

    r = json.loads(verdict_path.read_text())
    print("=" * 72)
    print("VERDICT:", r.get("verdict"))
    for reason in r.get("reasons", []):
        print("  REASON:", reason)
    print("=" * 72)

    cases = r.get("cases", {})
    for label in sorted(cases):
        c = cases[label]
        if "error" in c:
            print(f"\n## {label}: ERROR {c['error']}")
            continue
        print(f"\n## {label}  mode={c.get('mode')} face={c.get('face')}")
        for route in ("cpu", "fp32", "fp16"):
            e = c.get("routes", {}).get(route)
            if not e:
                continue
            b = e.get("bench", {})
            vf = e.get("vs_fixture") or {}
            det = b.get("determinism_bit_identical")
            mpc = b.get("ms_per_chunk_best") or b.get("ms_per_chunk")
            extra = f" vs_cpu={fmt(e.get('vs_cpu_max_abs'), 5)}" \
                if "vs_cpu_max_abs" in e else ""
            print(f"   {route:>4}: max={fmt(vf.get('max_abs'))} "
                  f"mean={fmt(vf.get('mean_abs'))} "
                  f"agree={fmt(vf.get('frame_agreement'))} "
                  f"det={det} ms/chunk={fmt(mpc, 5)}{extra}")

    g = r.get("gates", {})
    if g:
        print("\ngates:", json.dumps(g, indent=1)[:800])

    if not args.no_archive:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(verdict_path, ARCHIVE / "step6_verdict.json")
        for f in d.glob("*.log"):
            shutil.copy2(f, ARCHIVE / f.name)
        for f in d.glob("ge_t4_*.f32"):
            shutil.copy2(f, ARCHIVE / f.name)
        print("\narchived to", ARCHIVE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
