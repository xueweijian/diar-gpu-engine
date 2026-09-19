#!/usr/bin/env python3
"""Harvest + merge the M3 Step 6b SPLIT sessions (4 CPU + 1 GPU kernel).

The 5-way split replaces the single ~4.9h session with a ~2.3h wall:
  s6b-a   (cpu): v12-short-streaming (cpu x2) + v12-mid-offline-full (cpu x2)
  s6b-b   (cpu): v13-mid-streaming cpu rep0
  s6b-c   (cpu): v13-mid-streaming cpu rep1  (det = b-vs-c bit diff at merge)
  s6b-d   (cpu): v13-mid-offline-preset (cpu x2)
  s6b-gpu (T4) : fp32/fp16/fp32jit routes on short/streaming/preset

Merge computes the gates that no single session could:
  G-B2  route-vs-cpu  (local wire diff: gpu dumps vs cpu dumps)
  G-C   cross-session determinism for split-rep mid-streaming
Everything else passes through from per-session verdicts.

Usage:
    python3 scripts/harvest_step6b.py            # download + merge + judge
    python3 scripts/harvest_step6b.py --skip-download
"""
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIA_IP = str(ROOT / "scripts" / "kaggle_via_ip.py")
SLUGS = {
    "a": "weijianxue/diar-m3-s6b-a",
    "b": "weijianxue/diar-m3-s6b-b",
    "c": "weijianxue/diar-m3-s6b-c",
    "d": "weijianxue/diar-m3-s6b-d",
    "gpu": "weijianxue/diar-m3-s6b-gpu",
}
ARCHIVE = ROOT / "shared" / "diar-gpu-engine" / "m3-step6b"
OFFICIAL_MS = 45.5  # NeMo T4 streaming fp32 (m3-official-bench v4)
# K6-tier fixture gates (mirrors kaggle/m6_step6/framework.py judge())
HARD = {"max_abs": 0.05, "mean_abs": 0.005, "frame_agreement": 0.999}
ADV = {"max_abs": 1.0, "mean_abs": 0.02, "frame_agreement": 0.99}
ADVISORY = ("v13-mid-streaming",)  # v13 autotune-noise tier, advisory


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def download(sess: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = run(["python3", VIA_IP, "kernels-log", "--ref", SLUGS[sess],
             "--download", "--out", str(out_dir)])
    if p.returncode != 0:
        print(f"download[{sess}] rc={p.returncode}\n{p.stdout[-500:]}"
              f"\n{p.stderr[-500:]}", file=sys.stderr)
    return out_dir


def load_verdict(sess_dir: Path) -> dict:
    for cand in ("step6_verdict.json", "m3_step6_verdict.json"):
        f = sess_dir / cand
        if f.exists():
            return json.loads(f.read_text())
    raise SystemExit(f"no verdict json under {sess_dir}")


def load_wire(path: Path):
    """probdump wire format: 12B '<qi>' header (shape) + f32 payload."""
    raw = path.read_bytes()
    qi = raw[:12]
    shape = struct.unpack("<qii", qi)
    import numpy as np
    arr = np.frombuffer(raw[12:], dtype="<f4")
    return shape, arr


def wire_diff(a: Path, b: Path) -> float:
    import numpy as np
    sa, ra = load_wire(a)
    sb, rb = load_wire(b)
    assert sa == sb, f"shape mismatch {a.name} {sa} vs {b.name} {sb}"
    n = min(len(ra), len(rb))
    return float(np.abs(ra[:n].astype(np.float64) -
                        rb[:n].astype(np.float64)).max())


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--dir", default="/tmp/s6b_merge")
    args = ap.parse_args()
    base = Path(args.dir)

    if not args.skip_download:
        for s in SLUGS:
            download(s, base / s)

    V = {s: load_verdict(base / s) for s in SLUGS}

    # ---- pass-through gates from the gpu session ------------------------
    reasons = []
    g = V["gpu"]["gates"]
    for k in ("cpu_selftest", "fatbin_3sass", "fatbin_ptx60", "jit_bin_ptx_only"):
        if g.get(k) is not True:
            reasons.append(f"gpu-session gate {k}={g.get(k)!r}")

    # ---- per-case merge -------------------------------------------------
    merged = {}
    for s in ("a", "b", "c", "d", "gpu"):
        for label, c in V[s].get("cases", {}).items():
            if "error" in c:
                reasons.append(f"[{s}] {label}: {c['error']}")
                continue
            entry = merged.setdefault(label, {"sessions": {}})
            entry["sessions"][s] = c

    table = {}
    for label, entry in sorted(merged.items()):
        tier = ADV if any(a in label for a in ADVISORY) else HARD
        row = {}
        for sess, c in entry["sessions"].items():
            for route, r in c.get("routes", {}).items():
                b = r.get("bench", {})
                vf = r.get("vs_fixture")
                row[route] = {
                    "ms_chunk": b.get("ms_per_chunk_best"),
                    "det": b.get("determinism_bit_identical"),
                    "vs_fixture_max": None if vf is None else vf.get("max_abs"),
                    "rows_delta": None if vf is None else vf.get("rows_delta"),
                    "vs_cpu_max": r.get("vs_cpu_max_abs"),
                    "jit_vs_sass": r.get("jit_vs_sass_max_abs"),
                    "headed": (b.get("route_head") or {}).get("headed_calls"),
                    "stem_refused": (b.get("stem_route") or {}).get("refused"),
                }
        table[label] = row

    # ---- G-B2 at merge time: gpu dumps vs cpu dumps ----------------------
    def find(sess_dir: Path, pattern: str):
        hits = sorted(sess_dir.rglob(pattern))
        return hits[0] if hits else None

    for label in ("v12-short-streaming-r0", "v13-mid-streaming-r0",
                  "v13-mid-offline-preset-r0"):
        if label not in merged:
            continue
        cpu_sess = {"v12-short-streaming-r0": "a",
                    "v13-mid-streaming-r0": "b",
                    "v13-mid-offline-preset-r0": "d"}[label]
        cpu_tag = "cpu" if cpu_sess != "b" else "cpu.rep0"
        for face in ("pregate", "postgate"):
            cpu_f = find(base / cpu_sess, f"{label}.{cpu_tag}.{face}.f32")
            for route in ("fp32", "fp16", "fp32jit"):
                rf = find(base / "gpu", f"{label}.{route}.{face}.f32")
                if cpu_f is None or rf is None:
                    continue
                d = wire_diff(rf, cpu_f)
                table[label].setdefault(route, {})["vs_cpu_merge"] = d
                if route in ("fp32", "fp32jit") and d > 0.05:
                    reasons.append(f"{label}/{route} vs_cpu(merge) {d:.3g} > 0.05")

    # ---- G-C cross-session: mid-streaming cpu rep0 vs rep1 ---------------
    if "v13-mid-streaming-r0" in merged:
        r0 = find(base / "b", "v13-mid-streaming-r0.cpu.rep0.postgate.f32")
        r1 = find(base / "c", "v13-mid-streaming-r0.cpu.rep1.postgate.f32")
        if r0 is not None and r1 is not None:
            d = wire_diff(r0, r1)
            table["v13-mid-streaming-r0"]["cpu_cross_det"] = d
            if d != 0.0:
                reasons.append(f"mid-streaming cpu rep0-vs-rep1 cross det {d:.3g} != 0")
        else:
            reasons.append("cross-det dumps missing (b/c)")

    # ---- G-D perf table vs official --------------------------------------
    perf = {}
    for label, row in table.items():
        for route, m in row.items():
            if isinstance(m, dict) and m.get("ms_chunk"):
                perf[f"{label}/{route}"] = {
                    "ms_chunk": round(m["ms_chunk"], 2),
                    "vs_official": round(OFFICIAL_MS / m["ms_chunk"], 2),
                }

    # ---- in-process det + fixture gates from each session ----------------
    for label, row in table.items():
        tier = ADV if any(a in label for a in ADVISORY) else HARD
        for route, m in row.items():
            if not isinstance(m, dict):
                continue
            if m.get("det") is False:
                reasons.append(f"{label}/{route}: in-process determinism red")
            vf = m.get("vs_fixture_max")
            if vf is None and route == "cpu":
                reasons.append(f"{label}/{route}: no vs_fixture metrics")
            if vf is not None and route in ("cpu", "fp32", "fp32jit"):
                t = tier
                if vf > t["max_abs"]:
                    reasons.append(f"{label}/{route}: vs_fixture {vf:.4g} "
                                   f"> {t['max_abs']}")
            if m.get("stem_refused") not in (None, 0):
                reasons.append(f"{label}/{route}: stem refused "
                               f"{m['stem_refused']}")
            if route == "fp32jit" and m.get("jit_vs_sass") not in (None, 0.0):
                reasons.append(f"{label}/fp32jit: jit_vs_sass "
                               f"{m['jit_vs_sass']:.3g} != 0")

    verdict = "s6b-green" if not reasons else "s6b-red"
    out = {"verdict": verdict, "reasons": reasons, "table": table,
           "perf": perf, "official_ms_chunk": OFFICIAL_MS,
           "sessions": {s: {"role": V[s].get("role")} for s in SLUGS}}

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "step6b_merged_verdict.json").write_text(
        json.dumps(out, indent=1, default=str))
    print(json.dumps({"verdict": verdict,
                      "reasons": reasons[:12]}, indent=1, default=str))
    print("\nperf (vs official 45.5 ms/chunk):")
    for k, v in sorted(perf.items(), key=lambda kv: -kv[1]["vs_official"]):
        print(f"  {k:44s} {v['ms_chunk']:>9.2f} ms  {v['vs_official']:>5.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
