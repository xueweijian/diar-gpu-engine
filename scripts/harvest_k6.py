#!/usr/bin/env python3
"""Harvest + summarize m2_stage3_k6_verdict.json (q8 four-fixture gate).

Usage:
  python3 scripts/harvest_k6.py <verdict.json> [--full]

Exit is always 0: a red verdict is data, not a tool error.
"""
from __future__ import annotations

import json
import sys


def fmt(v, nd=6):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def summarize(d: dict) -> list[str]:
    lines: list[str] = []
    lines.append(f"verdict : {d.get('verdict')}")
    lines.append(f"finished: {d.get('finished_utc')}  seconds={d.get('seconds')}")
    lines.append(f"gates   : {json.dumps(d.get('gates', {}))}")
    if d.get("reasons"):
        lines.append("reasons :")
        for r in d["reasons"]:
            lines.append(f"  - {r}")
    if d.get("notes"):
        lines.append("notes   :")
        for n in d["notes"]:
            lines.append(f"  - {n}")

    g0 = d.get("G0_selftest", {})
    p0 = d.get("G0_gguf_probe", {})
    lines.append(f"G0 selftest rc={g0.get('rc')}  gguf_probe rc={p0.get('rc')}")

    lines.append("")
    lines.append(f"{'case':<28} {'cmp':<9} {'rows':>11} {'d':>3} "
                 f"{'max_abs':>10} {'mean_abs':>10} {'agree':>8} {'det':>5}")
    for label, r in (d.get("per_case") or {}).items():
        if "error" in r:
            lines.append(f"{label:<28} ERROR: {r['error'][:120]}")
            continue
        rows = f"{r.get('rows_ours')}/{r.get('rows_ref')}"
        det = r.get("determinism_bit_identical")
        det_s = "-" if det is None else ("yes" if det else "NO")
        lines.append(
            f"{label:<28} {str(r.get('compare_file')):<9} {rows:>11} "
            f"{r.get('rows_delta', 0):>+3d} {fmt(r.get('max_abs')):>10} "
            f"{fmt(r.get('mean_abs')):>10} {fmt(r.get('frame_agreement')):>8} "
            f"{det_s:>5}")
    # percentile detail per case (threshold-pinning data)
    for label, r in (d.get("per_case") or {}).items():
        if "error" in r:
            continue
        lines.append(
            f"  {label}: row_max p50={fmt(r.get('row_max_p50'))} "
            f"p95={fmt(r.get('row_max_p95'))} p99={fmt(r.get('row_max_p99'))} "
            f"worst_row={r.get('worst_row')} args={' '.join(r.get('runner_args') or [])}")
        for key in ("ours_tail_rows_max", "ref_tail_rows_max"):
            if key in r:
                lines.append(f"    {key}={fmt(r[key])}")
    return lines


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    full = "--full" in sys.argv
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for line in summarize(d):
        print(line)
    if full:
        print(json.dumps(d.get("per_case", {}), indent=1)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
