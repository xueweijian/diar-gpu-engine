#!/usr/bin/env python3
"""Harvest + summarize m2_stage3_k5a_verdict.json.

Usage:
  python3 scripts/harvest_k5a.py <verdict.json> [--full]
"""
import json
import sys


def fmt(v, nd=6):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def summarize_audio(label, r):
    print(f"\n=== {label} ({r.get('seconds', '?')}s) ===")
    g5 = r.get("G5_length", {})
    print(f"  G5 rows: ours={g5.get('ours')} nemo={g5.get('nemo')} "
          f"expected_ours={g5.get('expected_ours')} diff={g5.get('diff')} "
          f"phantom={g5.get('phantom_rows')} all_zero={g5.get('phantom_all_zero')}")
    g1 = r.get("G1_chunk0", {})
    print(f"  G1 chunk0: max_abs={fmt(g1.get('max_abs'))} "
          f"mean_abs={fmt(g1.get('mean_abs'))} frame_agreement={fmt(g1.get('frame_agreement'))}")
    g2 = r.get("G2_timeline", {})
    ov = g2.get("overall", {})
    print(f"  G2 overall: max_abs={fmt(ov.get('max_abs'))} "
          f"mean_abs={fmt(ov.get('mean_abs'))} frame_agreement={fmt(ov.get('frame_agreement'))} "
          f"band={fmt(g2.get('argmax_band_agreement'))}")
    print(f"     pre_1st_compress_max={fmt(g2.get('pre_first_compress_max'))} "
          f"post_1st_compress_max={fmt(g2.get('post_first_compress_max'))} "
          f"@chunk={g2.get('first_compress_chunk')} worst_chunk={g2.get('worst_chunk')}")
    pcm = g2.get("per_chunk_max_abs", [])
    if pcm and len(pcm) <= 40:
        print(f"     per_chunk={[fmt(m) for m in pcm]}")
    else:
        bad = [(i, m) for i, m in enumerate(pcm)
               if not isinstance(m, dict) and (m is None or m > 1e-3)]
        print(f"     per_chunks>1e-3: {[(i, fmt(m)) for i, m in bad[:12]]} "
              f"(n={len(pcm)})")
    g3 = r.get("G3_aosc", {})
    print(f"  G3 aosc: geometry_ok={g3.get('geometry_ok')} "
          f"spk={fmt(g3.get('spk_max_abs'))} fifo={fmt(g3.get('fifo_max_abs'))} "
          f"mean_sil={fmt(g3.get('mean_sil_max_abs'))}")
    g4 = r.get("G4_compress", {})
    print(f"  G4 compress: match={g4.get('match')} ours={g4.get('ours_sim')} "
          f"nemo={g4.get('nemo')}")
    g6 = r.get("G6_determinism", {})
    print(f"  G6 determinism: bit_identical={g6.get('bit_identical')} "
          f"n_frames_equal={g6.get('n_frames_equal')}")


def main():
    path = sys.argv[1]
    full = "--full" in sys.argv
    d = json.load(open(path))
    print(f"verdict : {d.get('verdict')}")
    print(f"finished: {d.get('finished_utc')}")
    print(f"gates   : {json.dumps(d.get('gates', {}))}")
    if d.get("reasons"):
        print("reasons :")
        for r in d["reasons"]:
            print(f"  - {r}")
    g0 = d.get("G0_selftest", {})
    print(f"G0 selftest rc={g0.get('rc')}")
    for label, r in (d.get("per_audio") or {}).items():
        if "error" in r:
            print(f"\n=== {label} ===\n  ERROR: {r['error']}")
            continue
        summarize_audio(label, r)
        if full:
            print(json.dumps(r, indent=1)[:8000])


if __name__ == "__main__":
    main()
