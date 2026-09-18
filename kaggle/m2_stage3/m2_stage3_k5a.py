"""M2 Stage 3 K5-A: free-running DiarEngine (fp32 DFW1) vs NeMo m2-ref npz.

The Stage 2 gates were teacher-forced (each block ate the reference
predecessor). K5-A is the END-TO-END gate: the compiled C++ engine runs the
raw audio free-running — its own FE, scheduler, AOSC feedback, BirthGate —
and every comparison walks the npz chunk by chunk:

  G0 infra      k5_runner --selftest (tiny DFW1 -> load -> run -> determinism)
  G1 chunk0     open-loop (empty state): emitted rows [0:20] vs total_preds
                [0:20], max_abs <= 1e-5 (hard; the plan's open-loop gate)
  G2 timeline   per-chunk emitted slices vs total_preds: series reported,
                hard safety <= 0.05 max_abs and frame_agreement >= 0.999
                (M1 frame-gate tiers); pre/post-first-compress split reported
                for threshold pinning (plan §3.3 layered gates)
  G3 AOSC state spkcache/fifo/mean_sil embeddings after EVERY chunk vs
                spkcache_after / fifo_after / mean_sil_emb_after: geometry
                (frame counts) hard, embedding max_abs reported + 0.05 hard
  G4 compress   our compress events (derived from the ledger's spkcache
                evolution) == npz compression_chunks (hard, exact)
  G5 length     our n_frames vs NeMo total_preds rows: python NeMo pads the
                final window to a 32-mel multiple and emits masked pad rows
                (all-zero probs) — short +1, mid +2 phantoms. Gate: our rows
                == npz rows - phantom_count (phantom derived, hard) and the
                phantom rows themselves must be all-zero (hard); comparison
                runs on the real prefix
  G6 determinism whole-feed vs 1-sample-drip feed bit-identical (hard)

A gate FAIL is data (verdict json still written; kernel exits 0) — only
infra errors (missing ref/ckpt/runner) fail the kernel loudly.

Local mechanics: tests/test_k5a_gates.py runs the metric helpers + a tiny
synthetic end-to-end (k5_runner selftest outputs vs themselves) with no
torch/weights; the kernel only adds the real DFW1 + npz.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPORT: dict[str, object] = {
    "schema_version": 1,
    "scope": "pure_speaker_diarization",
    "job": "m2_stage3_k5a_free_running",
}

OUT = Path("/kaggle/working")
WORK = Path("/tmp/m2-stage3")

# Gate constants (plan §3.3: chunk0 open-loop 1e-5; closed-loop layered —
# first measurement reports the series; safety tiers mirror M1 frame gates).
G1_CHUNK0_MAX_ABS = 1e-5
G2_SAFETY_MAX_ABS = 0.05
G2_FRAME_AGREEMENT = 0.999
G3_SAFETY_MAX_ABS = 0.05


def metrics(ref: np.ndarray, got: np.ndarray) -> dict:
    r = np.asarray(ref, dtype=np.float64)
    g = np.asarray(got, dtype=np.float64)
    n = min(r.shape[0], g.shape[0])
    r, g = r[:n], g[:n]
    d = np.abs(r - g)
    ra = (r > 0.5).astype(int)
    ga = (g > 0.5).astype(int)
    cos_denom = np.sqrt((r * r).sum() * (g * g).sum()) + 1e-12
    return {
        "rows_compared": int(n),
        "max_abs": float(d.max()) if n else 0.0,
        "mean_abs": float(d.mean()) if n else 0.0,
        "cosine": float((r * g).sum() / cos_denom) if n else 1.0,
        "frame_agreement": float((ra == ga).mean()) if n else 1.0,
    }


def load_npz(path: Path):
    return np.load(str(path), allow_pickle=True)


def run_runner(runner: Path, args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([str(runner), *args], capture_output=True, text=True, timeout=timeout)


def run_feed(label: str, feed: str, runner: Path, dfw1: Path, work: Path,
             pcm: Path, n_spk: int, offline: bool = False) -> tuple[str, dict]:
    """Run ONE feed of one case (the unit of parallelism).

    Each job is an independent k5_runner process writing its own
    <label>_<feed>.* files, so jobs are order-free and bit-identical to
    running them sequentially (same binary, same inputs).
    """
    prefix = str(work / f"{label}_{feed}")
    args = ["--run", "--weights", str(dfw1), "--audio", str(pcm),
            "--out", prefix, "--feed", feed]
    if offline:
        args.append("--offline")
    p = run_runner(runner, args)
    if p.returncode != 0:
        return feed, {"error": f"runner rc={p.returncode}: {(p.stderr or p.stdout)[-400:]}"}
    return feed, {
        "meta": json.loads(Path(prefix + ".meta.json").read_text()),
        "ledger": json.loads(Path(prefix + ".ledger.json").read_text()),
        "aosc_index": json.loads(Path(prefix + ".aosc.json").read_text()),
        "aosc_f32": prefix + ".aosc.f32",
        "pregate": np.fromfile(prefix + ".pregate.f32", dtype="<f4").reshape(-1, n_spk),
    }


def compare_audio(label: str, z, runner: Path, dfw1: Path, work: Path,
                  n_spk: int, offline: bool = False, runs: dict | None = None) -> dict:
    """Free-run one npz case and gate it. Returns the report subtree.

    ``runs`` may carry pre-executed {feed: run-record} (parallel driver);
    when None the feeds run sequentially here (legacy path / tests).
    """
    audio = np.asarray(z["audio"], dtype=np.float32)
    tp = np.asarray(z["total_preds"], dtype=np.float64)
    geometry = [int(v) for v in z["geometry"]]
    n_chunks = int(z["n_chunks"][0])
    sub = 8  # compiled FE pin

    pcm = work / f"{label}.f32"
    pcm.write_bytes(np.ascontiguousarray(audio, dtype="<f4").tobytes())

    if runs is None:
        runs = {}
        for feed in ("whole", "drip"):
            feed, rec = run_feed(label, feed, runner, dfw1, work, pcm, n_spk, offline)
            runs[feed] = rec
    for rec in runs.values():
        if "error" in rec:
            return {"error": rec["error"]}

    ours = runs["whole"]["pregate"]
    ledger = runs["whole"]["ledger"]
    n_ours = ours.shape[0]

    # ---- G5 length (python-NeMo phantom tail rows) ----
    # valid final rows = subsampled_len(final feat_length), the SAME
    # (len+1)/2 chain the production stem uses; python pads to a 32-mel
    # multiple and appends masked pad rows (all-zero probs).
    n_chunks_npz = int(z["n_chunks"][0])
    last_key = f"chunk{n_chunks_npz - 1:03d}"
    feat_len_last = int(np.asarray(z[f"{last_key}/feat_length"]).ravel()[0])
    valid_last = feat_len_last
    for _ in range(3):  # subsampling_factor 8 -> three (len+1)/2 stages
        valid_last = (valid_last + 1) // 2
    expected_ours = (int(tp.shape[0])
                     - (int(np.asarray(z[f"{last_key}/preds_full"]).shape[0])
                        - int(z[f"{last_key}/state_lens_before"][0])
                        - int(z[f"{last_key}/state_lens_before"][1]))
                     + valid_last)
    phantom = int(tp.shape[0]) - expected_ours
    phantom_rows = tp[expected_ours:] if phantom > 0 else np.zeros((0, tp.shape[1]))
    phantom_all_zero = bool(np.all(phantom_rows == 0.0)) if phantom > 0 else True
    len_diff = int(n_ours - expected_ours)

    # ---- G1 chunk0 open-loop ----
    g1 = metrics(tp[:20], ours[:20]) if n_ours >= 20 else {"error": "short timeline"}

    # ---- G2 per-chunk emitted slices ----
    per_chunk = []
    cum = 0
    for e in ledger:
        lo, hi = cum, cum + e["emitted"]
        m = metrics(tp[lo:hi], ours[lo:hi]) if hi <= min(n_ours, tp.shape[0]) else \
            {"error": f"emitted range {lo}:{hi} out of bounds"}
        m.update({"chunk": e["chunk"], "emitted": e["emitted"],
                  "spkcache_frames": e["spkcache_frames"],
                  "fifo_frames": e["fifo_frames"]})
        per_chunk.append(m)
        cum = hi
    first_compress = None
    # G4 compress-event simulation (ledger + geometry state machine) — also
    # splits the pre/post first-compress reporting tiers.
    events_sim: list[int] = []
    spk = 0
    fifo = 0
    cap = geometry[4]
    period = geometry[5]
    for e in ledger:
        fifo += e["emitted"]
        if fifo > geometry[3]:
            pop = min(max(period, fifo - geometry[3]), fifo)
            spk += pop
            fifo -= pop
        if spk > cap:
            events_sim.append(e["chunk"])
            spk = cap  # compress clamps back to cap
    if events_sim:
        first_compress = events_sim[0]
    pre = [m["max_abs"] for m in per_chunk if isinstance(m.get("max_abs"), float) and
           (first_compress is None or m["chunk"] <= first_compress)]
    post = [m["max_abs"] for m in per_chunk if isinstance(m.get("max_abs"), float) and
            first_compress is not None and m["chunk"] > first_compress]
    overall = metrics(tp, ours)
    # frame agreement over the common prefix (argmax bands)
    n_cmp = min(n_ours, tp.shape[0])
    band = int((np.argmax(tp[:n_cmp], axis=1) == np.argmax(ours[:n_cmp], axis=1)).mean() * 10000) / 10000.0

    # ---- G3 AOSC state embeddings ----
    aosc = []
    snaps = runs["whole"]["aosc_index"]
    # Blob path travels WITH the run record (label-derived fallback keeps
    # older callers working): the parallel driver's runs may have been
    # executed under a different label than this comparison's.
    blob_path = runs["whole"].get("aosc_f32", str(work / f"{label}_whole.aosc.f32"))
    blob = np.fromfile(blob_path, dtype="<f4")
    state_geometry_ok = True
    for i, (snap, ci) in enumerate(zip(snaps, [f"chunk{c:03d}" for c in range(len(snaps))])):
        entry: dict = {"chunk": i}
        for kind, npz_key, frames, off in (
            ("spk", "spkcache_after", "spk_frames", "spk_off"),
            ("fifo", "fifo_after", "fifo_frames", "fifo_off"),
        ):
            n_ref = z.get(f"{ci}/{npz_key}")
            if n_ref is not None:
                ref = np.asarray(n_ref, dtype=np.float64)
                got = blob[snap[off]:snap[off] + ref.size].astype(np.float64).reshape(ref.shape)
                if ref.shape[0] != snap[frames]:
                    entry[kind] = {"geometry-mismatch": [snap[frames], int(ref.shape[0])]}
                    state_geometry_ok = False
                else:
                    entry[kind] = metrics(ref, got)
        ms_ref = z.get(f"{ci}/mean_sil_emb_after")
        if ms_ref is not None:
            ref = np.asarray(ms_ref, dtype=np.float64)
            got = blob[snap["mean_off"]:snap["mean_off"] + ref.size].astype(np.float64)
            entry["mean_sil"] = metrics(ref, got)
        sil_ref = z.get(f"{ci}/n_sil_frames_after")
        if sil_ref is not None:
            entry["silence_frames"] = {"ours": int(snap["silence_frames"]),
                                       "nemo": int(np.asarray(sil_ref).ravel()[0])}
        aosc.append(entry)

    def _series_max(entries, key):
        vals = []
        for e in entries:
            v = e.get(key)
            if isinstance(v, dict) and "max_abs" in v:
                vals.append(v["max_abs"])
        return max(vals) if vals else None

    # ---- G4 compress events ----
    npz_events = [int(v) for v in np.asarray(z["compression_chunks"]).ravel()]

    return {
        "geometry": geometry,
        "n_chunks_npz": n_chunks,
        "n_chunks_ours": len(ledger),
        "G5_length": {"ours": int(n_ours), "nemo": int(tp.shape[0]),
                      "expected_ours": int(expected_ours), "diff": len_diff,
                      "phantom_rows": phantom,
                      "phantom_all_zero": phantom_all_zero},
        "G1_chunk0": g1,
        "G2_timeline": {
            "overall": overall,
            "argmax_band_agreement": band,
            "per_chunk_max_abs": [m.get("max_abs") if "max_abs" in m else m
                                  for m in per_chunk],
            "worst_chunk": max(per_chunk, key=lambda m: m.get("max_abs", -1))["chunk"],
            "pre_first_compress_max": max(pre) if pre else None,
            "post_first_compress_max": max(post) if post else None,
            "first_compress_chunk": first_compress,
        },
        "G3_aosc": {
            "geometry_ok": state_geometry_ok,
            "spk_max_abs": _series_max(aosc, "spk"),
            "fifo_max_abs": _series_max(aosc, "fifo"),
            "mean_sil_max_abs": _series_max(aosc, "mean_sil"),
            "per_chunk": aosc,
        },
        "G4_compress": {"ours_sim": events_sim, "nemo": npz_events,
                        "match": events_sim == npz_events},
        "G6_determinism": {
            "n_frames_equal": runs["whole"]["meta"]["n_frames"] == runs["drip"]["meta"]["n_frames"],
            "bit_identical": bool(np.array_equal(runs["whole"]["pregate"],
                                                 runs["drip"]["pregate"])),
        },
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--dfw1", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--labels", default="short,mid")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel k5_runner processes (each run is an "
                         "independent process; results are bit-identical "
                         "to sequential execution)")
    ap.add_argument("--out", default=str(OUT / "m2_stage3_k5a_verdict.json"))
    args = ap.parse_args()

    runner = Path(args.runner)
    dfw1 = Path(args.dfw1)
    ref_dir = Path(args.ref_dir)
    REPORT["runner"] = str(runner)
    REPORT["dfw1"] = str(dfw1)
    REPORT["ref_dir"] = str(ref_dir)

    # ---- G0 infra ----
    p = run_runner(runner, ["--selftest"])
    REPORT["G0_selftest"] = {
        "rc": p.returncode,
        "tail": (p.stdout or p.stderr)[-400:],
    }
    if p.returncode != 0:
        REPORT["verdict"] = "infra-fail-selftest"
        _emit(args.out)
        return 0

    WORK.mkdir(parents=True, exist_ok=True)
    labels = [lb for lb in args.labels.split(",")]
    zs: dict[str, object] = {}
    for label in labels:
        zpath = ref_dir / f"m2_ref_{label}.npz"
        if zpath.exists():
            zs[label] = load_npz(zpath)

    # ---- parallel driver: one job per (label, feed) ----
    # Jobs are independent processes; long audio first so the critical path
    # starts immediately even when workers < jobs.
    jobs: list[tuple[str, str]] = []
    pcm_paths: dict[str, Path] = {}
    for label in labels:
        z = zs.get(label)
        if z is None:
            continue
        audio = np.asarray(z["audio"], dtype=np.float32)
        pcm = WORK / f"{label}.f32"
        pcm.write_bytes(np.ascontiguousarray(audio, dtype="<f4").tobytes())
        pcm_paths[label] = pcm
        jobs += [(label, "whole"), (label, "drip")]
    jobs.sort(key=lambda j: -(pcm_paths[j[0]].stat().st_size))
    n_workers = max(1, min(args.jobs, len(jobs)))
    print(f"[k5a] jobs={len(jobs)} workers={n_workers} "
          f"({', '.join(f'{lb}/{fd}' for lb, fd in jobs)})", flush=True)
    REPORT["jobs"] = [{"label": lb, "feed": fd} for lb, fd in jobs]
    REPORT["workers"] = n_workers

    runs_by_label: dict[str, dict] = {lb: {} for lb in pcm_paths}
    t0_all = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {}
        for label, feed in jobs:
            z = zs[label]
            n_spk = int(np.asarray(z["total_preds"]).shape[1])
            futs[ex.submit(run_feed, label, feed, runner, dfw1, WORK,
                           pcm_paths[label], n_spk, False)] = (label, feed)
        for fut in as_completed(futs):
            label, feed = futs[fut]
            try:
                feed_name, rec = fut.result()
                runs_by_label[label][feed_name] = rec
            except Exception as exc:  # keep the other jobs' results
                runs_by_label[label][feed] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[k5a] done {label}/{feed} "
                  f"(t+{time.time() - t0_all:.0f}s)", flush=True)

    per_audio: dict[str, dict] = {}
    for label in labels:
        if label not in zs:
            per_audio[label] = {"error": f"missing {ref_dir / f'm2_ref_{label}.npz'}"}
            continue
        z = zs[label]
        n_spk = int(np.asarray(z["total_preds"]).shape[1])
        t0 = time.time()
        per_audio[label] = compare_audio(label, z, runner, dfw1, WORK, n_spk,
                                         runs=runs_by_label[label])
        per_audio[label]["seconds"] = round(time.time() - t0, 1)
    REPORT["per_audio"] = per_audio
    REPORT["runner_wall_seconds"] = round(time.time() - t0_all, 1)

    # ---- verdict ----
    reasons: list[str] = []
    for label, r in per_audio.items():
        if "error" in r:
            reasons.append(f"{label}: {r['error']}")
            continue
        g1 = r["G1_chunk0"]
        if "max_abs" not in g1 or g1["max_abs"] > G1_CHUNK0_MAX_ABS:
            reasons.append(f"{label}: G1 chunk0 max_abs={g1.get('max_abs')}")
        g2 = r["G2_timeline"]
        if g2["overall"]["max_abs"] > G2_SAFETY_MAX_ABS:
            reasons.append(f"{label}: G2 safety max_abs={g2['overall']['max_abs']}")
        if g2["overall"]["frame_agreement"] < G2_FRAME_AGREEMENT:
            reasons.append(f"{label}: G2 agreement={g2['overall']['frame_agreement']}")
        g3 = r["G3_aosc"]
        if not g3["geometry_ok"]:
            reasons.append(f"{label}: G3 state geometry mismatch")
        for key, gate in (("spk_max_abs", G3_SAFETY_MAX_ABS),
                          ("fifo_max_abs", G3_SAFETY_MAX_ABS),
                          ("mean_sil_max_abs", G3_SAFETY_MAX_ABS)):
            v = g3[key]
            if v is not None and v > gate:
                reasons.append(f"{label}: G3 {key}={v}")
        if not r["G4_compress"]["match"]:
            reasons.append(f"{label}: G4 compress events {r['G4_compress']['ours_sim']} "
                           f"!= {r['G4_compress']['nemo']}")
        if r["G5_length"]["diff"] != 0:
            reasons.append(f"{label}: G5 rows {r['G5_length']['ours']} != "
                           f"expected {r['G5_length']['expected_ours']}")
        if not r["G5_length"]["phantom_all_zero"]:
            reasons.append(f"{label}: G5 phantom rows not all-zero")
        if not r["G6_determinism"]["bit_identical"]:
            reasons.append(f"{label}: G6 feed-sharding not bit-identical")

    REPORT["gates"] = {
        "G1_chunk0_max_abs": G1_CHUNK0_MAX_ABS,
        "G2_safety_max_abs": G2_SAFETY_MAX_ABS,
        "G2_frame_agreement": G2_FRAME_AGREEMENT,
        "G3_safety_max_abs": G3_SAFETY_MAX_ABS,
    }
    REPORT["verdict"] = "k5a-green" if not reasons else "k5a-red"
    REPORT["reasons"] = reasons
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _emit(args.out)
    print(json.dumps({k: v for k, v in REPORT.items()
                      if k not in ("per_audio",)}, indent=1)[:3000])
    return 0


def _emit(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
