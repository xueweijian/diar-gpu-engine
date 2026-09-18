"""M2 Stage 3 K6: q8-vs-q8 four-fixture endorsement (plan §3.4).

The M1 fixtures are POST-BirthGate production timelines (q8 GGUF, upstream
binary, T4). K6 runs OUR engine on the SAME q8 GGUF with the SAME case
geometry and compares against the fixture probs:

  case                       production CLI        our runner           compare
  v12-short-streaming-r0     (default streaming)   --run                postgate
  v13-mid-streaming-r0       (default streaming)   --run                postgate
  v13-mid-offline-preset-r0  --preset offline      --run --offline      postgate
  v12-mid-offline-full-r0    --offline             --full-offline       pregate

  (production --offline = DiarModel::diarize_offline: full attention, peak
  normalize, NO BirthGate -> raw probs == our pregate on that path.)

Gates (hard, M1 frame tiers — both sides q8, diffs come from fp32
accumulation order only):
  frame_agreement >= 0.999, max_abs <= 0.05, mean_abs <= 0.005
Reported as data (NOT gated): row-count delta (production tail semantics vs
python-NeMo phantom padding differ by construction), per-row diff percentiles
(for tightening). Determinism (one case re-run) is hard: bit-identical.

A gate FAIL is data (verdict json still written); only infra errors raise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

REPORT: dict[str, object] = {
    "schema_version": 1,
    "job": "m2_stage3_k6_q8_fixtures",
}

OUT = Path("/kaggle/working")
WORK = Path("/tmp/m2-stage3-k6")

GATE_MAX_ABS = 0.05
GATE_MEAN_ABS = 0.005
GATE_FRAME_AGREEMENT = 0.999

# manifest case field -> (extra runner argv, which output file to compare)
CASE_PLAN = {
    "full_offline": (["--full-offline"], "pregate"),
    "offline_preset": (["--run", "--offline"], "postgate"),
    "streaming": (["--run"], "postgate"),
}


def classify(manifest: dict) -> tuple[list[str], str]:
    if manifest["case"].get("offline"):
        return CASE_PLAN["full_offline"]
    if manifest["case"].get("preset"):
        return CASE_PLAN["offline_preset"]
    return CASE_PLAN["streaming"]


def metrics(ref: np.ndarray, got: np.ndarray) -> dict:
    r = np.asarray(ref, dtype=np.float64)
    g = np.asarray(got, dtype=np.float64)
    n = min(r.shape[0], g.shape[0])
    r, g = r[:n], g[:n]
    d = np.abs(r - g)
    ra = (r > 0.5).astype(int)
    ga = (g > 0.5).astype(int)
    pct = np.percentile(d.max(axis=1), [50, 95, 99]) if n else [0.0] * 3
    return {
        "rows_compared": int(n),
        "max_abs": float(d.max()) if n else 0.0,
        "mean_abs": float(d.mean()) if n else 0.0,
        "frame_agreement": float((ra == ga).mean()) if n else 1.0,
        "row_max_p50": float(pct[0]),
        "row_max_p95": float(pct[1]),
        "row_max_p99": float(pct[2]),
        "worst_row": int(d.max(axis=1).argmax()) if n else -1,
    }


def decode_wav(path: Path) -> np.ndarray:
    """16 kHz mono PCM -> f32 in [-1, 1), matching dr_wav's int16 convention."""
    with wave.open(str(path), "rb") as w_:
        nch = w_.getnchannels()
        sampwidth = w_.getsampwidth()
        rate = w_.getframerate()
        nframes = w_.getnframes()
        raw = w_.readframes(nframes)
    if rate != 16000:
        raise RuntimeError(f"{path.name}: expected 16 kHz, got {rate}")
    if nch != 1:
        raise RuntimeError(f"{path.name}: expected mono, got {nch}ch")
    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        pcm = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise RuntimeError(f"{path.name}: unsupported sampwidth {sampwidth}")
    return np.ascontiguousarray(pcm)


def find_audio(basename: str, roots: list[Path]) -> Path:
    for root in roots:
        hits = sorted(root.rglob(basename))
        if hits:
            return hits[0]
    raise RuntimeError(f"audio {basename} not found under {roots}")


def run_runner(runner: Path, args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([str(runner), *args], capture_output=True, text=True,
                          timeout=timeout)


def run_case(label: str, fdir: Path, runner: Path, weights: Path,
             roots: list[Path], work: Path) -> dict:
    manifest = json.loads((fdir / "manifest.json").read_text())
    n_spk = int(manifest["observation"]["n_spk"])
    ref = np.fromfile(str(fdir / "probs.f32"), dtype="<f4").reshape(-1, n_spk)
    argv_extra, which = classify(manifest)
    audio_src = find_audio(Path(manifest["case"]["audio"]).name, roots)
    pcm = decode_wav(audio_src)
    af32 = work / f"{label}.f32"
    af32.write_bytes(pcm.tobytes())

    runs: list[np.ndarray] = []
    metas: list[dict] = []
    n_feed_passes = 2 if label == DETERMINISM_CASE else 1
    for i in range(n_feed_passes):
        prefix = str(work / f"{label}.run{i}")
        p = run_runner(runner, [*argv_extra, "--weights", str(weights),
                                "--audio", str(af32), "--out", prefix])
        if p.returncode != 0:
            return {"error": f"runner rc={p.returncode}: "
                             f"{(p.stderr or p.stdout)[-400:]}"}
        metas.append(json.loads(Path(prefix + ".meta.json").read_text()))
        runs.append(np.fromfile(f"{prefix}.{which}.f32", dtype="<f4")
                    .reshape(-1, n_spk))
    ours = runs[0]

    row_delta = int(ours.shape[0] - ref.shape[0])
    m = metrics(ref, ours)
    m.update({
        "rows_ours": int(ours.shape[0]),
        "rows_ref": int(ref.shape[0]),
        "rows_delta": row_delta,
        "compare_file": which,
        "runner_args": argv_extra,
    })
    if row_delta > 0:
        # tail semantics data: what do OUR extra rows say?
        tail = ours[ref.shape[0]:]
        m["ours_tail_rows_max"] = float(np.abs(tail).max()) if tail.size else 0.0
    elif row_delta < 0:
        tail = ref[ours.shape[0]:]
        m["ref_tail_rows_max"] = float(np.abs(tail).max()) if tail.size else 0.0
    if n_feed_passes == 2:
        m["determinism_bit_identical"] = bool(np.array_equal(runs[0], runs[1]))
    return m


DETERMINISM_CASE = "v12-short-streaming-r0"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--fixtures-dir", required=True)
    ap.add_argument("--cases", default="")
    ap.add_argument("--audio-roots", default="/kaggle/input,/kaggle/working")
    ap.add_argument("--out", default=str(OUT / "m2_stage3_k6_verdict.json"))
    args = ap.parse_args()

    runner = Path(args.runner)
    gguf = Path(args.gguf)
    fixtures = Path(args.fixtures_dir)
    REPORT["runner"] = str(runner)
    REPORT["gguf"] = str(gguf)
    REPORT["fixtures_dir"] = str(fixtures)

    # ---- G0 infra ----
    p = run_runner(runner, ["--selftest"])
    REPORT["G0_selftest"] = {"rc": p.returncode, "tail": (p.stdout or p.stderr)[-400:]}
    if p.returncode != 0:
        REPORT["verdict"] = "infra-fail-selftest"
        _emit(args.out)
        return 0
    probe = run_runner(runner, ["--probe-weights", "--weights", str(gguf)])
    REPORT["G0_gguf_probe"] = {"rc": probe.returncode,
                               "tail": (probe.stdout or probe.stderr)[-400:]}
    if probe.returncode != 0:
        REPORT["verdict"] = "infra-fail-gguf-probe"
        _emit(args.out)
        return 0

    roots = [Path(r) for r in args.audio_roots.split(",") if r]
    WORK.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(d for d in fixtures.iterdir()
                       if d.is_dir() and (d / "probs.f32").exists())
    if args.cases:
        keep = set(args.cases.split(","))
        case_dirs = [d for d in case_dirs if d.name in keep]
    if not case_dirs:
        REPORT["verdict"] = "infra-no-fixtures"
        _emit(args.out)
        return 0

    per_case: dict[str, dict] = {}
    t0 = time.time()
    for d in case_dirs:
        per_case[d.name] = run_case(d.name, d, runner, gguf, roots, WORK)
    per_case = dict(sorted(per_case.items()))
    REPORT["per_case"] = per_case
    REPORT["seconds"] = round(time.time() - t0, 1)

    reasons: list[str] = []
    notes: list[str] = []
    for label, r in per_case.items():
        if "error" in r:
            reasons.append(f"{label}: {r['error']}")
            continue
        if "determinism_bit_identical" in r and not r["determinism_bit_identical"]:
            reasons.append(f"{label}: determinism not bit-identical")
        if r["max_abs"] > GATE_MAX_ABS:
            reasons.append(f"{label}: max_abs={r['max_abs']:.4g}")
        if r["mean_abs"] > GATE_MEAN_ABS:
            reasons.append(f"{label}: mean_abs={r['mean_abs']:.4g}")
        if r["frame_agreement"] < GATE_FRAME_AGREEMENT:
            reasons.append(f"{label}: frame_agreement={r['frame_agreement']:.6f}")
        if r["rows_delta"] != 0:
            notes.append(f"{label}: rows ours={r['rows_ours']} ref={r['rows_ref']} "
                         f"(delta {r['rows_delta']:+d}, tail data attached)")

    REPORT["gates"] = {"max_abs": GATE_MAX_ABS, "mean_abs": GATE_MEAN_ABS,
                       "frame_agreement": GATE_FRAME_AGREEMENT}
    REPORT["notes"] = notes
    REPORT["verdict"] = "k6-green" if not reasons else "k6-red"
    REPORT["reasons"] = reasons
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _emit(args.out)
    print(json.dumps({k: v for k, v in REPORT.items() if k != "per_case"},
                     indent=1)[:3000])
    return 0


def _emit(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
