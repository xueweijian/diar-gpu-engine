"""Harvester CLI tests: K5-A and K6 verdict summarizers.

Both harvesters are the ONLY reader of their verdicts (the kernels run on
Kaggle; results are harvested locally), so their rendering is pinned here
against synthetic verdicts carrying the exact schema the gates emit.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def test_harvest_k6_green(tmp_path):
    verdict = {
        "schema_version": 1,
        "job": "m2_stage3_k6_q8_fixtures",
        "G0_selftest": {"rc": 0},
        "G0_gguf_probe": {"rc": 0},
        "per_case": {
            "v12-mid-offline-full-r0": {
                "rows_compared": 4467, "max_abs": 0.0123, "mean_abs": 0.0004,
                "frame_agreement": 0.9995, "row_max_p50": 0.004,
                "row_max_p95": 0.02, "row_max_p99": 0.03, "worst_row": 110,
                "rows_ours": 4466, "rows_ref": 4467, "rows_delta": -1,
                "compare_file": "pregate", "runner_args": ["--full-offline"],
                "ref_tail_rows_max": 0.0,
            },
            "v12-short-streaming-r0": {
                "rows_compared": 711, "max_abs": 0.02, "mean_abs": 0.001,
                "frame_agreement": 1.0, "row_max_p50": 0.01, "row_max_p95": 0.02,
                "row_max_p99": 0.03, "worst_row": 5, "rows_ours": 711,
                "rows_ref": 711, "rows_delta": 0, "compare_file": "postgate",
                "runner_args": ["--run"], "determinism_bit_identical": True,
            },
        },
        "seconds": 123.4,
        "gates": {"max_abs": 0.05, "mean_abs": 0.005, "frame_agreement": 0.999},
        "notes": ["v12-mid-offline-full-r0: rows ours=4466 ref=4467 (delta -1, tail data attached)"],
        "verdict": "k6-green",
        "reasons": [],
        "finished_utc": "2026-09-18T09:00:00Z",
    }
    p = tmp_path / "k6_verdict.json"
    p.write_text(json.dumps(verdict))
    out = subprocess.run([sys.executable, str(SCRIPTS / "harvest_k6.py"), str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "k6-green" in out
    assert "v12-short-streaming-r0" in out and "v12-mid-offline-full-r0" in out
    assert "pregate" in out and "postgate" in out
    assert "yes" in out  # determinism rendered
    assert "G0 selftest rc=0" in out


def test_harvest_k6_red_with_error_case(tmp_path):
    verdict = {
        "G0_selftest": {"rc": 0}, "G0_gguf_probe": {"rc": 0},
        "per_case": {
            "v13-mid-streaming-r0": {"error": "runner rc=1: boom"},
            "v13-mid-offline-preset-r0": {
                "rows_compared": 4467, "max_abs": 0.5, "mean_abs": 0.05,
                "frame_agreement": 0.9, "rows_ours": 4466, "rows_ref": 4467,
                "rows_delta": -1, "compare_file": "postgate",
                "runner_args": ["--run", "--offline"],
            },
        },
        "verdict": "k6-red",
        "reasons": ["v13-mid-streaming-r0: runner rc=1: boom",
                    "v13-mid-offline-preset-r0: max_abs=0.5"],
        "finished_utc": "2026-09-18T09:00:00Z",
    }
    p = tmp_path / "v.json"
    p.write_text(json.dumps(verdict))
    out = subprocess.run([sys.executable, str(SCRIPTS / "harvest_k6.py"), str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "k6-red" in out
    assert "ERROR: runner rc=1" in out
    assert "max_abs=0.5" in out  # reasons section


def test_harvest_k5a_synthetic(tmp_path):
    """Schema smoke: render a verdict shaped exactly like the K5-A gate's."""
    verdict = {
        "schema_version": 1, "job": "m2_stage3_k5a_free_running",
        "G0_selftest": {"rc": 0, "tail": "k5-runner selftest ok"},
        "per_audio": {
            "short": {
                "geometry": [20, 80, 160, 80, 160, 8],
                "n_chunks_npz": 36, "n_chunks_ours": 36,
                "G5_length": {"ours": 711, "nemo": 712, "expected_ours": 711,
                              "diff": 0, "phantom_rows": 1,
                              "phantom_all_zero": True},
                "G1_chunk0": {"max_abs": 1e-7, "mean_abs": 2e-8,
                              "frame_agreement": 1.0},
                "G2_timeline": {
                    "overall": {"max_abs": 1e-4, "mean_abs": 1e-5,
                                "frame_agreement": 1.0},
                    "argmax_band_agreement": 1.0,
                    "per_chunk_max_abs": [1e-4, 2e-4],
                    "worst_chunk": 1, "pre_first_compress_max": 1e-4,
                    "post_first_compress_max": None,
                    "first_compress_chunk": None,
                },
                "G3_aosc": {"geometry_ok": True, "spk_max_abs": 1e-5,
                            "fifo_max_abs": 1e-5, "mean_sil_max_abs": None,
                            "per_chunk": []},
                "G4_compress": {"ours_sim": [], "nemo": [], "match": True},
                "G6_determinism": {"n_frames_equal": True,
                                   "bit_identical": True},
                "seconds": 42.0,
            },
        },
        "gates": {"G1_chunk0_max_abs": 1e-05, "G2_safety_max_abs": 0.05,
                  "G2_frame_agreement": 0.999, "G3_safety_max_abs": 0.05},
        "verdict": "k5a-green", "reasons": [],
        "finished_utc": "2026-09-18T09:00:00Z",
    }
    p = tmp_path / "k5a_verdict.json"
    p.write_text(json.dumps(verdict))
    out = subprocess.run([sys.executable, str(SCRIPTS / "harvest_k5a.py"), str(p)],
                         capture_output=True, text=True, check=True).stdout
    assert "k5a-green" in out
    assert "G1 chunk0" in out and "G2 overall" in out
    assert "G5 rows: ours=711" in out
    assert "G4 compress: match=True" in out
    assert "G6 determinism: bit_identical=True" in out
