"""parity framework tests — synthetic fixtures only, no kernel data needed.

Covers: manifest schema enforcement, payload integrity enforcement, gate
verdicts (exact + tolerance), and gate sensitivity (perturbed inputs must
flip the verdict — an always-green gate is worthless).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from parity import (
    FixtureIntegrityError,
    ManifestError,
    frame_probs_gate,
    load_case,
)
from parity.loader import FrameProbsTensor


def make_fixture(directory: Path, *, values: list[float], n_frames: int, n_spk: int,
                 mutate: dict | None = None) -> Path:
    """Write a minimal valid fixture; ``mutate`` patches the manifest/payload."""
    payload = struct.pack("<qi", n_frames, n_spk) + struct.pack(
        f"<{len(values)}f", *values)
    (directory / "probs.f32").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "name": "synthetic",
        "case": {"audio": "synthetic-tone", "preset": None, "offline": False},
        "source": {
            "kernel": "diar-gpu-engine-sortformer-matrix",
            "kernel_version": 9,
            "src_commit": "659fc15",
            "upstream_commit": "a5b6953",
            "captured_at": "2026-09-15T00:00:00Z",
            "gpu": "Tesla P100",
            "repeats": 1,
        },
        "observation": {
            "frame_grid_ms": 80, "n_frames": n_frames, "n_spk": n_spk,
            "deterministic": True, "hash_groups": {},
        },
        "tensors": [{
            "name": "probs", "layer": "frame_probs", "dtype": "f32",
            "shape": [n_frames, n_spk], "nbytes": len(values) * 4,
            "sha256": hashlib.sha256(payload).hexdigest(), "path": "probs.f32",
        }],
    }
    if mutate:
        manifest = mutate(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return directory


@pytest.fixture()
def fixture_values() -> list[float]:
    # 3 frames x 2 speakers, clearly separated argmax per frame.
    return [0.9, 0.1, 0.2, 0.8, 0.7, 0.3]


def tensor_of(values: list[float], n_frames: int = 3, n_spk: int = 2) -> FrameProbsTensor:
    return FrameProbsTensor(n_frames=n_frames, n_spk=n_spk, values=tuple(values))


class TestManifestSchema:
    def test_valid_fixture_loads(self, tmp_path, fixture_values):
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2)
        manifest, tensors = load_case(tmp_path)
        assert manifest.name == "synthetic"
        assert tensors["probs"].n_frames == 3
        assert tensors["probs"].n_spk == 2

    def test_missing_schema_version_rejected(self, tmp_path, fixture_values):
        def drop_version(m):
            del m["schema_version"]
            return m
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2,
                     mutate=drop_version)
        with pytest.raises(ManifestError, match="schema_version"):
            load_case(tmp_path)

    def test_future_schema_rejected(self, tmp_path, fixture_values):
        def bump(m):
            m["schema_version"] = 99
            return m
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2,
                     mutate=bump)
        with pytest.raises(ManifestError, match="schema_version"):
            load_case(tmp_path)

    def test_blank_provenance_rejected(self, tmp_path, fixture_values):
        def blank(m):
            m["source"]["src_commit"] = "   "
            return m
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2,
                     mutate=blank)
        with pytest.raises(ManifestError, match="src_commit"):
            load_case(tmp_path)

    def test_unknown_layer_rejected(self, tmp_path, fixture_values):
        def bad_layer(m):
            m["tensors"][0]["layer"] = "hidden_42"
            return m
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2,
                     mutate=bad_layer)
        with pytest.raises(ManifestError, match="layer"):
            load_case(tmp_path)

    def test_nbytes_shape_mismatch_rejected(self, tmp_path, fixture_values):
        def bad_nbytes(m):
            m["tensors"][0]["nbytes"] = 12
            return m
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2,
                     mutate=bad_nbytes)
        with pytest.raises(ManifestError, match="nbytes"):
            load_case(tmp_path)


class TestPayloadIntegrity:
    def test_bit_flip_detected(self, tmp_path, fixture_values):
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2)
        raw = bytearray((tmp_path / "probs.f32").read_bytes())
        raw[-1] ^= 0x01
        (tmp_path / "probs.f32").write_bytes(bytes(raw))
        with pytest.raises(FixtureIntegrityError, match="sha256"):
            load_case(tmp_path)

    def test_truncated_payload_detected(self, tmp_path, fixture_values):
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2)
        raw = (tmp_path / "probs.f32").read_bytes()
        (tmp_path / "probs.f32").write_bytes(raw[:-4])
        with pytest.raises(FixtureIntegrityError):
            load_case(tmp_path)

    def test_header_observation_disagreement_detected(self, tmp_path, fixture_values):
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["observation"]["n_frames"] = 4
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(FixtureIntegrityError, match="disagrees"):
            load_case(tmp_path)

    def test_path_escape_rejected(self, tmp_path, fixture_values):
        def escape(m):
            m["tensors"][0]["path"] = "../probs.f32"
            return m
        make_fixture(tmp_path, values=fixture_values, n_frames=3, n_spk=2,
                     mutate=escape)
        with pytest.raises(FixtureIntegrityError, match="escapes"):
            load_case(tmp_path)


class TestGateVerdicts:
    def test_exact_pass_and_fail(self, fixture_values):
        fx = tensor_of(fixture_values)
        assert frame_probs_gate(list(fixture_values), fx, mode="exact")["verdict"] == "PASS"
        perturbed = list(fixture_values)
        perturbed[0] += 1e-7
        assert frame_probs_gate(perturbed, fx, mode="exact")["verdict"] == "FAIL"

    def test_size_mismatch_fails_with_reason(self, fixture_values):
        fx = tensor_of(fixture_values)
        out = frame_probs_gate(fixture_values[:-1], fx, mode="tolerance")
        assert out["verdict"] == "FAIL"
        assert "size mismatch" in out["reason"]

    def test_tolerance_passes_small_perturbation(self, fixture_values):
        fx = tensor_of(fixture_values)
        perturbed = [v + 1e-6 for v in fixture_values]
        out = frame_probs_gate(perturbed, fx)
        assert out["verdict"] == "PASS"
        assert out["max_abs"] < 1e-5

    def test_tolerance_catches_argmax_flip(self, fixture_values):
        # Frame 0 argmax flips 0.9/0.1 -> 0.1/0.9: one frame disagreement
        # already violates frame_agreement_min on a 3-frame fixture.
        fx = tensor_of(fixture_values)
        flipped = list(fixture_values)
        flipped[0], flipped[1] = flipped[1], flipped[0]
        out = frame_probs_gate(flipped, fx)
        assert out["verdict"] == "FAIL"
        assert out["frame_agreement"] == pytest.approx(2 / 3)

    def test_tolerance_catches_large_offset(self, fixture_values):
        fx = tensor_of(fixture_values)
        shifted = [v + 0.2 for v in fixture_values]
        out = frame_probs_gate(shifted, fx)
        assert out["verdict"] == "FAIL"
        assert any("max_abs" in f for f in out["failures"])

    def test_explicit_tolerance_override(self, fixture_values):
        fx = tensor_of(fixture_values)
        shifted = [v + 0.2 for v in fixture_values]
        loose = {"frame_agreement_min": 0.5, "max_abs_max": 0.5, "mean_abs_max": 0.5}
        assert frame_probs_gate(shifted, fx, tolerance=loose)["verdict"] == "PASS"

    def test_gate_is_sensitivity_checked_by_construction(self, fixture_values):
        # A gate that always passes is worthless: prove this suite catches
        # both an argmax flip AND a value shift with the default profile.
        fx = tensor_of(fixture_values)
        for bad in ([fixture_values[1], fixture_values[0]] + fixture_values[2:],
                    [v + 0.9 for v in fixture_values]):
            assert frame_probs_gate(bad, fx)["verdict"] == "FAIL"
