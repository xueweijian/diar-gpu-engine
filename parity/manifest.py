"""Parity fixture manifest (schema v1).

A parity fixture pins one upstream-binary observation — the ground truth an
M2 engine layer must reproduce. Truth discipline: fixtures are generated
exclusively from kernel output by ``scripts/fill_parity_fixture.py`` (to be
added with the v9 verdict); the manifest records the kernel version and
source commits that produced it, and tensor payloads are sha256-pinned, so
hand-edited truth is detectable. Consumers are the parity gates in
``parity/gate.py``.

Layout on disk (one directory per fixture under parity/fixtures/<name>/):

    manifest.json          — this schema
    probs.f32              — binary tensors referenced by the manifest

Schema v1 fields:

    schema_version : 1
    name           : str, stable fixture id ("v9-short-streaming-r0")
    case           : {audio: str, preset: str|None, offline: bool,
                      geometry: {chunk,fifo,spkcache,...}|None}
    source         : {kernel: str, kernel_version: int, src_commit: str,
                      upstream_commit: str, captured_at: str (RFC3339 UTC),
                      gpu: str, repeats: int}
    observation    : {frame_grid_ms: int, n_frames: int, n_spk: int,
                      deterministic: bool, hash_groups: {...} (v9 verdict)}
    tensors        : [{name: "probs", layer: "frame_probs", dtype: "f32",
                       shape: [n_frames, n_spk], nbytes: int, sha256: str,
                       path: "probs.f32"}]
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Layers a v1 manifest may declare and the gate names that consume them.
KNOWN_LAYERS = ("frame_probs",)

REQUIRED_TOP = ("schema_version", "name", "case", "source", "observation", "tensors")
REQUIRED_SOURCE = ("kernel", "kernel_version", "src_commit", "upstream_commit", "captured_at")
REQUIRED_TENSOR = ("name", "layer", "dtype", "shape", "nbytes", "sha256", "path")


class ManifestError(Exception):
    """Raised when a manifest violates the schema or its pins."""


@dataclasses.dataclass(frozen=True)
class TensorPin:
    name: str
    layer: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    sha256: str
    path: str

    @property
    def n_elements(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n


@dataclasses.dataclass(frozen=True)
class Manifest:
    name: str
    case: dict[str, Any]
    source: dict[str, Any]
    observation: dict[str, Any]
    tensors: tuple[TensorPin, ...]
    raw: dict[str, Any]

    def tensor(self, name: str) -> TensorPin:
        for pin in self.tensors:
            if pin.name == name:
                return pin
        raise ManifestError(f"manifest {self.name!r} has no tensor {name!r}")


def load_manifest(manifest_path: Path) -> Manifest:
    """Parse and structurally validate a manifest.json (no payload I/O)."""
    try:
        raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path}: not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ManifestError(f"{manifest_path}: manifest must be a JSON object")
    for key in REQUIRED_TOP:
        if key not in raw:
            raise ManifestError(f"{manifest_path}: missing required key {key!r}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"{manifest_path}: schema_version {raw['schema_version']!r} != {SCHEMA_VERSION}")

    for key in REQUIRED_SOURCE:
        if key not in raw["source"]:
            raise ManifestError(f"{manifest_path}: source missing {key!r}")
    # A fixture without provenance is unverifiable truth: require real values.
    for key in ("kernel", "src_commit", "upstream_commit", "captured_at"):
        value = raw["source"][key]
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"{manifest_path}: source.{key} must be a non-empty string")
    if not isinstance(raw["source"]["kernel_version"], int):
        raise ManifestError(f"{manifest_path}: source.kernel_version must be an int")

    tensors = raw["tensors"]
    if not isinstance(tensors, list) or not tensors:
        raise ManifestError(f"{manifest_path}: tensors must be a non-empty list")
    pins = []
    for i, entry in enumerate(tensors):
        for key in REQUIRED_TENSOR:
            if key not in entry:
                raise ManifestError(f"{manifest_path}: tensors[{i}] missing {key!r}")
        if entry["layer"] not in KNOWN_LAYERS:
            raise ManifestError(
                f"{manifest_path}: tensors[{i}].layer {entry['layer']!r} not in {KNOWN_LAYERS}")
        if entry["dtype"] != "f32":
            raise ManifestError(
                f"{manifest_path}: tensors[{i}].dtype {entry['dtype']!r}: only f32 in v1")
        shape = tuple(entry["shape"])
        if not shape or not all(isinstance(d, int) and d > 0 for d in shape):
            raise ManifestError(f"{manifest_path}: tensors[{i}].shape must be positive ints")
        n_elems = 1
        for d in shape:
            n_elems *= d
        # nbytes counts the LOGICAL tensor payload (n_elems * 4). frame_probs
        # files additionally carry the 12-byte probdump header on disk; the
        # loader knows the wire format and validates file size accordingly.
        if entry["nbytes"] != n_elems * 4:
            raise ManifestError(
                f"{manifest_path}: tensors[{i}].nbytes {entry['nbytes']} != "
                f"{n_elems * 4} for shape {list(shape)} f32")
        if len(entry["sha256"]) != 64:
            raise ManifestError(f"{manifest_path}: tensors[{i}].sha256 must be hex sha256")
        pins.append(TensorPin(name=entry["name"], layer=entry["layer"], dtype="f32",
                              shape=shape, nbytes=entry["nbytes"],
                              sha256=entry["sha256"], path=entry["path"]))

    obs = raw["observation"]
    if not isinstance(obs.get("frame_grid_ms"), int) or obs["frame_grid_ms"] <= 0:
        raise ManifestError(f"{manifest_path}: observation.frame_grid_ms must be positive")
    if not isinstance(obs.get("n_frames"), int) or obs["n_frames"] <= 0:
        raise ManifestError(f"{manifest_path}: observation.n_frames must be positive")
    if not isinstance(obs.get("n_spk"), int) or obs["n_spk"] <= 0:
        raise ManifestError(f"{manifest_path}: observation.n_spk must be positive")

    return Manifest(name=raw["name"], case=raw["case"], source=raw["source"],
                    observation=obs, tensors=tuple(pins), raw=raw)
