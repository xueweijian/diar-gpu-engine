"""M2 Stage 0 spike: capture NeMo fp32 reference tensors (or fail loudly).

Goal per docs/M2-NEURAL-CORE-PLAN.md §2: run upstream
scripts/asr/dump_sortformer_reference.py (vendored below as
dump_m2_reference, with per-block hooks) against the SAME audios and
geometry the matrix uses, and answer S0-a/b/c.

Spike discipline: this kernel MUST end in a verdict, not a partial dump.
- S0-a: HF repo file list contains a .nemo checkpoint -> else verdict
  "no-nemo-checkpoint" (proceed to ggml-probdump fallback).
- S0-b: nemo_toolkit[asr] installs and imports on this image -> else
  verdict "nemo-not-installable" (same fallback).
- S0-c: dump completes and total_preds sanity-checks against the ggml
  binary on one short audio within tolerance -> verdict "nemo-ok"
  (Stage 1/2 unblocked) or "nemo-diverged" (investigate, do NOT silently
  promote the dump to truth).

Everything is pure diarization. Artifacts are small (.npz per audio +
spike_verdict.json); build trees, wheels and checkpoints stay in /tmp.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def locate_harness() -> Path:
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    listing: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        try:
            for entry in sorted(root.iterdir()):
                listing.append(str(entry))
        except OSError as exc:
            listing.append(f"{root}: {exc}")
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("diar_harness.py"):
            return candidate.parent
    raise RuntimeError(
        "diar_harness.py not found. Kaggle input listing: " + " | ".join(listing)
    )


HARNESS_DIR = locate_harness()
sys.path.insert(0, str(HARNESS_DIR))

import diar_harness as h  # noqa: E402

OUT = Path("/kaggle/working")
TMP = Path("/tmp/m2-refdump")
FIXTURE_ROOT = Path("/kaggle/input")
HF_REPO = "nvidia/diar_streaming_sortformer_4spk-v2"
# Matrix geometry (streaming preset, aosc_state.h): chunk 20 / fifo 80 /
# spkcache 160 / update-period 80 encoder frames; lc=rc=0.
GEOMETRY = {
    "chunk": 20, "lc": 0, "rc": 0,
    "fifo": 80, "spkcache": 160, "update_period": 80,
}
# Same audios the parity fixtures pin (short + mid only; spike, not sweep).
WANT_AUDIOS = {
    "short": "diar-smoke-audio/0-four-speakers-zh.wav",
    "mid": "diar-real-audio-5/video2_audio.wav",
}

# Self-containment: Kaggle script kernels upload ONLY code_file, so the
# vendored sibling dump_m2_reference.py never arrives at /kaggle/src. The
# embed step (kaggle/m2_refdump/embed_dump.py, run locally before push)
# inlines the file below; at runtime we prefer a sibling file if present
# and otherwise materialize this embedded copy. Do not hand-edit.
EMBEDDED_DUMP_M2_REFERENCE = '#!/usr/bin/env python3\n# M2 fork of upstream dump_sortformer_reference.py (NeMo-Speech.cpp a5b6953).\n# ONLY delta vs upstream: _capture_block_outputs helper + per-block hook\n# capture (conformer_block_NN / transformer_block_NN) on deep chunks.\n# The NeMo forward path is untouched (hooks observe only).\n# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"""Dump NeMo streaming Sortformer reference tensors for ggml parity tests.\n\nReplays the sync streaming loop (SortformerEncLabelModel.forward_streaming with\nasync_streaming=False) chunk by chunk, capturing every intermediate the C++\nport needs to check against:\n\n  per chunk: mel window, pre_encode embeddings, concat lengths, full preds,\n             chunk preds, post-update AOSC state (spkcache/preds/fifo/\n             mean_sil_emb/n_sil_frames), compression flag.\n  stream:    full mel sequence, final concatenated chunk preds.\n\nDefault geometry uses the runtime streaming preset. Deep intermediates\n(fc-encoder output, transformer output) are stored for the first\n--deep-chunks chunks and for every compression chunk.\n\nUsage:\n    python dump_sortformer_reference.py <ckpt.nemo> <audio.wav> <out.npz> \\\n        [--device cpu] [--max-sec 40] [--chunk 8 --lc 0 --rc 8 \\\n         --fifo 80 --spkcache 160 --update-period 80]\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport math\nfrom pathlib import Path\n\nimport numpy as np\nimport soundfile as sf\nimport torch\n\n\ndef _capture_block_outputs(model):\n    """Register forward hooks capturing per-block outputs (deep chunks).\n\n    Hooks only observe: the NeMo forward path is untouched. Returns\n    (handles, conf_outs, trans_outs); the caller removes handles right\n    after the forward.\n\n    Layer identity is by OUTPUT SHAPE (v3 ouroboros rule, Stage 0\n    finding): NeMo-side class names are not trusted. A block output of\n    (..., 512) is a conformer block, (..., 192) a transformer block;\n    anything else is recorded as unknown and fails the dump loudly\n    (never silently dropped). Hooks fire in forward order, so both\n    lists are in stack order.\n    """\n    conf_outs: list = []\n    trans_outs: list = []\n    unknown_shapes: list = []\n    handles = []\n\n    def _mk(store):\n        def _fn(module, args, output):\n            out = output[0] if isinstance(output, (tuple, list)) else output\n            arr = out.detach().cpu().numpy()\n            if arr.shape[-1] == 512:\n                conf_outs.append(arr)\n            elif arr.shape[-1] == 192:\n                trans_outs.append(arr)\n            else:\n                unknown_shapes.append(arr.shape)\n                store.append(arr)\n        return _fn\n\n    for mod in model.modules():\n        cls = type(mod).__name__\n        if cls in ("ConformerLayer", "TransformerEncoderBlock",\n                   "TransformerEncoderLayer", "TransformerLayer"):\n            handles.append(mod.register_forward_hook(_mk(conf_outs\n                                                          if cls == "ConformerLayer"\n                                                          else trans_outs)))\n    # Ouroboros: verify the class-name routing against shape routing on\n    # the first deep chunk is done by the caller via counts (17/18).\n    _capture_block_outputs.unknown_shapes = unknown_shapes\n    return handles, conf_outs, trans_outs\n\n\ndef main() -> int:\n    ap = argparse.ArgumentParser()\n    ap.add_argument("ckpt", help=".nemo checkpoint path")\n    ap.add_argument("audio", help="16 kHz mono wav")\n    ap.add_argument("out", help="output .npz path")\n    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))\n    ap.add_argument("--max-sec", type=float, default=40.0)\n    # Defaults = the runtime streaming preset (DiarGeometry, aosc_state.h) for\n    # convenience only: the chosen geometry is recorded in the dump and the\n    # parity test reads it from there, so any values make a valid reference.\n    ap.add_argument("--chunk", type=int, default=20, help="chunk_len in 80ms encoder frames")\n    ap.add_argument("--lc", type=int, default=0, help="chunk_left_context in encoder frames")\n    ap.add_argument("--rc", type=int, default=0, help="chunk_right_context in encoder frames")\n    ap.add_argument("--fifo", type=int, default=80)\n    ap.add_argument("--spkcache", type=int, default=160)\n    ap.add_argument("--update-period", type=int, default=80)\n    ap.add_argument(\n        "--deep-chunks",\n        type=int,\n        default=3,\n        help="store fc-encoder/transformer outputs for the first N chunks",\n    )\n    args = ap.parse_args()\n\n    from nemo.collections.asr.models import SortformerEncLabelModel\n\n    device = torch.device(args.device)\n    model = SortformerEncLabelModel.restore_from(restore_path=args.ckpt, map_location=device)\n    model.eval()\n    model.to(device)\n\n    sm = model.sortformer_modules\n    sm.chunk_len = args.chunk\n    sm.chunk_left_context = args.lc\n    sm.chunk_right_context = args.rc\n    sm.fifo_len = args.fifo\n    sm.spkcache_len = args.spkcache\n    sm.spkcache_update_period = args.update_period\n    sm._check_streaming_parameters()\n\n    # Determinism: no dither (riva sets 0 at inference too).\n    model.preprocessor.featurizer.dither = 0.0\n\n    audio, sr = sf.read(args.audio, dtype="float32")\n    if audio.ndim > 1:\n        audio = audio[:, 0]\n    assert sr == 16000, f"expected 16 kHz audio, got {sr}"\n    max_samples = int(args.max_sec * sr)\n    audio = audio[:max_samples]\n\n    sig = torch.from_numpy(audio).unsqueeze(0).to(device)\n    sig_len = torch.tensor([sig.shape[1]], device=device)\n\n    with torch.inference_mode():\n        # Streaming mode: no max-normalization (SortformerEncLabelModel.\n        # process_signal only rescales when streaming_mode is off).\n        mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)\n\n        out: dict[str, np.ndarray] = {\n            "audio": audio,\n            "mel": mel[0].cpu().numpy(),  # (128, T)\n            "mel_len": mel_len.cpu().numpy(),\n            "geometry": np.array(\n                [args.chunk, args.lc, args.rc, args.fifo, args.spkcache, args.update_period],\n                dtype=np.int64,\n            ),\n        }\n\n        state = sm.init_streaming_state(batch_size=1, async_streaming=False, device=device)\n        total_preds = torch.zeros((1, 0, sm.n_spk), device=device)\n        offset = torch.zeros((1,), dtype=torch.long, device=device)\n        sub = model.encoder.subsampling_factor\n\n        n_chunks = 0\n        compression_chunks = []\n        for idx, chunk_feat, feat_lengths, left_off, right_off in sm.streaming_feat_loader(\n            feat_seq=mel, feat_seq_length=mel_len, feat_seq_offset=offset\n        ):\n            p = f"chunk{idx:03d}/"\n            out[p + "mel_window"] = chunk_feat[0].cpu().numpy()  # (T_mel, 128)\n            out[p + "feat_length"] = feat_lengths.cpu().numpy()\n            out[p + "offsets"] = np.array([left_off, right_off], dtype=np.int64)\n\n            pre_embs, pre_lens = model.encoder.pre_encode(x=chunk_feat, lengths=feat_lengths)\n            out[p + "pre_encode"] = pre_embs[0].cpu().numpy()  # (T_enc, 512)\n\n            spk_len_before = state.spkcache.shape[1]\n            fifo_len_before = state.fifo.shape[1]\n            out[p + "state_lens_before"] = np.array(\n                [spk_len_before, fifo_len_before, int(pre_lens[0])], dtype=np.int64\n            )\n\n            concat_embs = sm.concat_embs(\n                [state.spkcache, state.fifo, pre_embs], dim=1, device=device\n            )\n            concat_lens = spk_len_before + fifo_len_before + pre_lens\n\n            handles, conf_outs, trans_outs = (\n                _capture_block_outputs(model)\n                if idx < args.deep_chunks else ([], [], []))\n            fc_embs, fc_lens = model.frontend_encoder(\n                processed_signal=concat_embs,\n                processed_signal_length=concat_lens,\n                bypass_pre_encode=True,\n            )\n            for hd in handles:\n                hd.remove()\n            preds = model.forward_infer(emb_seq=fc_embs, emb_seq_length=fc_lens)\n            preds = sm.apply_mask_to_preds(preds, fc_lens)\n            out[p + "preds_full"] = preds[0].cpu().numpy()  # (L1+L2+L3, 4)\n\n            deep = idx < args.deep_chunks\n            if deep:\n                # frontend_encoder applies encoder_proj, so this is the\n                # post-projection (L, 192) input to the transformer stack.\n                out[p + "fc_encoder"] = fc_embs[0].cpu().numpy()\n                if len(conf_outs) != 17 or len(trans_outs) != 18:\n                    raise RuntimeError(\n                        f"chunk{idx:03d}: ouroboros count mismatch: "\n                        f"{len(conf_outs)} conformer (want 17) / "\n                        f"{len(trans_outs)} transformer (want 18)")\n                if getattr(_capture_block_outputs, "unknown_shapes", None):\n                    raise RuntimeError(\n                        f"chunk{idx:03d}: unknown block shapes: "\n                        f"{_capture_block_outputs.unknown_shapes}")  # noqa: E501\n                for bi, arr in enumerate(conf_outs):\n                    out[p + f"conformer_block_{bi:02d}"] = arr[0]\n                for bi, arr in enumerate(trans_outs):\n                    out[p + f"transformer_block_{bi:02d}"] = arr[0]\n                out[p + "n_conformer_blocks"] = __import__("numpy").array(\n                    [len(conf_outs)], dtype=__import__("numpy").int64)\n                out[p + "n_transformer_blocks"] = __import__("numpy").array(\n                    [len(trans_outs)], dtype=__import__("numpy").int64)\n            lc = round(left_off / sub)\n            rc = math.ceil(right_off / sub)\n\n            state, chunk_preds = sm.streaming_update(\n                streaming_state=state, chunk=pre_embs, preds=preds, lc=lc, rc=rc\n            )\n            total_preds = torch.cat([total_preds, chunk_preds], dim=1)\n\n            out[p + "chunk_preds"] = chunk_preds[0].cpu().numpy()\n            out[p + "spkcache_after"] = state.spkcache[0].cpu().numpy()\n            if state.spkcache_preds is not None:\n                out[p + "spkcache_preds_after"] = state.spkcache_preds[0].cpu().numpy()\n            out[p + "fifo_after"] = state.fifo[0].cpu().numpy()\n            out[p + "mean_sil_emb_after"] = state.mean_sil_emb[0].cpu().numpy()\n            out[p + "n_sil_frames_after"] = state.n_sil_frames.cpu().numpy()\n            # Compression happened iff the cache is back at its cap after\n            # having exceeded it (pre-update len + pop > cap).\n            if (\n                state.spkcache.shape[1] == sm.spkcache_len\n                and spk_len_before + (fifo_len_before + chunk_preds.shape[1] - state.fifo.shape[1])\n                > sm.spkcache_len\n            ):\n                compression_chunks.append(idx)\n            n_chunks = idx + 1\n\n        out["total_preds"] = total_preds[0].cpu().numpy()\n        out["n_chunks"] = np.array([n_chunks], dtype=np.int64)\n        out["compression_chunks"] = np.array(compression_chunks, dtype=np.int64)\n\n    Path(args.out).parent.mkdir(parents=True, exist_ok=True)\n    np.savez_compressed(args.out, **out)\n    total_mb = sum(v.nbytes for v in out.values()) / 1e6\n    print(\n        f"[dump] {n_chunks} chunks, compressions at {compression_chunks}, "\n        f"{total_mb:.1f} MB raw -> {args.out}"\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    import sys\n\n    sys.exit(main())\n'

REPORT: dict[str, object] = {
    "schema_version": 1,
    "scope": "pure_speaker_diarization",
    "job": "m2_stage0_refdump_spike",
    "harness_source": "diar_harness.py resolved at runtime from /kaggle/input",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "verdict": "not-run",
}


def find_audio(name: str, suffix: str) -> Path | None:
    hits = sorted(FIXTURE_ROOT.rglob(suffix))
    return hits[0] if hits else None


def sh(argv: list[str], timeout: int = 1800) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        tail = (p.stdout + "\n" + p.stderr)[-3000:]
        return p.returncode, tail
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace")[-1500:]
        err = (exc.stderr or b"").decode("utf-8", "replace")[-1500:]
        return 124, f"TIMEOUT {argv[0]}: {out}\n{err}"


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["environment"] = h.environment_record()
    REPORT["gpu_before"] = h.gpu_snapshot()

    # S0-a: does the HF repo carry a .nemo checkpoint?
    code, tree_txt = sh([
        "curl", "-L", "--fail", "--retry", "2", "--silent", "--show-error",
        f"https://huggingface.co/api/models/{HF_REPO}/tree/main?recursive=true",
    ], timeout=300)
    nemo_files: list[str] = []
    if code == 0:
        try:
            for entry in json.loads(tree_txt):
                path = str(entry.get("path", ""))
                if path.endswith(".nemo"):
                    nemo_files.append(path)
        except (ValueError, AttributeError):
            nemo_files = []
    REPORT["s0_a_hf_tree_rc"] = code
    REPORT["s0_a_nemo_files"] = nemo_files
    if not nemo_files:
        REPORT["verdict"] = "no-nemo-checkpoint"
        REPORT["note"] = (
            "HF repo has no .nemo file (or tree API unreachable); "
            "Stage 0 falls back to the ggml-probdump patch per plan §2."
        )
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage0_verdict.json")
        print(json.dumps(REPORT, indent=1)[:2000])
        return 0

    # S0-b: can nemo_toolkit[asr] install and import here?
    code, pip_txt = sh([
        "pip", "install", "--quiet", "nemo_toolkit[asr]",
    ], timeout=3600)
    REPORT["s0_b_pip_rc"] = code
    REPORT["s0_b_pip_tail"] = pip_txt[-1500:]
    code, imp_txt = sh([
        "python3", "-c",
        "import torch; from nemo.collections.asr.models import "
        "SortformerEncLabelModel; print(torch.__version__, torch.cuda.is_available())",
    ], timeout=600)
    REPORT["s0_b_import_rc"] = code
    REPORT["s0_b_import_tail"] = imp_txt[-1000:]
    if code != 0:
        REPORT["verdict"] = "nemo-not-installable"
        REPORT["note"] = (
            "nemo_toolkit unavailable on this image; "
            "Stage 0 falls back to the ggml-probdump patch per plan §2."
        )
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage0_verdict.json")
        print(json.dumps(REPORT, indent=1)[:2000])
        return 0

    # S0-c: download .nemo + vendored dump script, run on short audio.
    ckpt = TMP / "sortformer.nemo"
    if not ckpt.exists():
        code, dl_txt = sh([
            "curl", "-L", "--fail", "--retry", "3", "--silent", "--show-error",
            "-o", str(ckpt),
            f"https://huggingface.co/{HF_REPO}/resolve/main/{nemo_files[0]}?download=true",
        ], timeout=3600)
        REPORT["s0_c_ckpt_rc"] = code
        REPORT["s0_c_ckpt_tail"] = dl_txt[-500:]
        if code != 0 or not ckpt.exists():
            REPORT["verdict"] = "nemo-ckpt-download-failed"
            REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h.emit_report(REPORT, name="m2_stage0_verdict.json")
            return 0
    REPORT["s0_c_ckpt_bytes"] = ckpt.stat().st_size

    short = find_audio("short", WANT_AUDIOS["short"])
    mid = find_audio("mid", WANT_AUDIOS["mid"])
    REPORT["audios"] = {k: (str(v) if v else None)
                        for k, v in (("short", short), ("mid", mid))}
    if short is None:
        REPORT["verdict"] = "audio-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage0_verdict.json")
        return 0

    # Full-length dumps: --max-sec must cover the whole audio, else the
    # reference is a truncated prefix and any comparison against the full
    # ggml fixture is invalid (Stage 0 finding). Expected frame counts come
    # from the pinned parity fixtures (same audios, same geometry).
    import wave as _wave
    EXPECTED_FRAMES = {
        "short": 711,   # parity/fixtures/v12-short-streaming-r0
        "mid": 4467,    # parity/fixtures/v13-mid-streaming-r0
    }

    def _wav_seconds(path: Path) -> float:
        with _wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())

    dump_py = Path(__file__).parent / "dump_m2_reference.py"
    if dump_py.exists():
        REPORT["dump_script"] = str(dump_py) + " (sibling)"
    elif EMBEDDED_DUMP_M2_REFERENCE:
        dump_py = TMP / "dump_m2_reference.py"
        dump_py.write_text(EMBEDDED_DUMP_M2_REFERENCE, encoding="utf-8")
        REPORT["dump_script"] = str(dump_py) + " (embedded)"
    else:
        REPORT["dump_script"] = str(dump_py) + " (MISSING)"
        REPORT["verdict"] = "dump-script-missing"
        REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(REPORT, name="m2_stage0_verdict.json")
        return 0

    device = "cuda"
    g = GEOMETRY
    for label, audio in (("short", short), ("mid", mid) if mid else ()):
        out_npz = OUT / f"m2_ref_{label}.npz"
        # No truncation: dump the whole file (+5s headroom), then verify
        # the frame count equals the pinned fixture before accepting.
        max_sec = _wav_seconds(audio) + 5.0
        REPORT[f"s0_c_dump_{label}_max_sec"] = max_sec
        code, dump_txt = sh([
            "python3", str(dump_py), str(ckpt), str(audio), str(out_npz),
            "--device", device,
            "--max-sec", str(max_sec),
            "--chunk", str(g["chunk"]), "--lc", str(g["lc"]), "--rc", str(g["rc"]),
            "--fifo", str(g["fifo"]), "--spkcache", str(g["spkcache"]),
            "--update-period", str(g["update_period"]),
        ], timeout=5400)
        REPORT[f"s0_c_dump_{label}_rc"] = code
        REPORT[f"s0_c_dump_{label}_tail"] = dump_txt[-1500:]
        REPORT[f"s0_c_dump_{label}_npz"] = (
            str(out_npz) if out_npz.exists() else None)
        if code != 0 or not out_npz.exists():
            REPORT["verdict"] = "nemo-dump-failed"
            REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h.emit_report(REPORT, name="m2_stage0_verdict.json")
            print(json.dumps(REPORT, indent=1)[:2000])
            return 0
        # Frame-count ouroboros: the reference must cover exactly the same
        # frames as the pinned fixture (same audio, same geometry). A
        # truncated prefix would silently poison Stage 2 gates.
        import numpy as _np
        got_frames = int(_np.load(str(out_npz))["total_preds"].shape[0])
        want_frames = EXPECTED_FRAMES[label]
        REPORT[f"s0_c_dump_{label}_frames"] = got_frames
        if got_frames != want_frames:
            REPORT["verdict"] = "truncated"
            REPORT["note"] = (
                f"{label}: total_preds has {got_frames} frames, fixture pins "
                f"{want_frames}; reference rejected, not promoted to truth."
            )
            REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            h.emit_report(REPORT, name="m2_stage0_verdict.json")
            print(json.dumps(REPORT, indent=1)[:2000])
            return 0

    REPORT["verdict"] = "nemo-ok"
    REPORT["note"] = (
        ".npz reference(s) captured full-length; Stage 1/2 unblocked. "
        "NeMo total_preds is pre-gate raw sigmoid, ggml fixtures are "
        "post-BirthGate timelines: compare layer-to-layer only, never "
        "total_preds-vs-fixture directly."
    )
    REPORT["gpu_after"] = h.gpu_snapshot()
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    h.emit_report(REPORT, name="m2_stage0_verdict.json")
    print(json.dumps(REPORT, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
