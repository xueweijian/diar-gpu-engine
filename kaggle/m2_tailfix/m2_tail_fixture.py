"""M2 tail-speech fixture kernel (independent small task, Step 0 appendix).

Builds ONE self-contained fixture for the production tail-block bug: an
audio slice whose last ~1 s is loud speech, run through BOTH

  * the upstream ggml production binary (probdump-patched harness build,
    q8 GGUF, default streaming preset), and
  * python NeMo (same geometry, fp32) via the vendored dump script,

so the tail rows' divergence (production lacks the MaskedConvSequential
feat_len semantics) becomes dynamically VISIBLE — unlike the four existing
fixtures whose final emitted rows are silent.

Slice choice (local analysis, /var/minis m2-ref mid npz = same audio):
  video2_audio.wav [283.4 s, 343.4 s]
  tail 1.5 s rms ~0.21, last 150 ms rms ~0.205
  local tiny-engine ledger check: last chunk t_mel=81 (81 % 32 = 17 != 0),
  t3=11 emitted rows -> exactly the "必然分歧" geometry of fd5af5e/Step 0.

Outputs (small, downloaded from /kaggle/working):
  tail_slice.wav              16 kHz mono slice (fixture audio)
  tail_slice.probs.f32        production probdump (12B <qi> header + f32)
  tail_slice.nemo.npz         NeMo reference (total_preds + per-chunk)
  m2_tail_fixture_verdict.json

A FAIL is data: the verdict records both sides' emits even when they
disagree; only infra errors raise.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# slice parameters (pinned; see module docstring for the analysis)
# ---------------------------------------------------------------------------
SLICE_START_SEC = 283.4
SLICE_LENGTH_SEC = 60.0
SRC_NEEDLE = "video2_audio.wav"          # diar-real-audio-5
HF_REPO = "nvidia/diar_streaming_sortformer_4spk-v2"
GEOMETRY = {"chunk": 20, "lc": 0, "rc": 0, "fifo": 80, "spkcache": 160,
            "update_period": 80}

OUT = Path("/kaggle/working")
TMP = Path("/tmp/m2-tail-fixture")

# Vendored NeMo dump script (same fork the Stage 0 refdump kernel uses).
# Kaggle script kernels upload ONLY code_file, so the sibling never ships;
# scripts/embed_tailfix.py inlines it below before push. Do not hand-edit.
EMBEDDED_DUMP_M2_REFERENCE = '#!/usr/bin/env python3\n# M2 fork of upstream dump_sortformer_reference.py (NeMo-Speech.cpp a5b6953).\n# ONLY delta vs upstream: _capture_block_outputs helper + per-block hook\n# capture (conformer_block_NN / transformer_block_NN) on deep chunks.\n# The NeMo forward path is untouched (hooks observe only).\n# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"""Dump NeMo streaming Sortformer reference tensors for ggml parity tests.\n\nReplays the sync streaming loop (SortformerEncLabelModel.forward_streaming with\nasync_streaming=False) chunk by chunk, capturing every intermediate the C++\nport needs to check against:\n\n  per chunk: mel window, pre_encode embeddings, concat lengths, full preds,\n             chunk preds, post-update AOSC state (spkcache/preds/fifo/\n             mean_sil_emb/n_sil_frames), compression flag.\n  stream:    full mel sequence, final concatenated chunk preds.\n\nDefault geometry uses the runtime streaming preset. Deep intermediates\n(fc-encoder output, transformer output) are stored for the first\n--deep-chunks chunks and for every compression chunk.\n\nUsage:\n    python dump_sortformer_reference.py <ckpt.nemo> <audio.wav> <out.npz> \\\n        [--device cpu] [--max-sec 40] [--chunk 8 --lc 0 --rc 8 \\\n         --fifo 80 --spkcache 160 --update-period 80]\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport math\nfrom pathlib import Path\n\nimport numpy as np\nimport soundfile as sf\nimport torch\n\n\ndef _capture_block_outputs(model):\n    """Register forward hooks capturing per-block outputs (deep chunks).\n\n    Hooks only observe: the NeMo forward path is untouched. Returns\n    (handles, conf_outs, trans_outs); the caller removes handles right\n    after the forward.\n\n    Layer identity is by OUTPUT SHAPE (v3 ouroboros rule, Stage 0\n    finding): NeMo-side class names are not trusted. A block output of\n    (..., 512) is a conformer block, (..., 192) a transformer block;\n    anything else is recorded as unknown and fails the dump loudly\n    (never silently dropped). Hooks fire in forward order, so both\n    lists are in stack order.\n    """\n    conf_outs: list = []\n    trans_outs: list = []\n    unknown_shapes: list = []\n    handles = []\n\n    def _mk(store):\n        def _fn(module, args, output):\n            out = output[0] if isinstance(output, (tuple, list)) else output\n            arr = out.detach().cpu().numpy()\n            if arr.shape[-1] == 512:\n                conf_outs.append(arr)\n            elif arr.shape[-1] == 192:\n                trans_outs.append(arr)\n            else:\n                unknown_shapes.append(arr.shape)\n                store.append(arr)\n        return _fn\n\n    for mod in model.modules():\n        cls = type(mod).__name__\n        if cls in ("ConformerLayer", "TransformerEncoderBlock",\n                   "TransformerEncoderLayer", "TransformerLayer"):\n            handles.append(mod.register_forward_hook(_mk(conf_outs\n                                                          if cls == "ConformerLayer"\n                                                          else trans_outs)))\n    # Ouroboros: verify the class-name routing against shape routing on\n    # the first deep chunk is done by the caller via counts (17/18).\n    _capture_block_outputs.unknown_shapes = unknown_shapes\n    return handles, conf_outs, trans_outs\n\n\ndef main() -> int:\n    ap = argparse.ArgumentParser()\n    ap.add_argument("ckpt", help=".nemo checkpoint path")\n    ap.add_argument("audio", help="16 kHz mono wav")\n    ap.add_argument("out", help="output .npz path")\n    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))\n    ap.add_argument("--max-sec", type=float, default=40.0)\n    # Defaults = the runtime streaming preset (DiarGeometry, aosc_state.h) for\n    # convenience only: the chosen geometry is recorded in the dump and the\n    # parity test reads it from there, so any values make a valid reference.\n    ap.add_argument("--chunk", type=int, default=20, help="chunk_len in 80ms encoder frames")\n    ap.add_argument("--lc", type=int, default=0, help="chunk_left_context in encoder frames")\n    ap.add_argument("--rc", type=int, default=0, help="chunk_right_context in encoder frames")\n    ap.add_argument("--fifo", type=int, default=80)\n    ap.add_argument("--spkcache", type=int, default=160)\n    ap.add_argument("--update-period", type=int, default=80)\n    ap.add_argument(\n        "--deep-chunks",\n        type=int,\n        default=3,\n        help="store fc-encoder/transformer outputs for the first N chunks",\n    )\n    args = ap.parse_args()\n\n    from nemo.collections.asr.models import SortformerEncLabelModel\n\n    device = torch.device(args.device)\n    model = SortformerEncLabelModel.restore_from(restore_path=args.ckpt, map_location=device)\n    model.eval()\n    model.to(device)\n\n    sm = model.sortformer_modules\n    sm.chunk_len = args.chunk\n    sm.chunk_left_context = args.lc\n    sm.chunk_right_context = args.rc\n    sm.fifo_len = args.fifo\n    sm.spkcache_len = args.spkcache\n    sm.spkcache_update_period = args.update_period\n    sm._check_streaming_parameters()\n\n    # Determinism: no dither (riva sets 0 at inference too).\n    model.preprocessor.featurizer.dither = 0.0\n\n    audio, sr = sf.read(args.audio, dtype="float32")\n    if audio.ndim > 1:\n        audio = audio[:, 0]\n    assert sr == 16000, f"expected 16 kHz audio, got {sr}"\n    max_samples = int(args.max_sec * sr)\n    audio = audio[:max_samples]\n\n    sig = torch.from_numpy(audio).unsqueeze(0).to(device)\n    sig_len = torch.tensor([sig.shape[1]], device=device)\n\n    with torch.inference_mode():\n        # Streaming mode: no max-normalization (SortformerEncLabelModel.\n        # process_signal only rescales when streaming_mode is off).\n        mel, mel_len = model.preprocessor(input_signal=sig, length=sig_len)\n\n        out: dict[str, np.ndarray] = {\n            "audio": audio,\n            "mel": mel[0].cpu().numpy(),  # (128, T)\n            "mel_len": mel_len.cpu().numpy(),\n            "geometry": np.array(\n                [args.chunk, args.lc, args.rc, args.fifo, args.spkcache, args.update_period],\n                dtype=np.int64,\n            ),\n        }\n\n        state = sm.init_streaming_state(batch_size=1, async_streaming=False, device=device)\n        total_preds = torch.zeros((1, 0, sm.n_spk), device=device)\n        offset = torch.zeros((1,), dtype=torch.long, device=device)\n        sub = model.encoder.subsampling_factor\n\n        n_chunks = 0\n        compression_chunks = []\n        for idx, chunk_feat, feat_lengths, left_off, right_off in sm.streaming_feat_loader(\n            feat_seq=mel, feat_seq_length=mel_len, feat_seq_offset=offset\n        ):\n            p = f"chunk{idx:03d}/"\n            out[p + "mel_window"] = chunk_feat[0].cpu().numpy()  # (T_mel, 128)\n            out[p + "feat_length"] = feat_lengths.cpu().numpy()\n            out[p + "offsets"] = np.array([left_off, right_off], dtype=np.int64)\n\n            pre_embs, pre_lens = model.encoder.pre_encode(x=chunk_feat, lengths=feat_lengths)\n            out[p + "pre_encode"] = pre_embs[0].cpu().numpy()  # (T_enc, 512)\n\n            spk_len_before = state.spkcache.shape[1]\n            fifo_len_before = state.fifo.shape[1]\n            out[p + "state_lens_before"] = np.array(\n                [spk_len_before, fifo_len_before, int(pre_lens[0])], dtype=np.int64\n            )\n\n            concat_embs = sm.concat_embs(\n                [state.spkcache, state.fifo, pre_embs], dim=1, device=device\n            )\n            concat_lens = spk_len_before + fifo_len_before + pre_lens\n\n            handles, conf_outs, trans_outs = (\n                _capture_block_outputs(model)\n                if idx < args.deep_chunks else ([], [], []))\n            fc_embs, fc_lens = model.frontend_encoder(\n                processed_signal=concat_embs,\n                processed_signal_length=concat_lens,\n                bypass_pre_encode=True,\n            )\n            # NOTE: handles stay registered through forward_infer: the 18\n            # transformer blocks fire there, not in frontend_encoder (v3\n            # removed too early and captured 0 transformer outputs).\n            preds = model.forward_infer(emb_seq=fc_embs, emb_seq_length=fc_lens)\n            preds = sm.apply_mask_to_preds(preds, fc_lens)\n            for hd in handles:\n                hd.remove()\n            out[p + "preds_full"] = preds[0].cpu().numpy()  # (L1+L2+L3, 4)\n\n            deep = idx < args.deep_chunks\n            if deep:\n                # frontend_encoder applies encoder_proj, so this is the\n                # post-projection (L, 192) input to the transformer stack.\n                out[p + "fc_encoder"] = fc_embs[0].cpu().numpy()\n                if len(conf_outs) != 17 or len(trans_outs) != 18:\n                    raise RuntimeError(\n                        f"chunk{idx:03d}: ouroboros count mismatch: "\n                        f"{len(conf_outs)} conformer (want 17) / "\n                        f"{len(trans_outs)} transformer (want 18)")\n                if getattr(_capture_block_outputs, "unknown_shapes", None):\n                    raise RuntimeError(\n                        f"chunk{idx:03d}: unknown block shapes: "\n                        f"{_capture_block_outputs.unknown_shapes}")  # noqa: E501\n                for bi, arr in enumerate(conf_outs):\n                    out[p + f"conformer_block_{bi:02d}"] = arr[0]\n                for bi, arr in enumerate(trans_outs):\n                    out[p + f"transformer_block_{bi:02d}"] = arr[0]\n                out[p + "n_conformer_blocks"] = __import__("numpy").array(\n                    [len(conf_outs)], dtype=__import__("numpy").int64)\n                out[p + "n_transformer_blocks"] = __import__("numpy").array(\n                    [len(trans_outs)], dtype=__import__("numpy").int64)\n            lc = round(left_off / sub)\n            rc = math.ceil(right_off / sub)\n\n            state, chunk_preds = sm.streaming_update(\n                streaming_state=state, chunk=pre_embs, preds=preds, lc=lc, rc=rc\n            )\n            total_preds = torch.cat([total_preds, chunk_preds], dim=1)\n\n            out[p + "chunk_preds"] = chunk_preds[0].cpu().numpy()\n            out[p + "spkcache_after"] = state.spkcache[0].cpu().numpy()\n            if state.spkcache_preds is not None:\n                out[p + "spkcache_preds_after"] = state.spkcache_preds[0].cpu().numpy()\n            out[p + "fifo_after"] = state.fifo[0].cpu().numpy()\n            out[p + "mean_sil_emb_after"] = state.mean_sil_emb[0].cpu().numpy()\n            out[p + "n_sil_frames_after"] = state.n_sil_frames.cpu().numpy()\n            # Compression happened iff the cache is back at its cap after\n            # having exceeded it (pre-update len + pop > cap).\n            if (\n                state.spkcache.shape[1] == sm.spkcache_len\n                and spk_len_before + (fifo_len_before + chunk_preds.shape[1] - state.fifo.shape[1])\n                > sm.spkcache_len\n            ):\n                compression_chunks.append(idx)\n            n_chunks = idx + 1\n\n        out["total_preds"] = total_preds[0].cpu().numpy()\n        out["n_chunks"] = np.array([n_chunks], dtype=np.int64)\n        out["compression_chunks"] = np.array(compression_chunks, dtype=np.int64)\n\n    Path(args.out).parent.mkdir(parents=True, exist_ok=True)\n    np.savez_compressed(args.out, **out)\n    total_mb = sum(v.nbytes for v in out.values()) / 1e6\n    print(\n        f"[dump] {n_chunks} chunks, compressions at {compression_chunks}, "\n        f"{total_mb:.1f} MB raw -> {args.out}"\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    import sys\n\n    sys.exit(main())\n'


def sh(argv: list[str], timeout: int = 3600) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + "\n" + p.stderr)[-3000:]
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace")[-1500:]
        err = (exc.stderr or b"").decode("utf-8", "replace")[-1500:]
        return 124, f"TIMEOUT {argv[0]}: {out}\n{err}"


def locate_harness() -> Path:
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("diar_harness.py")):
            return candidate.parent
    raise RuntimeError("diar_harness.py not found under /kaggle/input|working")


def find_input(name: str) -> Path:
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise RuntimeError(f"{name} not found under /kaggle/input|working")


def cut_wav(source: Path, destination: Path, start_seconds: float,
            seconds: float) -> Path:
    """Contiguous slice, preserving the source format (16k mono int16)."""
    with wave.open(str(source), "rb") as handle:
        params = handle.getparams()
        if params.framerate != 16000 or params.nchannels != 1:
            raise RuntimeError(
                f"unexpected source format: {params.framerate} Hz "
                f"{params.nchannels}ch (want 16 kHz mono)")
        handle.setpos(int(start_seconds * params.framerate))
        frames = handle.readframes(int(seconds * params.framerate))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return destination


def main() -> int:
    started = time.time()
    report: dict = {
        "schema_version": 1,
        "job": "m2_tail_fixture",
        "scope": "pure_speaker_diarization",
        "slice": {"src": SRC_NEEDLE, "start_sec": SLICE_START_SEC,
                  "length_sec": SLICE_LENGTH_SEC},
        "geometry": GEOMETRY,
    }
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    harness_dir = locate_harness()
    sys.path.insert(0, str(harness_dir))
    import diar_harness as h  # noqa: E402

    report["environment"] = h.environment_record()
    report["gpu_before"] = h.gpu_snapshot()
    report["harness_dir"] = str(harness_dir)

    src = find_input(SRC_NEEDLE)
    report["src_audio"] = str(src)
    slice_wav = OUT / "tail_slice.wav"
    cut_wav(src, slice_wav, SLICE_START_SEC, SLICE_LENGTH_SEC)
    report["slice_bytes"] = slice_wav.stat().st_size
    report["slice_sha256"] = h.sha256(slice_wav)
    print(f"[tail] slice -> {slice_wav} ({report['slice_bytes']} bytes)",
          flush=True)

    # ---- production side: build (probdump patch) + diarize --------------
    build = h.build_runtime()
    report["build"] = build
    print(f"[tail] build: {build['build_seconds']}s "
          f"patch={build['probdump_patch']}", flush=True)
    model = h.download_model()
    report["model"] = model
    binary = Path(build["binary"])

    probs_path = OUT / "tail_slice.probs.f32"
    rec = h.diarize_once(binary, slice_wav, OUT / "tail_slice.rttm",
                         dump_probs_path=probs_path)
    report["production"] = rec
    print(f"[tail] production rc={rec.get('returncode')} "
          f"probs_sha={rec.get('probs_sha256')}", flush=True)
    if rec.get("returncode") != 0 or not probs_path.exists():
        report["verdict"] = "production-failed"
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(report, name="m2_tail_fixture_verdict.json")
        return 0

    # decode the probdump header for the record (no numpy needed yet)
    import struct
    raw = probs_path.read_bytes()
    n_frames, n_spk = struct.unpack("<qi", raw[:12])
    report["production"]["probdump_frames"] = n_frames
    report["production"]["probdump_spk"] = n_spk
    print(f"[tail] producdump: {n_frames} frames x {n_spk} spk", flush=True)

    # determinism double-run (same session, M1 discipline)
    probs2 = TMP / "tail_slice.probs.run2.f32"
    rec2 = h.diarize_once(binary, slice_wav, OUT / "tail_slice.run2.rttm",
                          dump_probs_path=probs2)
    same = (probs2.exists() and probs2.read_bytes() == raw)
    report["production_run2_sha"] = rec2.get("probs_sha256")
    report["production_bit_identical"] = same
    print(f"[tail] run2 bit_identical={same}", flush=True)

    # ---- NeMo side: nemo_toolkit + .nemo + dump on the same slice --------
    code, pip_txt = sh(["pip", "install", "--quiet", "nemo_toolkit[asr]"],
                       timeout=3600)
    report["nemo_pip_rc"] = code
    if code != 0:
        report["nemo_pip_tail"] = pip_txt[-1500:]
        report["verdict"] = "nemo-not-installable"
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(report, name="m2_tail_fixture_verdict.json")
        return 0

    ckpt = TMP / "sortformer.nemo"
    code, dl_txt = sh(["curl", "-L", "--fail", "--retry", "3", "--silent",
                       "--show-error", "-o", str(ckpt),
                       f"https://huggingface.co/{HF_REPO}/resolve/main/"
                       "diar_streaming_sortformer_4spk-v2.nemo?download=true"])
    report["nemo_ckpt_rc"] = code
    if code != 0 or not ckpt.exists():
        report["nemo_ckpt_tail"] = dl_txt[-500:]
        report["verdict"] = "nemo-ckpt-download-failed"
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(report, name="m2_tail_fixture_verdict.json")
        return 0
    report["nemo_ckpt_bytes"] = ckpt.stat().st_size

    dump_py = Path(__file__).parent / "dump_m2_reference.py"
    if dump_py.exists():
        report["dump_script"] = str(dump_py) + " (sibling)"
    elif EMBEDDED_DUMP_M2_REFERENCE:
        dump_py = TMP / "dump_m2_reference.py"
        dump_py.write_text(EMBEDDED_DUMP_M2_REFERENCE, encoding="utf-8")
        report["dump_script"] = str(dump_py) + " (embedded)"
    else:
        report["verdict"] = "dump-script-missing"
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h.emit_report(report, name="m2_tail_fixture_verdict.json")
        return 0

    npz_path = OUT / "tail_slice.nemo.npz"
    g = GEOMETRY
    code, dump_txt = sh([
        "python3", str(dump_py), str(ckpt), str(slice_wav), str(npz_path),
        "--device", "cuda",
        "--max-sec", str(SLICE_LENGTH_SEC + 5.0),
        "--chunk", str(g["chunk"]), "--lc", str(g["lc"]), "--rc", str(g["rc"]),
        "--fifo", str(g["fifo"]), "--spkcache", str(g["spkcache"]),
        "--update-period", str(g["update_period"]),
    ], timeout=5400)
    report["nemo_dump_rc"] = code
    report["nemo_dump_tail"] = dump_txt[-2500:]
    print(f"[tail] nemo dump rc={code}\n{dump_txt[-800:]}", flush=True)

    if code == 0 and npz_path.exists():
        report["nemo_npz_bytes"] = npz_path.stat().st_size
        # quick side-by-side header check (needs numpy, available after pip)
        try:
            import numpy as np
            z = np.load(npz_path, allow_pickle=True)
            tp = np.asarray(z["total_preds"], dtype=np.float64)
            prod = np.frombuffer(raw, "<f4", count=n_frames * n_spk,
                                 offset=12).reshape(n_frames, n_spk)
            n = min(len(tp), n_frames)
            d = np.abs(tp[:n] - prod[:n])
            report["compare"] = {
                "rows_prod": int(n_frames), "rows_nemo": int(len(tp)),
                "rows_compared": int(n),
                "max_abs": float(d.max()),
                "tail16_prod_max": float(np.abs(prod[-16:]).max()),
                "tail16_diff_max": float(d[-16:].max()),
                "last_row_diff_max": float(d[-1].max()),
            }
            print(f"[tail] compare: {report['compare']}", flush=True)
        except Exception as exc:      # comparison is a bonus, never fatal
            report["compare_error"] = f"{type(exc).__name__}: {exc}"
        report["verdict"] = "fixture-built"
    else:
        report["verdict"] = "nemo-dump-failed"

    shutil.copy(probs_path, TMP / probs_path.name)  # stage INTO tmp, not onto itself
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["seconds"] = round(time.time() - started, 1)
    h.emit_report(report, name="m2_tail_fixture_verdict.json")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("verdict", "production", "compare", "seconds")},
                     indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
