"""Re-vendor kaggle/m2_refdump/dump_m2_reference.py from upstream.

Delta vs upstream dump_sortformer_reference.py (a5b6953):
  1. module-level _capture_block_outputs helper (forward hooks, observe-only);
  2. hook register/remove around the frontend_encoder call on deep chunks
     (sequential, NOT try/finally: an exception aborts the whole dump, so
     there is no path that needs handle cleanup — and this keeps every
     upstream line byte-verbatim, enforced by test_fork_diff_only_additive);
  3. per-block outputs (conformer_block_NN / transformer_block_NN + counts)
     stored for deep chunks.
The NeMo forward path is untouched.
"""
from __future__ import annotations

UPSTREAM = "/tmp/nemo-src/scripts/asr/dump_sortformer_reference.py"
OUT = "/var/minis/workspace/diar-gpu-engine/kaggle/m2_refdump/dump_m2_reference.py"

HELPER = '''

def _capture_block_outputs(model):
    """Register forward hooks capturing per-block outputs (deep chunks).

    Hooks only observe: the NeMo forward path is untouched. Returns
    (handles, conf_outs, trans_outs); the caller removes handles right
    after the forward.

    Layer identity is by OUTPUT SHAPE (v3 ouroboros rule, Stage 0
    finding): NeMo-side class names are not trusted. A block output of
    (..., 512) is a conformer block, (..., 192) a transformer block;
    anything else is recorded as unknown and fails the dump loudly
    (never silently dropped). Hooks fire in forward order, so both
    lists are in stack order.
    """
    conf_outs: list = []
    trans_outs: list = []
    unknown_shapes: list = []
    handles = []

    def _mk(store):
        def _fn(module, args, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            arr = out.detach().cpu().numpy()
            if arr.shape[-1] == 512:
                conf_outs.append(arr)
            elif arr.shape[-1] == 192:
                trans_outs.append(arr)
            else:
                unknown_shapes.append(arr.shape)
                store.append(arr)
        return _fn

    for mod in model.modules():
        cls = type(mod).__name__
        if cls in ("ConformerLayer", "TransformerEncoderBlock",
                   "TransformerEncoderLayer", "TransformerLayer"):
            handles.append(mod.register_forward_hook(_mk(conf_outs
                                                          if cls == "ConformerLayer"
                                                          else trans_outs)))
    # Ouroboros: verify the class-name routing against shape routing on
    # the first deep chunk is done by the caller via counts (17/18).
    _capture_block_outputs.unknown_shapes = unknown_shapes
    return handles, conf_outs, trans_outs

'''

OLD_FWD = """            fc_embs, fc_lens = model.frontend_encoder(
                processed_signal=concat_embs,
                processed_signal_length=concat_lens,
                bypass_pre_encode=True,
            )"""
NEW_FWD = """            handles, conf_outs, trans_outs = (
                _capture_block_outputs(model)
                if idx < args.deep_chunks else ([], [], []))
            fc_embs, fc_lens = model.frontend_encoder(
                processed_signal=concat_embs,
                processed_signal_length=concat_lens,
                bypass_pre_encode=True,
            )
            # NOTE: handles stay registered through forward_infer: the 18
            # transformer blocks fire there, not in frontend_encoder (v3
            # removed too early and captured 0 transformer outputs).
            preds = model.forward_infer(emb_seq=fc_embs, emb_seq_length=fc_lens)
            preds = sm.apply_mask_to_preds(preds, fc_lens)
            for hd in handles:
                hd.remove()"""

OLD_POST = """            preds = model.forward_infer(emb_seq=fc_embs, emb_seq_length=fc_lens)
            preds = sm.apply_mask_to_preds(preds, fc_lens)
            out[p + "preds_full"] = preds[0].cpu().numpy()  # (L1+L2+L3, 4)"""
NEW_POST = """            out[p + "preds_full"] = preds[0].cpu().numpy()  # (L1+L2+L3, 4)"""

OLD_FC = """                out[p + "fc_encoder"] = fc_embs[0].cpu().numpy()"""
NEW_FC = OLD_FC + """
                if len(conf_outs) != 17 or len(trans_outs) != 18:
                    raise RuntimeError(
                        f"chunk{idx:03d}: ouroboros count mismatch: "
                        f"{len(conf_outs)} conformer (want 17) / "
                        f"{len(trans_outs)} transformer (want 18)")
                if getattr(_capture_block_outputs, "unknown_shapes", None):
                    raise RuntimeError(
                        f"chunk{idx:03d}: unknown block shapes: "
                        f"{_capture_block_outputs.unknown_shapes}")  # noqa: E501
                for bi, arr in enumerate(conf_outs):
                    out[p + f"conformer_block_{bi:02d}"] = arr[0]
                for bi, arr in enumerate(trans_outs):
                    out[p + f"transformer_block_{bi:02d}"] = arr[0]
                out[p + "n_conformer_blocks"] = __import__("numpy").array(
                    [len(conf_outs)], dtype=__import__("numpy").int64)
                out[p + "n_transformer_blocks"] = __import__("numpy").array(
                    [len(trans_outs)], dtype=__import__("numpy").int64)"""


def main() -> None:
    src = open(UPSTREAM).read()
    anchor = "def main() -> int:"
    assert anchor in src, "main anchor missing"
    src = src.replace(anchor, HELPER.strip("\n") + "\n\n\n" + anchor, 1)
    assert OLD_FWD in src, "frontend_encoder call site missing"
    src = src.replace(OLD_FWD, NEW_FWD, 1)
    assert OLD_POST in src, "preds post site missing"
    src = src.replace(OLD_POST, NEW_POST, 1)
    assert OLD_FC in src, "fc_encoder site missing"
    src = src.replace(OLD_FC, NEW_FC, 1)
    lines = src.splitlines()
    assert lines[0].startswith("#!"), "unexpected first line"
    fork_note = (
        "# M2 fork of upstream dump_sortformer_reference.py (NeMo-Speech.cpp a5b6953).\n"
        "# ONLY delta vs upstream: _capture_block_outputs helper + per-block hook\n"
        "# capture (conformer_block_NN / transformer_block_NN) on deep chunks.\n"
        "# The NeMo forward path is untouched (hooks observe only).\n"
    )
    src = lines[0] + "\n" + fork_note + "\n".join(lines[1:]) + "\n"
    import ast
    ast.parse(src)
    open(OUT, "w").write(src)
    print("written, bytes:", len(src))


if __name__ == "__main__":
    main()
