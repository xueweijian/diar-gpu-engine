"""M2 Stage 0 spike files: mechanics tests (no torch/NeMo needed).

Covers what can be pinned locally:
- both kernel-side scripts compile;
- the vendored dump FORK DIFF touches only the three intended regions
  (helper + hook wrap + deep-block store); the NeMo forward path lines
  are verbatim upstream (anti-drift: a future re-vendor that silently
  alters the streaming loop fails here);
- _capture_block_outputs (extracted from the shipped file by AST, not a
  copy) captures ConformerLayer/TransformerEncoderBlock in execution
  order, ignores other modules, and returns removable handles.
"""
from __future__ import annotations

import ast
import py_compile
from pathlib import Path

REFDUMP_DIR = Path(__file__).parent.parent / "kaggle" / "m2_refdump"
DUMP = REFDUMP_DIR / "dump_m2_reference.py"
SPIKE = REFDUMP_DIR / "m2_stage0_refdump.py"
UPSTREAM = Path("/tmp/nemo-src/scripts/asr/dump_sortformer_reference.py")

# NeMo forward-path lines the fork must never alter (verbatim upstream).
FORWARD_PATH_LINES = (
    "state = sm.init_streaming_state(batch_size=1, async_streaming=False, device=device)",
    "for idx, chunk_feat, feat_lengths, left_off, right_off in sm.streaming_feat_loader(",
    "pre_embs, pre_lens = model.encoder.pre_encode(x=chunk_feat, lengths=feat_lengths)",
    "state, chunk_preds = sm.streaming_update(",
    "total_preds = torch.cat([total_preds, chunk_preds], dim=1)",
)


def test_spike_scripts_compile() -> None:
    for path in (DUMP, SPIKE):
        assert path.exists(), f"{path} missing"
        py_compile.compile(str(path), doraise=True)


def test_fork_diff_only_additive() -> None:
    upstream = UPSTREAM.read_text().splitlines()
    fork = DUMP.read_text().splitlines()
    # Every upstream line survives verbatim and in order (pure insertion).
    it = iter(fork)
    for line in upstream:
        for fline in it:
            if fline == line:
                break
        else:
            raise AssertionError(f"upstream line lost or altered: {line!r}")


def test_forward_path_verbatim() -> None:
    text = DUMP.read_text()
    for line in FORWARD_PATH_LINES:
        assert line in text, f"forward path line altered: {line!r}"


def _load_helper():
    tree = ast.parse(DUMP.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_capture_block_outputs":
            ns: dict = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(DUMP), "exec"), ns)
            return ns["_capture_block_outputs"]
    raise AssertionError("_capture_block_outputs missing from shipped dump script")


class _FakeTensor:
    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _FakeModule:
    def __init__(self, out):
        self._out = out
        self._hooks: list = []

    def register_forward_hook(self, fn):
        handle = _Handle()
        self._hooks.append((handle, fn))
        return handle

    def fire(self):
        for handle, fn in self._hooks:
            fn(self, (), self._out)


class ConformerLayer(_FakeModule):
    pass


class TransformerEncoderBlock(_FakeModule):
    pass


class Linear(_FakeModule):
    pass


class _FakeModel:
    def __init__(self, mods):
        self._mods = mods

    def modules(self):
        return [self, *self._mods]


def test_hook_helper_order_and_selectivity() -> None:
    helper = _load_helper()
    c0, c1 = ConformerLayer(_FakeTensor("c0")), ConformerLayer(_FakeTensor("c1"))
    lin = Linear(_FakeTensor("lin-out"))
    t0 = TransformerEncoderBlock(_FakeTensor("t0"))
    model = _FakeModel([c0, lin, c1, t0])
    handles, conf_outs, trans_outs = helper(model)
    assert len(handles) == 3, f"expected 3 hooked layers, got {len(handles)}"
    # Fire in forward order; Linear must never have been hooked.
    assert lin._hooks == []
    for mod in (c0, lin, c1, t0):
        mod.fire()
    assert conf_outs == ["c0", "c1"], conf_outs
    assert trans_outs == ["t0"], trans_outs
    # Removal contract: every handle removable (dump loop try/finally).
    for hd in handles:
        hd.remove()
        assert hd.removed


def test_spike_has_three_verdicts() -> None:
    text = SPIKE.read_text()
    for verdict in ("no-nemo-checkpoint", "nemo-not-installable", "nemo-ok"):
        assert verdict in text, f"spike verdict {verdict!r} missing"
    assert "dump_m2_reference.py" in text
