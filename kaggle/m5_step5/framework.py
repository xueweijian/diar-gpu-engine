# M3 Step 5 kernel — fp16 weight storage (GemmEx 16F-in / 32F-acc, opt-in).
# Kaggle GPU (T4, CUDA 12.8). Embedded sources: 12 headers + 9 impls + harness.
# Every Step-4 gate reruns as regression (the layer bodies are templates now;
# the fp32 instantiation must reproduce the Step-4 numbers), plus:
#   G-S5a cast kernel bit-identity vs host float_to_half (diff == 0)
#   G-S5b fp16 linear parity, 5 engine shapes (+ hq dequant-ref rows)
#   G-S5c conformer layer fp16 parity (t=20)
#   G-S5d transformer block fp16 parity (t=12/20/26)
#   G-S5e GEMM p50 fp16 <= fp32 per shape (bandwidth evidence)
#   G-S5f fp16 timelines (conf x17 / full chain) not slower than fp32
# PTX-JIT parity reruns the parity section (incl. every fp16 gate).
import base64, json, os, subprocess, sys, time

EMBED = {}
#__EMBED_TABLE__

WORK = "/tmp/step5"
os.makedirs(WORK + "/include/diar", exist_ok=True)
os.makedirs(WORK + "/src", exist_ok=True)
os.makedirs(WORK + "/tools", exist_ok=True)

FILES = {
    "cpp_backend":        WORK + "/src/backend.cpp",
    "hpp_backend":        WORK + "/include/diar/backend.hpp",
    "hpp_cublas_layout":  WORK + "/include/diar/cublas_layout.hpp",
    "hpp_backend_cuda":   WORK + "/include/diar/backend_cuda.hpp",
    "hpp_nn":             WORK + "/include/diar/nn.hpp",
    "hpp_gguf":           WORK + "/include/diar/gguf.hpp",
    "hpp_mha":            WORK + "/include/diar/mha.hpp",
    "hpp_conv":           WORK + "/include/diar/conv.hpp",
    "hpp_conformer":      WORK + "/include/diar/conformer.hpp",
    "hpp_layers":         WORK + "/include/diar/layers.hpp",
    "hpp_posenc":         WORK + "/include/diar/posenc.hpp",
    "hpp_subsampling":    WORK + "/include/diar/subsampling.hpp",
    "cpp_cublas_layout":  WORK + "/src/cublas_layout.cpp",
    "cpp_backend_cuda":   WORK + "/src/backend_cuda.cpp",
    "cpp_nn":             WORK + "/src/nn.cpp",
    "cpp_gguf":           WORK + "/src/gguf.cpp",
    "cpp_mha":            WORK + "/src/mha.cpp",
    "cpp_conv":           WORK + "/src/conv.cpp",
    "cpp_conformer":      WORK + "/src/conformer.cpp",
    "cpp_layers":         WORK + "/src/layers.cpp",
    "cpp_posenc":         WORK + "/src/posenc.cpp",
    "cpp_subsampling":    WORK + "/src/subsampling.cpp",
    "cpp_bench_main":     WORK + "/tools/step5_bench_main.cpp",
}
for key, path in FILES.items():
    with open(path, "wb") as f:
        f.write(base64.b64decode(EMBED[key]))
print("[s5k] sources materialized:", len(FILES))

REPORT = {"gates": {}, "notes": []}
T0 = time.time()

def sh(cmd, timeout=1200, check=True):
    print("[s5k] $", " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    print(f"[s5k] rc={p.returncode} ({dt:.1f}s)", flush=True)
    if p.stdout.strip():
        print(p.stdout[-3000:], flush=True)
    if p.returncode != 0 and check:
        if p.stderr.strip():
            print(p.stderr[-3000:], flush=True)
        REPORT["notes"].append(f"FAILED: {' '.join(cmd[:6])}... rc={p.returncode}")
        raise SystemExit(3)
    return p

REF_CPPS = [FILES["cpp_cublas_layout"], FILES["cpp_nn"], FILES["cpp_gguf"],
            FILES["cpp_mha"], FILES["cpp_conv"], FILES["cpp_conformer"],
            FILES["cpp_layers"], FILES["cpp_posenc"], FILES["cpp_subsampling"]]

# 1) CPU selftest (reference chain + naive oracles, harness logic)
sh(["g++", "-std=c++17", "-Wall", "-Wextra", "-I", WORK + "/include", "-O2",
    FILES["cpp_bench_main"], *REF_CPPS, "-o", WORK + "/selftest_cpu"])
st = sh([WORK + "/selftest_cpu"])
REPORT["gates"]["cpu_selftest"] = '"mode":"cpu-selftest-complete"' in st.stdout
print("[s5k] cpu selftest gates:", REPORT["gates"]["cpu_selftest"])

# 2) nvcc SASS fatbin: sm_60/70/75 + compute_60 PTX
GENCODE = ["-gencode", "arch=compute_60,code=sm_60",
           "-gencode", "arch=compute_70,code=sm_70",
           "-gencode", "arch=compute_75,code=sm_75",
           "-gencode", "arch=compute_60,code=compute_60"]
sh(["nvcc", "-O3", "-std=c++17", "-DDIAR_WITH_CUDA", "-I", WORK + "/include",
    *GENCODE,
    "-x", "cu", FILES["cpp_backend_cuda"],
    "-x", "cu", FILES["cpp_bench_main"],
    FILES["cpp_backend"], *REF_CPPS,
    "-lcublas", "-o", WORK + "/bench_sass"], timeout=1200)

# verify the fatbin really carries 3 SASS + PTX
elf = sh(["cuobjdump", "--list-elf", WORK + "/bench_sass"], check=False).stdout
ptx = sh(["cuobjdump", "--list-ptx", WORK + "/bench_sass"], check=False).stdout
print("[s5k] raw --list-elf:\n" + elf)
print("[s5k] raw --list-ptx:\n" + ptx)
def _archs(text, prefix):
    toks = {t.strip(",;:").split(".")[0] for t in text.replace(".", ". ").split()}
    return sorted(t for t in toks if t.startswith(prefix))
sass_archs = _archs(elf, "sm_")
ptx_archs = _archs(ptx, "sm_") + _archs(ptx, "compute_")
print("[s5k] SASS archs in fatbin:", sass_archs)
print("[s5k] PTX archs in fatbin:", ptx_archs)
REPORT["gates"]["fatbin_sass_all3"] = all(a in sass_archs for a in ("sm_60", "sm_70", "sm_75"))
REPORT["gates"]["fatbin_ptx_compute60"] = ("compute_60" in ptx_archs or
                                           ".sm_60.ptx" in ptx)

# 3) GPU bench (SASS path): full gates + timelines
b1 = sh([WORK + "/bench_sass"])
parsed = []
for l in b1.stdout.splitlines():
    if l.startswith("[s5] "):
        try:
            parsed.append(json.loads(l[5:]))
        except Exception:
            pass
REPORT["bench_sass"] = parsed
def gate_from(rows, key, gate_name):
    sel = [r for r in rows if r.get("k") == key]
    REPORT["gates"][gate_name] = bool(sel) and all(r.get("pass") for r in sel)
    return sel
# Step-4 regression family
gate_from(parsed, "parity_softmax", "g_s4a_softmax")
gate_from(parsed, "parity_glu", "g_s4a_glu")
gate_from(parsed, "parity_dwconv", "g_s4a_dwconv")
gate_from(parsed, "parity_mha", "g_s4b_mha")
gate_from(parsed, "parity_layer", "g_s4c_layer")
gate_from(parsed, "parity_tf", "g_s4e_tf")
# Step-5 family
gate_from(parsed, "parity_cast", "g_s5a_cast")
gate_from(parsed, "parity_linear_h", "g_s5b_linear_h")
gate_from(parsed, "parity_linear_hq", "g_s5b_linear_hq")
gate_from(parsed, "parity_layer_h", "g_s5c_layer_h")
gate_from(parsed, "parity_tf_h", "g_s5d_tf_h")
gate_from(parsed, "gemm_p50", "g_s5e_gemm_p50")
for tl_key in ("timeline", "timeline_tf", "timeline_full",
               "timeline_h", "timeline_h_full"):
    rows = [r for r in parsed if r.get("k") == tl_key]
    if rows:
        REPORT[tl_key] = rows

def _fp16_not_slower() -> bool:
    """G-S5f: fp16 full-chain timeline <= fp32 per t (both configs)."""
    def by_t(rows):
        return {r.get("t"): r.get("ms_per_chunk") for r in rows}
    f32 = by_t(REPORT.get("timeline_full", []))
    f16 = by_t(REPORT.get("timeline_h_full", []))
    if not f32 or not f16:
        print("[s5k] fp16-not-slower: missing timeline rows")
        return False
    ok = True
    for t, ms32 in f32.items():
        ms16 = f16.get(t)
        print(f"[s5k] fp16-not-slower t={t}: fp32={ms32} fp16={ms16}")
        if ms16 is None or ms16 > ms32:
            ok = False
    return ok

REPORT["gates"]["g_s5f_fp16_not_slower"] = _fp16_not_slower()

# 4) PTX-only binary (compute_60 PTX; JIT-compiles to sm_75 at load) —
#    parity section must reproduce the SASS numbers (Step-3 mechanism).
sh(["nvcc", "-O3", "-std=c++17", "-DDIAR_WITH_CUDA", "-I", WORK + "/include",
    "-gencode", "arch=compute_60,code=compute_60",
    "-x", "cu", FILES["cpp_backend_cuda"],
    "-x", "cu", FILES["cpp_bench_main"],
    FILES["cpp_backend"], *REF_CPPS,
    "-lcublas", "-o", WORK + "/bench_ptx60"], timeout=1200)
b2 = sh([WORK + "/bench_ptx60", "--parity-only"])
parsed2 = []
for l in b2.stdout.splitlines():
    if l.startswith("[s5] "):
        try:
            parsed2.append(json.loads(l[5:]))
        except Exception:
            pass
REPORT["bench_ptx60"] = parsed2
gate_from(parsed2, "parity_softmax", "g_s4jit_softmax")
gate_from(parsed2, "parity_glu", "g_s4jit_glu")
gate_from(parsed2, "parity_dwconv", "g_s4jit_dwconv")
gate_from(parsed2, "parity_mha", "g_s4jit_mha")
gate_from(parsed2, "parity_layer", "g_s4jit_layer")
gate_from(parsed2, "parity_tf", "g_s4jit_tf")
gate_from(parsed2, "parity_cast", "g_s5jit_cast")
gate_from(parsed2, "parity_linear_h", "g_s5jit_linear_h")
gate_from(parsed2, "parity_linear_hq", "g_s5jit_linear_hq")
gate_from(parsed2, "parity_layer_h", "g_s5jit_layer_h")
gate_from(parsed2, "parity_tf_h", "g_s5jit_tf_h")
# PTX-JIT semantics identical to SASS: same worst numbers on EVERY parity
# output. Rows are keyed by (kind, shape-or-out) so multi-shape sections
# compare pairwise.
def _jit_identical() -> bool:
    kinds = ("parity_softmax", "parity_glu", "parity_dwconv", "parity_layer",
             "parity_linear", "parity_mha", "parity_tf", "parity_cast",
             "parity_linear_h", "parity_linear_hq", "parity_layer_h",
             "parity_tf_h")

    def index(rows):
        out = {}
        for r in rows:
            key = (r.get("k"), r.get("shape") or r.get("out") or "")
            out[key] = r.get("max_abs", r.get("diff"))
        return out

    a, b = index(parsed), index(parsed2)
    keys = [k for k in a if k[0] in kinds]
    if not keys:
        return False
    for k in keys:
        if k not in b:
            print("[s5k] jit-bitidentical: missing", k)
            return False
        if abs(a[k] - b[k]) > 0:
            print("[s5k] jit-bitidentical: mismatch", k, a[k], b[k])
            return False
    print("[s5k] jit-bitidentical: all", len(keys), "parity rows match")
    return True

REPORT["gates"]["g_s4d_jit_bitidentical"] = _jit_identical()

# verdict
verdict = "s5-green" if all(v for v in REPORT["gates"].values() if isinstance(v, bool)) else "s5-red"
REPORT["verdict"] = verdict
REPORT["wall_s"] = round(time.time() - T0, 1)
with open("/kaggle/working/step5_verdict.json", "w") as f:
    json.dump(REPORT, f, indent=1)
print("[s5k] VERDICT:", verdict, "gates:", REPORT["gates"])
for k in ("timeline", "timeline_tf", "timeline_full", "timeline_h",
          "timeline_h_full"):
    print("[s5k]", k + ":", REPORT.get(k))
print("[s5k] wall", REPORT["wall_s"], "s")
