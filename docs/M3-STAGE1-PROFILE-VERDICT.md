# M3 Stage 1 — Profile Verdict (2026-09-19, kernel diar-m3-stage1-prof v1)

Engine: real d=512 config (17 conformer + 18 transformer layers, 4 spk,
feat_in 128), DFW1 fp32, short reference audio, whole feed, 36 chunks,
free-tier CPU box. Wall 1439 s ≈ 40 s/chunk (matches the d^2 extrapolation
from bench_engine_synth.py: 30-40 s).

## Per-stage verdict (calls = 36 unless noted)

| stage    | pct     | ms/call  | note                                   |
|----------|---------|----------|----------------------------------------|
| conformer| 92.34 % | 36 839   | **the** target; 17 layers x ~2.17 s    |
| transformer | 6.85 % | 2 734    | 18 layers, d/2 width                   |
| stem     | 0.68 %  | 272      | 8x subsampling conv stack              |
| proj     | 0.07 %  | 29       | d -> d/2                               |
| head     | 0.03 %  | 10       |                                        |
| fe       | 0.02 %  | 175 x2   | mel production (2 segments)            |
| pe/xscale/aosc/concat/trim/gate | ~0.00 % | <=1 | host-side bookkeeping |

top3 coverage 99.87 %, top5 99.97 %.

## Decision (M3-P100-CUDA-PLAN §2)

1. **Port order is settled: the conformer layer, nothing else first.**
   Migrating the 17-layer conformer chain alone removes 92.3 % of the CPU
   wall; the remaining stages stay CPU-side until the conformer is green.
2. Step 3 slice #1 = intra-conformer GEMM families (FF up/down
   [260,512]<->[512,2048], MHA qkv+pos+out projections, pointwise k9 conv
   pair) — exactly the shapes measured in p100_m3_base.
3. transformer (6.9 %) is slice #2; stem/proj/head are free riders once
   GEMM plumbing exists (all are linear/conv families).
4. PE table, xscale, concat, trim, aosc, gate stay CPU-side permanently
   unless profiling after slice #2 says otherwise (all <0.7 %).

## Environment notes

- Profile JSON was written by the gate with a relative path while the
  runner harvests from WORK — cosmetic miss, stdout carries the full
  table (fixed for the next push: absolute --out path).
- GPU baseline (p100_m3_base, T4 by accident — machine_shape must be set
  at kernel creation, new slug p100-m3-base2 created with P100):
  fp32 SGEMM ff_up 0.212 ms / ff_down 0.228 ms / proj 0.043 ms (T4-class;
  P100 numbers pending). Even at T4 fp32 rates, a full conformer chunk is
  ~25 ms of GEMM vs 36 839 ms on CPU — three orders of magnitude headroom
  before any fusion work.
- naive-vs-cuBLAS cross-check had a host-side transpose bug in the
  harness itself (dd[c*g.m+r] vs dd[c+r*g.n]) — fixed in base2; latency
  numbers unaffected.
