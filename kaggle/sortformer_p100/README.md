# Sortformer v2 P100 test

This is the first real pure-diarization model test. It runs in an ephemeral
Kaggle P100 environment and does not save model weights to the repository.

The script:

1. clones NVIDIA NeMo-Speech.cpp;
2. builds only the CUDA diarization target for `sm_60`;
3. downloads NVIDIA's `diar_streaming_sortformer_4spk-v2.q8_0.gguf` inside Kaggle;
4. generates a deterministic 60-second waveform fixture;
5. runs standalone `nemo-speech diarize`, no ASR model or transcription;
6. records model hash, runtime commit, hardware, build time, end-to-end RTF,
   and RTTM output size.

This is deliberately a smoke/parity-runtime test, not an accuracy benchmark:
the synthetic tone fixture has no reference RTTM. Real DER tests come after a
pinned small fixture and reference outputs are prepared.
