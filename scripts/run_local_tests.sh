#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD="$ROOT/.local-build"
rm -rf "$BUILD"
mkdir -p "$BUILD"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/diar.cpp" "$ROOT/src/aosc.cpp" "$ROOT/src/birth_gate.cpp" "$ROOT/src/fe.cpp" "$ROOT/src/nn.cpp" "$ROOT/tests/test_core.cpp" \
  -o "$BUILD/test_core"
"$BUILD/test_core"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/nn.cpp" "$ROOT/tests/test_nn.cpp" \
  -o "$BUILD/test_nn"
"$BUILD/test_nn"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/cublas_layout.cpp" "$ROOT/src/nn.cpp" "$ROOT/src/gguf.cpp" "$ROOT/tests/test_cublas_layout.cpp" \
  -o "$BUILD/test_cublas_layout"
"$BUILD/test_cublas_layout"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/nn.cpp" "$ROOT/src/layers.cpp" "$ROOT/tests/test_layers.cpp" \
  -o "$BUILD/test_layers"
"$BUILD/test_layers"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/nn.cpp" "$ROOT/src/mha.cpp" "$ROOT/tests/test_mha.cpp" \
  -o "$BUILD/test_mha"
"$BUILD/test_mha"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/nn.cpp" "$ROOT/src/conv.cpp" "$ROOT/tests/test_conv.cpp" \
  -o "$BUILD/test_conv"
"$BUILD/test_conv"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/nn.cpp" "$ROOT/src/layers.cpp" "$ROOT/src/mha.cpp" "$ROOT/src/conv.cpp" "$ROOT/src/conformer.cpp" "$ROOT/tests/test_conformer.cpp" \
  -o "$BUILD/test_conformer"
"$BUILD/test_conformer"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/nn.cpp" "$ROOT/src/subsampling.cpp" "$ROOT/tests/test_subsampling.cpp" \
  -o "$BUILD/test_subsampling"
"$BUILD/test_subsampling"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/posenc.cpp" "$ROOT/tests/test_posenc.cpp" \
  -o "$BUILD/test_posenc"
"$BUILD/test_posenc"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/gguf.cpp" "$ROOT/tests/test_gguf.cpp" -o "$BUILD/test_gguf"
"$BUILD/test_gguf"
NN_SRC="$ROOT/src/nn.cpp $ROOT/src/layers.cpp $ROOT/src/mha.cpp $ROOT/src/conv.cpp $ROOT/src/conformer.cpp $ROOT/src/subsampling.cpp $ROOT/src/posenc.cpp"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" -I"$ROOT/tests" "$ROOT/src/gguf.cpp" "$ROOT/src/sortformer.cpp" $NN_SRC "$ROOT/tests/test_sortformer_weights.cpp" -o "$BUILD/test_sortformer_weights"
"$BUILD/test_sortformer_weights"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" -I"$ROOT/tests" "$ROOT/src/gguf.cpp" "$ROOT/src/sortformer.cpp" $NN_SRC "$ROOT/tests/test_sortformer_forward.cpp" -o "$BUILD/test_sortformer_forward"
"$BUILD/test_sortformer_forward"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" -I"$ROOT/tests" "$ROOT/src/gguf.cpp" "$ROOT/src/sortformer.cpp" $NN_SRC "$ROOT/tools/dump_forward.cpp" -o "$BUILD/diar_dump_forward"
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" -I"$ROOT/tests" "$ROOT/src/gguf.cpp" "$ROOT/src/sortformer.cpp" "$ROOT/src/engine.cpp" "$ROOT/src/fe.cpp" "$ROOT/src/diar.cpp" "$ROOT/src/aosc.cpp" "$ROOT/src/birth_gate.cpp" "$ROOT/src/tailfix.cpp" $NN_SRC "$ROOT/tests/test_engine.cpp" -o "$BUILD/test_engine"
"$BUILD/test_engine"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/gguf.cpp" "$ROOT/src/sortformer.cpp" "$ROOT/src/engine.cpp" "$ROOT/src/fe.cpp" "$ROOT/src/diar.cpp" "$ROOT/src/aosc.cpp" "$ROOT/src/birth_gate.cpp" "$ROOT/src/tailfix.cpp" $NN_SRC "$ROOT/tools/k5_runner.cpp" -o "$BUILD/k5_runner"
"$BUILD/k5_runner" --selftest
"$BUILD/k5_runner" --expected-names > /dev/null
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  "$ROOT/tests/probdump_oracle.cpp" -o "$BUILD/probdump_oracle"
"$BUILD/probdump_oracle"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/fe.cpp" "$ROOT/tests/fe_oracle.cpp" -o "$BUILD/fe_oracle"
"$BUILD/fe_oracle"
python3 -m pytest "$ROOT/tests" -q
