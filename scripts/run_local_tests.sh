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
  "$ROOT/tests/probdump_oracle.cpp" -o "$BUILD/probdump_oracle"
"$BUILD/probdump_oracle"
${CXX:-g++} -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
  -I"$ROOT/include" "$ROOT/src/fe.cpp" "$ROOT/tests/fe_oracle.cpp" -o "$BUILD/fe_oracle"
"$BUILD/fe_oracle"
python3 -m pytest "$ROOT/tests/test_harness.py" "$ROOT/tests/test_sortformer_matrix.py" "$ROOT/tests/test_probdump.py" "$ROOT/tests/test_prob_sweep.py" "$ROOT/tests/test_parity.py" -q
