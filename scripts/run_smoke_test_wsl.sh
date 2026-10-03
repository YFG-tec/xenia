#!/usr/bin/env bash
set -euo pipefail

# PLUTO_XMF smoke test runner (intended for WSL2/Linux)
# Usage (from repo root):
#   bash scripts/run_smoke_test_wsl.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOLVER_DIR="$ROOT_DIR/SSH_220804"
INI_FILE="$ROOT_DIR/examples/smoke_test/p_smoke_test.ini"

cd "$SOLVER_DIR"

echo "[smoke] building..."
make clean
make

echo "[smoke] running..."
./pluto -i "$INI_FILE"

echo "[smoke] PASS (compiled and ran without crashing)"
