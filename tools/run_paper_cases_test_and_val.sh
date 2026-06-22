#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"

"$PYTHON_BIN" tools/run_paper_cases_test.py \
  --save_dir visualization_outputs/vis_full_uncertainty_test_paper_cases \
  "$@"

"$PYTHON_BIN" tools/run_paper_cases_val.py \
  --save_dir visualization_outputs/vis_full_uncertainty_val_paper_cases \
  "$@"
