#!/usr/bin/env bash
set -euo pipefail

CFG_FILE="${CFG_FILE:-./cfgs/kitti_models/GLENet_VR_gaussian.yaml}"

python test_gaussian.py --eval_all --cfg_file "${CFG_FILE}" "$@"
