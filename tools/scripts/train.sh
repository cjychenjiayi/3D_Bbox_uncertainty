#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
CFG_FILE="${CFG_FILE:-./cfgs/kitti_models/GLENet_VR_gaussian.yaml}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" bash scripts/dist_train.sh "${NUM_GPUS}" \
    --cfg_file "${CFG_FILE}" "$@"
