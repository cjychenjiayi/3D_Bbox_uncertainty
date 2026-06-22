#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
CFG_FILE="${CFG_FILE:-cfgs/kitti_models/GLENet_VR_gaussian.yaml}"
CKPT="${CKPT:-../output/kitti_models/GLENet_VR_gaussian/default/ckpt/checkpoint_epoch_80.pth}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-4}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python test_gaussian.py \
    --cfg_file "${CFG_FILE}" \
    --ckpt "${CKPT}" \
    --batch_size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --save_to_file "$@"
