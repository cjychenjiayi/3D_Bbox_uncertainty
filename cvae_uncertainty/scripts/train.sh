#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${EXP_ID:-exp20}"
GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_PORT="${BASE_PORT:-18889}"

mkdir -p logs

for iter in $(seq 0 9); do
    echo "============================="
    echo "Running fold ${iter}"
    echo "============================="

    sed "s@# FOLD_IDX: 0@FOLD_IDX: ${iter}@" \
        cfgs/${EXP_ID}_gen_ori.yaml > cfgs/${EXP_ID}_gen.yaml

    grep FOLD cfgs/${EXP_ID}_gen.yaml

    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    bash scripts/dist_train.sh ${NUM_GPUS} \
        --cfg_file cfgs/${EXP_ID}_gen.yaml \
        --tcp_port ${BASE_PORT} \
        --max_ckpt_save_num 10 \
        --workers 1 \
        --extra_tag fold_${iter} \
        2>&1 | tee logs/${EXP_ID}_gen_fold_${iter}.log

    echo "Fold ${iter} finished."
done
