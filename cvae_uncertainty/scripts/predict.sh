#!/usr/bin/env bash
set -euo pipefail

EXP_ID="${EXP_ID:-exp20}"
EPOCH="${EPOCH:-400}"
GPU_ID="${GPU_ID:-0}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

if [[ -n "${WAIT_FOR_PID}" ]]; then
    echo "Waiting for training process ${WAIT_FOR_PID} to finish..."

    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
        echo "$(date) training still running..."
        sleep 300
    done

    echo "Training finished. Starting prediction..."
fi


for iter in $(seq 0 9); do
    echo "============================="
    echo "Predicting fold ${iter}"
    echo "============================="

    sed "s@# FOLD_IDX: 0@FOLD_IDX: ${iter}@" \
        cfgs/${EXP_ID}_gen_ori.yaml > cfgs/${EXP_ID}_gen.yaml

    grep FOLD cfgs/${EXP_ID}_gen.yaml

    sh predict.sh ${EXP_ID}_gen fold_${iter} ${EPOCH} ${GPU_ID} \
        2>&1 | tee logs/${EXP_ID}_predict_fold_${iter}.log

    echo "Fold ${iter} prediction finished."
done
