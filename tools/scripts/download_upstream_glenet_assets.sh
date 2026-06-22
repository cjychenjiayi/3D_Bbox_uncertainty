#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash tools/scripts/download_upstream_glenet_assets.sh [all|metadata|weights]"
}

MODE="${1:-all}"
if [[ "${MODE}" != "all" && "${MODE}" != "metadata" && "${MODE}" != "weights" ]]; then
    usage
    exit 1
fi

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required. Install it with: pip install gdown" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KITTI_DIR="${ROOT_DIR}/data/kitti"
WEIGHT_DIR="${ROOT_DIR}/checkpoints/upstream/glenet"

download_if_missing() {
    local url="$1"
    local out="$2"

    if [[ -f "${out}" ]]; then
        echo "Skip existing: ${out}"
        return
    fi

    mkdir -p "$(dirname "${out}")"
    gdown --fuzzy "${url}" -O "${out}"
}

if [[ "${MODE}" == "all" || "${MODE}" == "metadata" ]]; then
    mkdir -p "${KITTI_DIR}"
    download_if_missing \
        "https://drive.google.com/file/d/1iQl3krptYDBfmLsQFR8xip4Wtl8jj-Uy/view?usp=sharing" \
        "${KITTI_DIR}/kitti_infos_train_wconf_v5.pkl"
    download_if_missing \
        "https://drive.google.com/file/d/1bSmFeO3M4YgXsUG8qSfs_aPoXWXSPX3C/view?usp=sharing" \
        "${KITTI_DIR}/kitti_dbinfos_train_wconf_v5.pkl"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "weights" ]]; then
    mkdir -p "${WEIGHT_DIR}"
    download_if_missing \
        "https://drive.google.com/file/d/1gC-cGRer0X56F1i2AGr4WQar-0NTO6sN/view?usp=sharing" \
        "${WEIGHT_DIR}/GLENet_S.pth"
    download_if_missing \
        "https://drive.google.com/file/d/1wJfE7lDCsLuVhua-OyXpPLYO7rhK15eZ/view?usp=sharing" \
        "${WEIGHT_DIR}/GLENet_C.pth"
    download_if_missing \
        "https://drive.google.com/file/d/1FKZmaD7HCMFJg5TloBzAt_2S7QmQY9fX/view?usp=sharing" \
        "${WEIGHT_DIR}/GLENet_VR.pth"
fi

echo "Upstream GLENet assets are ready."
