#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash tools/scripts/setup_kitti_symlinks.sh /path/to/KITTI/object"
    echo
    echo "The KITTI path should contain ImageSets/, training/, and testing/."
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

KITTI_ROOT="$1"
if [[ ! -d "${KITTI_ROOT}" ]]; then
    echo "KITTI path does not exist: ${KITTI_ROOT}" >&2
    exit 1
fi

KITTI_ROOT="$(cd "${KITTI_ROOT}" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
PCDET_KITTI="${DATA_DIR}/kitti"
CVAE_KITTI="${ROOT_DIR}/cvae_uncertainty/kitti"

for required in training testing; do
    if [[ ! -e "${KITTI_ROOT}/${required}" ]]; then
        echo "Missing ${required}/ under ${KITTI_ROOT}" >&2
        exit 1
    fi
done

mkdir -p "${DATA_DIR}"

if [[ -L "${PCDET_KITTI}" ]]; then
    ln -sfnT "${KITTI_ROOT}" "${PCDET_KITTI}"
elif [[ -e "${PCDET_KITTI}" ]]; then
    echo "Keeping existing data/kitti directory: ${PCDET_KITTI}"
    for subdir in training testing; do
        target="${KITTI_ROOT}/${subdir}"
        dest="${PCDET_KITTI}/${subdir}"
        if [[ -L "${dest}" ]]; then
            ln -sfnT "${target}" "${dest}"
        elif [[ -e "${dest}" ]]; then
            echo "Keeping existing data/kitti/${subdir}"
        else
            ln -s "${target}" "${dest}"
        fi
    done

    if [[ ! -e "${PCDET_KITTI}/ImageSets" && -e "${KITTI_ROOT}/ImageSets" ]]; then
        ln -s "${KITTI_ROOT}/ImageSets" "${PCDET_KITTI}/ImageSets"
    fi
else
    ln -s "${KITTI_ROOT}" "${PCDET_KITTI}"
fi

if [[ ! -e "${PCDET_KITTI}/ImageSets" ]]; then
    echo "Warning: data/kitti/ImageSets is missing. Add train.txt, val.txt, and test.txt before creating KITTI infos." >&2
fi

if [[ -d "${CVAE_KITTI}" && ! -L "${CVAE_KITTI}" && -L "${CVAE_KITTI}/kitti" ]] && \
   [[ "$(find "${CVAE_KITTI}" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]]; then
    rm "${CVAE_KITTI}/kitti"
    rmdir "${CVAE_KITTI}"
    ln -s "../data/kitti" "${CVAE_KITTI}"
elif [[ -L "${CVAE_KITTI}" ]]; then
    ln -sfnT "../data/kitti" "${CVAE_KITTI}"
elif [[ -e "${CVAE_KITTI}" ]]; then
    echo "Keeping existing cvae_uncertainty/kitti: ${CVAE_KITTI}"
    echo "If this is not the KITTI directory, remove it and rerun this script."
else
    ln -s "../data/kitti" "${CVAE_KITTI}"
fi

echo "KITTI data ready:"
echo "  detector data root: data/kitti"
echo "  cvae_uncertainty/kitti -> ../data/kitti"
