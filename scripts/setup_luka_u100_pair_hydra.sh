#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Environment setup stopped. Preserve its directory and log; do not overwrite it." >&2' ERR
[[ $# == 2 ]]
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE="$(cd -- "$1" && pwd -P)"
ARCHIVE="$(realpath -- "$2")"
ENVROOT="${BASE}/environment_pair_hydra_v1"
test ! -e "${ENVROOT}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 -B -s "${PROJECT}/scripts/luka_u100_pair_transfer.py" verify --archive "${ARCHIVE}" \
    --sha256 2fbb97036df805967c651b9a15a4a14f89ef3a9a75fab1f9e96f1819ecc986f5
mkdir -- "${ENVROOT}"
# Dedicated unmanaged uv and managed Python: no shell profile or other venv edits.
curl --fail --location --proto '=https' https://astral.sh/uv/install.sh --output "${ENVROOT}/uv-install.sh"
env UV_UNMANAGED_INSTALL="${ENVROOT}/tools" sh "${ENVROOT}/uv-install.sh"
export UV_PYTHON_INSTALL_DIR="${ENVROOT}/python" UV_PYTHON_BIN_DIR="${ENVROOT}/python-bin"
export UV_CACHE_DIR="${ENVROOT}/uv-cache" UV_NO_CONFIG=1
"${ENVROOT}/tools/uv" --version >"${ENVROOT}/uv-version.txt"
"${ENVROOT}/tools/uv" python install 3.10.21
"${ENVROOT}/tools/uv" venv --seed --python 3.10.21 "${ENVROOT}/venv"
PYTHON="${ENVROOT}/venv/bin/python"
export VIRTUAL_ENV="${ENVROOT}/venv"
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${PYTHON}" -m pip --isolated install --no-cache-dir 'torch==2.5.1+cu118' \
    --index-url https://download.pytorch.org/whl/cu118
"${PYTHON}" -m pip --isolated install --no-cache-dir --index-url https://pypi.org/simple \
    'numpy==2.2.6' 'awkward==2.13.0' 'uproot==5.7.6' 'scikit-learn==1.7.2' \
    'scipy==1.15.3' 'threadpoolctl==3.6.0' 'weaver-core==0.5.3'
"${PYTHON}" -m pip --isolated check
"${PYTHON}" -m pip --isolated freeze >"${ENVROOT}/pip-freeze.txt"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" environment --archive "${ARCHIVE}" \
    >"${ENVROOT}/environment-check.json"
cat "${ENVROOT}/environment-check.json"
printf '\nEnvironment matched. No jobs submitted. Next: extract the bundle and dry-review Hydra submission.\n'
