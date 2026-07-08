#!/usr/bin/env bash
# Create conda environment "f-stereo" from environment.yml and install ONNX tooling.
#
# Usage:
#   ./scripts/setup_conda_env.sh
#   ./scripts/setup_conda_env.sh --force   # remove existing env and recreate

set -euo pipefail

ENV_NAME="f-stereo"
PIP_CACHE_DIR="/tmp/pip-cache"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/environment.yml"

export PIP_CACHE_DIR
mkdir -p "${PIP_CACHE_DIR}"

configure_pip_cache() {
  echo "Configuring pip cache: ${PIP_CACHE_DIR}"
  conda env config vars set -n "${ENV_NAME}" PIP_CACHE_DIR="${PIP_CACHE_DIR}"
}

install_onnx_packages() {
  echo "Installing ONNX packages ..."
  conda run -n "${ENV_NAME}" pip install onnxruntime-gpu onnxcli onnx
}

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Install Miniconda/Anaconda and ensure 'conda' is on PATH." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${FORCE}" -eq 1 ]]; then
    echo "Removing existing environment '${ENV_NAME}' ..."
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "Environment '${ENV_NAME}' already exists. Updating from ${ENV_FILE} ..."
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
    configure_pip_cache
    install_onnx_packages
    echo
    echo "Done. Activate with: conda activate ${ENV_NAME}"
    exit 0
  fi
fi

echo "Creating environment '${ENV_NAME}' from ${ENV_FILE} (pip cache: ${PIP_CACHE_DIR}) ..."
conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"

configure_pip_cache
install_onnx_packages

echo
echo "Done. Activate with: conda activate ${ENV_NAME}"
