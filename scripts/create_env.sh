#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${1:-${HOME}/.mrt_scratch/envs/mrt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${ENV_PATH}"
source "${ENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python -m pip install -e "${PROJECT_ROOT}"

echo "Environment ready at ${ENV_PATH}"
