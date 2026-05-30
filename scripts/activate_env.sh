#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-${HOME}/.mrt_scratch}"
ENV_PATH="${ENV_PATH:-${SCRATCH_ROOT}/envs/mrt}"

export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_ROOT}/cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCRATCH_ROOT}/cache/huggingface/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/cache}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH_ROOT}/cache/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/cache/pip}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/cache/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/cache/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/runs/mrt/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRATCH_ROOT}/cache/wandb}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

mkdir -p \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}" \
  "${MPLCONFIGDIR}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${SCRATCH_ROOT}/data/mrt" \
  "${SCRATCH_ROOT}/runs/mrt"

source "${ENV_PATH}/bin/activate"
