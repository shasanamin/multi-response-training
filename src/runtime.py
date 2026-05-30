from __future__ import annotations

import os
from pathlib import Path
from typing import Any


SCRATCH_ENV_VARS = {
    "HF_HOME": "hf_home",
    "HF_DATASETS_CACHE": "hf_datasets_cache",
    "TRANSFORMERS_CACHE": "transformers_cache",
    "XDG_CACHE_HOME": "xdg_cache_home",
    "TORCH_HOME": "torch_home",
    "TRITON_CACHE_DIR": "triton_cache_dir",
    "PIP_CACHE_DIR": "pip_cache_dir",
    "TMPDIR": "tmp_dir",
    "MPLCONFIGDIR": "mpl_config_dir",
    "WANDB_DIR": "wandb_dir",
    "WANDB_CACHE_DIR": "wandb_cache_dir",
}


def _paths_block(config: dict[str, Any]) -> dict[str, str]:
    return config.get("paths", {})


def apply_runtime_environment(config: dict[str, Any]) -> dict[str, str]:
    paths = _paths_block(config)
    for env_name, key in SCRATCH_ENV_VARS.items():
        value = paths.get(key)
        if value:
            os.environ[env_name] = str(value)

    token_env = config.get("huggingface", {}).get("token_env")
    if token_env:
        os.environ.setdefault("HF_TOKEN", os.environ.get(token_env, ""))
    return {name: os.environ.get(name, "") for name in SCRATCH_ENV_VARS}


def ensure_runtime_directories(config: dict[str, Any]) -> dict[str, Path]:
    materialized: dict[str, Path] = {}
    for key, value in _paths_block(config).items():
        if not key.endswith(("_root", "_dir", "_path", "_cache")):
            continue
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        materialized[key] = path
    return materialized


def run_root(config: dict[str, Any], run_name: str) -> Path:
    base = Path(config["paths"]["runs_root"])
    path = base / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
