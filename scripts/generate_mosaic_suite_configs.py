from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import dump_yaml


CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments" / "mosaic_suite"

SKYWORK_REWARD = {
    "model_name": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "batch_size": 8,
    "trust_remote_code": True,
}

MODEL_KEY = "llama31_8b_instruct"
MODEL_LABEL = "llama31_8b_instruct"

FULL_BASE_CONFIG: dict[str, Any] = {
    "data": {
        "source": "local_mosaic",
        "raw_path": "data/mosaic-1k.jsonl",
        "min_responses": 1,
        "train_fraction": 0.86,
        "validation_fraction": 0.04,
        "prompt_variants_per_cluster": 1,
        "eval_prompt_variants_per_cluster": 4,
        "response_mode": "all",
        "eval_response_mode": "all",
        "response_index": 0,
        "max_reference_responses_per_prompt": 4,
    },
    "selection": {
        "strategy": "rkon",
        "n": 4,
        "k": 1,
        "quality_weight_alpha": 1.0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "training": {
        "use_lora": True,
        "max_length": 1536,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 1,
        "learning_rate": 1.2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 100,
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "optim": "adamw_torch",
    },
    "generation": {
        "num_return_sequences_single": 1,
        "num_return_sequences_multi": 8,
        "multi_k_values": [2, 4, 8],
        "max_new_tokens": 192,
        "max_length_for_eval": 1536,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    "evaluation": {
        "eval_batch_size": 1,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "model": {
        "key": MODEL_KEY,
        "trust_remote_code": True,
    },
}

SMOKE_OVERRIDES: dict[str, Any] = {
    "data": {
        "max_train_clusters": 32,
        "max_validation_clusters": 8,
        "max_test_clusters": 8,
    },
    "training": {
        "max_length": 768,
        "max_steps": 1,
        "logging_steps": 1,
        "eval_steps": 1,
        "save_steps": 1,
        "save_total_limit": 1,
    },
    "generation": {
        "num_return_sequences_multi": 2,
        "multi_k_values": [2],
        "max_new_tokens": 48,
        "max_length_for_eval": 768,
    },
}

STRATEGIES: dict[str, dict[str, Any]] = {
    "rkon": {"strategy": "rkon", "alpha": 1.0, "reward": False},
    "bkon": {"strategy": "bkon", "alpha": 1.0, "reward": True},
    "dkon": {"strategy": "dkon", "alpha": 0.0, "reward": False},
    "dbkon": {"strategy": "dbkon", "alpha": 0.75, "reward": True},
}

FULL_RUN_SPECS: list[dict[str, Any]] = [
    # Selector grid on exact prompts: tests RKoN/BKoN/DKoN/DBKoN on MOSAIC.
    {"tag": "selector", "strategy": "rkon", "prompt_variants": 1, "response_mode": "all", "k": 1},
    {"tag": "selector", "strategy": "rkon", "prompt_variants": 1, "response_mode": "all", "k": 2},
    {"tag": "selector", "strategy": "rkon", "prompt_variants": 1, "response_mode": "all", "k": 3},
    {"tag": "selector", "strategy": "rkon", "prompt_variants": 1, "response_mode": "all", "k": 4},
    {"tag": "selector", "strategy": "bkon", "prompt_variants": 1, "response_mode": "all", "k": 1},
    {"tag": "selector", "strategy": "bkon", "prompt_variants": 1, "response_mode": "all", "k": 2},
    {"tag": "selector", "strategy": "bkon", "prompt_variants": 1, "response_mode": "all", "k": 3},
    {"tag": "selector", "strategy": "bkon", "prompt_variants": 1, "response_mode": "all", "k": 4},
    {"tag": "selector", "strategy": "dkon", "prompt_variants": 1, "response_mode": "all", "k": 2},
    {"tag": "selector", "strategy": "dkon", "prompt_variants": 1, "response_mode": "all", "k": 3},
    {"tag": "selector", "strategy": "dkon", "prompt_variants": 1, "response_mode": "all", "k": 4},
    {"tag": "selector", "strategy": "dbkon", "prompt_variants": 1, "response_mode": "all", "k": 2},
    {"tag": "selector", "strategy": "dbkon", "prompt_variants": 1, "response_mode": "all", "k": 3},
    {"tag": "selector", "strategy": "dbkon", "prompt_variants": 1, "response_mode": "all", "k": 4},
    # Matched supervised-pair redundancy tests: p*k=4 for explicit, hybrid, implicit.
    {"tag": "hybrid", "strategy": "rkon", "prompt_variants": 2, "response_mode": "all", "k": 2},
    {"tag": "implicit", "strategy": "rkon", "prompt_variants": 4, "response_mode": "cyclic_one", "k": 1},
    # Negative control: same prompt redundancy but no response-mode diversity.
    {"tag": "same", "strategy": "rkon", "prompt_variants": 4, "response_mode": "same_one", "k": 1},
    # Full lattice upper bound.
    {"tag": "full", "strategy": "rkon", "prompt_variants": 4, "response_mode": "all", "k": 4},
    # Equal-update SRT baseline against the p*k=4 triad.
    {"tag": "equalsteps", "strategy": "rkon", "prompt_variants": 1, "response_mode": "all", "k": 1, "max_steps": 215},
]

SMOKE_RUN_SPECS: list[dict[str, Any]] = [
    {"tag": "smoke_explicit", "strategy": "rkon", "prompt_variants": 1, "response_mode": "all", "k": 4},
    {"tag": "smoke_implicit", "strategy": "rkon", "prompt_variants": 4, "response_mode": "cyclic_one", "k": 1},
    {"tag": "smoke_dbkon", "strategy": "dbkon", "prompt_variants": 1, "response_mode": "all", "k": 2},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "full", "smoke"], default="all")
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def data_stem(spec: dict[str, Any], stage: str) -> str:
    size = "smoke" if stage == "smoke" else "860_40_100"
    return (
        f"pools_trainv{spec['prompt_variants']}_evalv4_"
        f"{spec['response_mode']}_{size}_seed7"
    )


def run_name(spec: dict[str, Any], stage: str) -> str:
    stage_suffix = "smoke" if stage == "smoke" else spec["tag"]
    return (
        f"mosaic_{MODEL_LABEL}_{spec['strategy']}_"
        f"p{spec['prompt_variants']}_k{spec['k']}_{stage_suffix}"
    )


def estimated_cost(spec: dict[str, Any], stage: str) -> float:
    train_multiplier = int(spec["prompt_variants"]) * int(spec["k"])
    if spec["response_mode"] != "all":
        train_multiplier = int(spec["prompt_variants"])
    stage_factor = 0.08 if stage == "smoke" else 1.0
    selector_factor = 1.2 if spec["strategy"] in {"bkon", "dbkon"} else 1.0
    if spec.get("max_steps"):
        train_multiplier = max(train_multiplier, 4)
    return round(float(train_multiplier) * selector_factor * stage_factor, 3)


def build_config(spec: dict[str, Any], stage: str) -> dict[str, Any]:
    config = deepcopy(FULL_BASE_CONFIG)
    if stage == "smoke":
        config = deep_update(config, SMOKE_OVERRIDES)

    strategy_cfg = STRATEGIES[spec["strategy"]]
    config["data"]["prompt_variants_per_cluster"] = int(spec["prompt_variants"])
    config["data"]["response_mode"] = str(spec["response_mode"])
    config["data"]["prepared_path"] = (
        "/scratch/gilbreth/${USER}/data/mrt/mosaic/"
        f"{data_stem(spec, stage)}.jsonl"
    )

    n_candidates = 1 if spec["response_mode"] != "all" else 4
    config["selection"]["strategy"] = strategy_cfg["strategy"]
    config["selection"]["n"] = n_candidates
    config["selection"]["k"] = int(spec["k"])
    config["selection"]["quality_weight_alpha"] = float(strategy_cfg["alpha"])
    if strategy_cfg["reward"]:
        config["selection"]["score_source"] = "reward_model"
        config["selection"]["reward_model"] = deepcopy(SKYWORK_REWARD)
    else:
        config["selection"].pop("score_source", None)
        config["selection"].pop("reward_model", None)

    if spec.get("max_steps"):
        config["training"]["max_steps"] = int(spec["max_steps"])
        config["training"]["eval_steps"] = 100
        config["training"]["save_steps"] = 100

    config["run"] = {"name": run_name(spec, stage), "seed": 7}
    return config


def manifest_entry(config_path: Path, spec: dict[str, Any], stage: str, queue_order: int) -> dict[str, Any]:
    return {
        "run_name": config_path.stem,
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "model_key": MODEL_KEY,
        "strategy": spec["strategy"],
        "k": int(spec["k"]),
        "prompt_variants_per_cluster": int(spec["prompt_variants"]),
        "response_mode": spec["response_mode"],
        "stage": stage,
        "experiment_tag": spec["tag"],
        "gpu_id": 0,
        "queue_order": queue_order,
        "checkpoint_eval": False,
        "estimated_cost": estimated_cost(spec, stage),
    }


def write_suite(suite_name: str, specs: list[dict[str, Any]], stage: str) -> Path:
    experiments = []
    for queue_order, spec in enumerate(specs, start=1):
        config = build_config(spec, stage)
        config_path = CONFIG_DIR / f"{config['run']['name']}.yaml"
        dump_yaml(config_path, config)
        experiments.append(manifest_entry(config_path, spec, stage, queue_order))

    manifest = {
        "suite_name": suite_name,
        "runtime_config": "configs/runtime/cluster.yaml",
        "reward_model_name": SKYWORK_REWARD["model_name"],
        "reward_batch_size": int(SKYWORK_REWARD["batch_size"]),
        "checkpoint_multi_k_values": "2,4,8",
        "experiments": experiments,
    }
    manifest_path = CONFIG_DIR / f"{suite_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if args.stage in {"all", "smoke"}:
        written.append(write_suite("mosaic_llama31_8b_smoke_v1", SMOKE_RUN_SPECS, "smoke"))
    if args.stage in {"all", "full"}:
        written.append(write_suite("mosaic_llama31_8b_full_v1", FULL_RUN_SPECS, "full"))
    for path in written:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
