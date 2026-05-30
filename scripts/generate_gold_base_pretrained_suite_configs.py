from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import dump_yaml


CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments" / "gold_base_pretrained_suite"

MODEL_BLOCK = {
    "key": "llama31_8b",
    "trust_remote_code": True,
}

SKYWORK_REWARD = {
    "model_name": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "batch_size": 8,
    "trust_remote_code": True,
}

COMMON_GENERATION = {
    "num_return_sequences_single": 1,
    "num_return_sequences_multi": 16,
    "multi_k_values": [2, 4, 6, 8, 16],
    "max_new_tokens": 128,
    "max_length_for_eval": 1024,
    "temperature": 0.8,
    "top_p": 0.95,
}

COMMON_EVALUATION = {
    "eval_batch_size": 2,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "reward_model": dict(SKYWORK_REWARD),
}

FULL_TRAINING = {
    "use_lora": True,
    "max_length": 1024,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "learning_rate": 1.2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "logging_steps": 5,
    "eval_steps": 100,
    "save_steps": 100,
    "save_total_limit": 4,
    "gradient_checkpointing": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "optim": "adamw_torch",
}

SMOKE_TRAINING = {
    "use_lora": True,
    "max_length": 1024,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1,
    "max_steps": 2,
    "learning_rate": 1.2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "logging_steps": 1,
    "eval_steps": 2,
    "save_steps": 2,
    "save_total_limit": 1,
    "gradient_checkpointing": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "optim": "adamw_torch",
}

COMMON_SELECTION = {
    "strategy": "rkon",
    "k": 1,
    "quality_weight_alpha": 1.0,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "score_source": "reward_model",
    "reward_model": dict(SKYWORK_REWARD),
    "n": 64,
}

GOLD_FULL_DATA = {
    "source": "hf_gold_labelled_gens",
    "hf_dataset_id": "tlc4418/gold_labelled_gens",
    "hf_split": "validation",
    "prepared_path": "/scratch/gilbreth/${USER}/data/mrt/gold_labelled_gens/prompt_pools_publication_800_64_100_seed7.jsonl",
    "min_responses": 4,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "max_train_prompts": 800,
    "max_validation_prompts": 64,
    "max_test_prompts": 100,
    "max_reference_responses_per_prompt": 32,
}

GOLD_SMOKE_DATA = {
    "source": "hf_gold_labelled_gens",
    "hf_dataset_id": "tlc4418/gold_labelled_gens",
    "hf_split": "validation",
    "prepared_path": "/scratch/gilbreth/${USER}/data/mrt/gold_labelled_gens/prompt_pools_smoke_48_16_16_seed7.jsonl",
    "min_responses": 4,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "max_train_prompts": 48,
    "max_validation_prompts": 16,
    "max_test_prompts": 16,
    "max_reference_responses_per_prompt": 16,
}

FULL_RUN_SPECS = [
    ("rkon", 1),
    ("rkon", 2),
    ("rkon", 4),
    ("rkon", 8),
    ("rkon", 16),
    ("bkon", 1),
    ("bkon", 2),
    ("bkon", 4),
    ("bkon", 8),
    ("bkon", 16),
    ("dkon", 1),
    ("dkon", 2),
    ("dkon", 4),
    ("dkon", 8),
    ("dkon", 16),
    ("dbkon", 1),
    ("dbkon", 2),
    ("dbkon", 4),
    ("dbkon", 8),
    ("dbkon", 16),
]

SUPPLEMENTAL_K8_RUN_SPECS = [
    ("rkon", 8),
    ("bkon", 8),
    ("dkon", 8),
    ("dbkon", 8),
]

SMOKE_RUN_SPECS = [
    ("rkon", 1),
    ("dbkon", 1),
]

STRATEGY_COST_MULTIPLIER = {
    "rkon": 1.0,
    "bkon": 1.0,
    "dkon": 1.08,
    "dbkon": 1.12,
}


def _deep_copy(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def build_base_config(*, data_block: dict[str, Any], training_block: dict[str, Any], selection_n: int) -> dict[str, Any]:
    config = {
        "data": _deep_copy(data_block),
        "selection": _deep_copy(COMMON_SELECTION),
        "training": _deep_copy(training_block),
        "generation": _deep_copy(COMMON_GENERATION),
        "evaluation": _deep_copy(COMMON_EVALUATION),
        "model": _deep_copy(MODEL_BLOCK),
    }
    config["selection"]["n"] = int(selection_n)
    return config


def build_run_name(strategy: str, k: int, stage: str) -> str:
    suffix = "gold_full" if stage == "full" else "smoke"
    return f"gold_gens_llama31_8b_base_{strategy}_k{k}_{suffix}"


def build_config(*, strategy: str, k: int, stage: str) -> dict[str, Any]:
    if stage == "full":
        config = build_base_config(
            data_block=GOLD_FULL_DATA,
            training_block=FULL_TRAINING,
            selection_n=64,
        )
    else:
        config = build_base_config(
            data_block=GOLD_SMOKE_DATA,
            training_block=SMOKE_TRAINING,
            selection_n=16,
        )
        config["generation"]["num_return_sequences_multi"] = 8
        config["generation"]["multi_k_values"] = [2, 4, 6, 8]
        config["generation"]["max_new_tokens"] = 96
        config["evaluation"]["eval_batch_size"] = 1
        config["evaluation"]["reward_model"]["batch_size"] = 4
        config["selection"]["reward_model"]["batch_size"] = 4

    config["run"] = {"name": build_run_name(strategy, k, stage), "seed": 7}
    config["selection"]["strategy"] = strategy
    config["selection"]["k"] = int(k)
    if strategy == "dkon":
        config["selection"]["quality_weight_alpha"] = 0.0
    elif strategy == "dbkon":
        config["selection"]["quality_weight_alpha"] = 0.75
    else:
        config["selection"]["quality_weight_alpha"] = 1.0
    return config


def estimate_cost(strategy: str, k: int, stage: str) -> float:
    stage_factor = 0.15 if stage == "smoke" else 1.0
    strategy_factor = STRATEGY_COST_MULTIPLIER[strategy]
    selection_factor = 1.0 + 0.04 * float(k)
    return round(stage_factor * strategy_factor * selection_factor, 3)


def write_manifest(*, suite_name: str, manifest_name: str, run_specs: list[tuple[str, int]], stage: str) -> Path:
    experiments: list[dict[str, Any]] = []
    for queue_order, (strategy, k) in enumerate(run_specs, start=1):
        config = build_config(strategy=strategy, k=k, stage=stage)
        config_path = CONFIG_DIR / f"{config['run']['name']}.yaml"
        dump_yaml(config_path, config)
        experiments.append(
            {
                "run_name": config["run"]["name"],
                "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                "model_key": "llama31_8b",
                "model_label": "llama31_8b_base",
                "dataset_label": "gold_gens",
                "stage_name": stage,
                "strategy": strategy,
                "k": k,
                "checkpoint_eval": False,
                "gpu_id": 0,
                "queue_order": queue_order,
                "estimated_cost": estimate_cost(strategy, k, stage),
            }
        )

    manifest = {
        "suite_name": suite_name,
        "runtime_config": "configs/runtime/cluster.yaml",
        "reward_model_name": SKYWORK_REWARD["model_name"],
        "reward_batch_size": SKYWORK_REWARD["batch_size"],
        "checkpoint_multi_k_values": "2,4,6,8,16",
        "experiments": experiments,
    }
    manifest_path = CONFIG_DIR / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    smoke_manifest = write_manifest(
        suite_name="gold_llama31_8b_base_smoke_v1",
        manifest_name="gold_llama31_8b_base_smoke_v1_manifest.json",
        run_specs=SMOKE_RUN_SPECS,
        stage="smoke",
    )
    full_manifest = write_manifest(
        suite_name="gold_llama31_8b_base_full_v1",
        manifest_name="gold_llama31_8b_base_full_v1_manifest.json",
        run_specs=FULL_RUN_SPECS,
        stage="full",
    )
    supplemental_k8_manifest = write_manifest(
        suite_name="gold_llama31_8b_base_k8_patch_v1",
        manifest_name="gold_llama31_8b_base_k8_patch_v1_manifest.json",
        run_specs=SUPPLEMENTAL_K8_RUN_SPECS,
        stage="full",
    )
    print(
        json.dumps(
            {
                "smoke_manifest": str(smoke_manifest.relative_to(PROJECT_ROOT)),
                "full_manifest": str(full_manifest.relative_to(PROJECT_ROOT)),
                "supplemental_k8_manifest": str(supplemental_k8_manifest.relative_to(PROJECT_ROOT)),
                "smoke_runs": len(SMOKE_RUN_SPECS),
                "full_runs": len(FULL_RUN_SPECS),
                "supplemental_k8_runs": len(SUPPLEMENTAL_K8_RUN_SPECS),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
