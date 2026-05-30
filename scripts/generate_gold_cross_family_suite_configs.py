from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import dump_yaml


CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments" / "gold_cross_family_suite"

SKYWORK_REWARD = {
    "model_name": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "batch_size": 8,
    "trust_remote_code": True,
}


FULL_BASE_CONFIG = {
    "data": {
        "source": "hf_gold_labelled_gens",
        "hf_dataset_id": "tlc4418/gold_labelled_gens",
        "hf_split": "validation",
        "prepared_path": "/scratch/gilbreth/${USER}/data/mrt/gold_labelled_gens/prompt_pools_paper_800_32_100_seed7.jsonl",
        "min_responses": 4,
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "max_train_prompts": 800,
        "max_validation_prompts": 32,
        "max_test_prompts": 100,
        "max_reference_responses_per_prompt": 32,
    },
    "selection": {
        "strategy": "rkon",
        "n": 64,
        "k": 4,
        "quality_weight_alpha": 1.0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "training": {
        "use_lora": True,
        "max_length": 1024,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "learning_rate": 1.5e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "logging_steps": 5,
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
        "num_return_sequences_multi": 16,
        "multi_k_values": [2, 4, 8, 16],
        "max_new_tokens": 128,
        "max_length_for_eval": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    "evaluation": {
        "eval_batch_size": 4,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reward_model": dict(SKYWORK_REWARD),
    },
}

SMOKE_BASE_CONFIG = {
    "data": {
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
    },
    "selection": {
        "strategy": "rkon",
        "n": 16,
        "k": 4,
        "quality_weight_alpha": 1.0,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "training": {
        "use_lora": True,
        "max_length": 1024,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "max_steps": 2,
        "learning_rate": 1.5e-4,
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
    },
    "generation": {
        "num_return_sequences_single": 1,
        "num_return_sequences_multi": 8,
        "multi_k_values": [2, 4, 8],
        "max_new_tokens": 96,
        "max_length_for_eval": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    "evaluation": {
        "eval_batch_size": 2,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "reward_model": dict(SKYWORK_REWARD),
    },
}

MODEL_PROFILES = [
    {
        "model_key": "llama32_1b_instruct",
        "model_label": "llama32_1b_instruct",
        "estimated_cost": 0.45,
        "training_overrides": {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 2,
        },
    },
    {
        "model_key": "llama32_3b_instruct",
        "model_label": "llama32_3b_instruct",
        "estimated_cost": 1.0,
        "training_overrides": {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
        },
    },
    {
        "model_key": "qwen35_2b",
        "model_label": "qwen35_2b",
        "estimated_cost": 0.85,
        "training_overrides": {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
        },
    },
    {
        "model_key": "qwen35_4b",
        "model_label": "qwen35_4b",
        "estimated_cost": 1.25,
        "training_overrides": {
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 8,
        },
    },
    {
        "model_key": "gemma4_e2b_it",
        "model_label": "gemma4_e2b_it",
        "estimated_cost": 0.95,
        "training_overrides": {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
        },
    },
    {
        "model_key": "gemma4_e4b_it",
        "model_label": "gemma4_e4b_it",
        "estimated_cost": 1.35,
        "training_overrides": {
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 8,
        },
    },
]

LEGACY_FULL_K_VALUES = [1, 4]
EXTENDED_FULL_K_VALUES = [1, 2, 4, 8, 16]
SMOKE_K = 4

SUITE_STRATEGIES = {
    "rkon": {
        "strategy": "rkon",
        "quality_weight_alpha": 1.0,
        "selection_reward": False,
        "smoke_suite_name": "gold_cross_family_smoke_v1",
        "legacy_suite_name": "gold_cross_family_rkon_v1",
        "extended_suite_name": "gold_cross_family_rkon_v2",
    },
    "bkon": {
        "strategy": "bkon",
        "quality_weight_alpha": 1.0,
        "selection_reward": True,
        "smoke_suite_name": "gold_cross_family_bkon_smoke_v1",
        "extended_suite_name": "gold_cross_family_bkon_v1",
    },
    "dkon": {
        "strategy": "dkon",
        "quality_weight_alpha": 0.0,
        "selection_reward": True,
        "smoke_suite_name": "gold_cross_family_dkon_smoke_v1",
        "extended_suite_name": "gold_cross_family_dkon_v1",
    },
    "dbkon": {
        "strategy": "dbkon",
        "quality_weight_alpha": 0.75,
        "selection_reward": True,
        "smoke_suite_name": "gold_cross_family_dbkon_smoke_v1",
        "extended_suite_name": "gold_cross_family_dbkon_v1",
    },
}

def build_run_name(model_label: str, strategy: str, k: int, stage: str) -> str:
    suffix = "crossfam_smoke" if stage == "smoke" else "crossfam"
    return f"gold_gens_{model_label}_{strategy}_k{k}_{suffix}"


def build_config(*, base_config: dict, profile: dict, strategy: str, k: int, stage: str) -> dict:
    config = deepcopy(base_config)
    suite_settings = SUITE_STRATEGIES[strategy]
    config["selection"]["strategy"] = suite_settings["strategy"]
    config["selection"]["k"] = k
    config["selection"]["quality_weight_alpha"] = float(suite_settings["quality_weight_alpha"])
    if suite_settings["selection_reward"]:
        config["selection"]["score_source"] = "reward_model"
        config["selection"]["reward_model"] = deepcopy(SKYWORK_REWARD)
    else:
        config["selection"].pop("score_source", None)
        config["selection"].pop("reward_model", None)
    config["training"].update(profile["training_overrides"])
    config["run"] = {"name": build_run_name(profile["model_label"], strategy, k, stage), "seed": 7}
    config["model"] = {"key": profile["model_key"], "trust_remote_code": True}
    return config


def _manifest_entry(
    *,
    config_path: Path,
    profile: dict,
    strategy: str,
    k: int,
    queue_order: int,
    stage: str,
) -> dict:
    stage_factor = 0.2 if stage == "smoke" else 1.0
    return {
        "run_name": config_path.stem,
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "model_key": profile["model_key"],
        "model_label": profile["model_label"],
        "strategy": strategy,
        "k": k,
        "stage": stage,
        "gpu_id": 0,
        "queue_order": queue_order,
        "checkpoint_eval": False,
        "estimated_cost": round(float(profile["estimated_cost"]) * (1.0 + float(k)) * stage_factor, 3),
    }


def _write_suite(
    *,
    suite_name: str,
    base_config: dict,
    strategy: str,
    run_specs: list[tuple[dict, int, str]],
) -> Path:
    experiments = []
    for queue_order, (profile, k, stage) in enumerate(run_specs, start=1):
        config = build_config(
            base_config=base_config,
            profile=profile,
            strategy=strategy,
            k=k,
            stage=stage,
        )
        config_path = CONFIG_DIR / f"{config['run']['name']}.yaml"
        dump_yaml(config_path, config)
        experiments.append(
            _manifest_entry(
                config_path=config_path,
                profile=profile,
                strategy=strategy,
                k=k,
                queue_order=queue_order,
                stage=stage,
            )
        )

    manifest = {
        "suite_name": suite_name,
        "runtime_config": "configs/runtime/cluster.yaml",
        "reward_model_name": base_config["evaluation"]["reward_model"]["model_name"],
        "reward_batch_size": int(base_config["evaluation"]["reward_model"]["batch_size"]),
        "checkpoint_multi_k_values": ",".join(str(value) for value in base_config["generation"]["multi_k_values"]),
        "experiments": experiments,
    }
    manifest_path = CONFIG_DIR / f"{suite_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    smoke_specs = [(profile, SMOKE_K, "smoke") for profile in MODEL_PROFILES]
    legacy_full_specs = [
        (profile, k, "full")
        for profile in MODEL_PROFILES
        for k in LEGACY_FULL_K_VALUES
    ]
    extended_full_specs = [
        (profile, k, "full")
        for profile in MODEL_PROFILES
        for k in EXTENDED_FULL_K_VALUES
    ]

    smoke_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["rkon"]["smoke_suite_name"],
        base_config=SMOKE_BASE_CONFIG,
        strategy="rkon",
        run_specs=smoke_specs,
    )
    legacy_full_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["rkon"]["legacy_suite_name"],
        base_config=FULL_BASE_CONFIG,
        strategy="rkon",
        run_specs=legacy_full_specs,
    )
    extended_full_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["rkon"]["extended_suite_name"],
        base_config=FULL_BASE_CONFIG,
        strategy="rkon",
        run_specs=extended_full_specs,
    )
    bkon_smoke_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["bkon"]["smoke_suite_name"],
        base_config=SMOKE_BASE_CONFIG,
        strategy="bkon",
        run_specs=smoke_specs,
    )
    bkon_full_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["bkon"]["extended_suite_name"],
        base_config=FULL_BASE_CONFIG,
        strategy="bkon",
        run_specs=extended_full_specs,
    )
    dkon_smoke_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["dkon"]["smoke_suite_name"],
        base_config=SMOKE_BASE_CONFIG,
        strategy="dkon",
        run_specs=smoke_specs,
    )
    dkon_full_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["dkon"]["extended_suite_name"],
        base_config=FULL_BASE_CONFIG,
        strategy="dkon",
        run_specs=extended_full_specs,
    )
    dbkon_smoke_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["dbkon"]["smoke_suite_name"],
        base_config=SMOKE_BASE_CONFIG,
        strategy="dbkon",
        run_specs=smoke_specs,
    )
    dbkon_full_manifest = _write_suite(
        suite_name=SUITE_STRATEGIES["dbkon"]["extended_suite_name"],
        base_config=FULL_BASE_CONFIG,
        strategy="dbkon",
        run_specs=extended_full_specs,
    )

    print(smoke_manifest)
    print(legacy_full_manifest)
    print(extended_full_manifest)
    print(bkon_smoke_manifest)
    print(bkon_full_manifest)
    print(dkon_smoke_manifest)
    print(dkon_full_manifest)
    print(dbkon_smoke_manifest)
    print(dbkon_full_manifest)


if __name__ == "__main__":
    main()
