from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import dump_yaml


CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments" / "publication_suite"

SKYWORK_REWARD = {
    "model_name": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "batch_size": 8,
    "trust_remote_code": True,
}

COMMON_GENERATION = {
    "num_return_sequences_single": 1,
    "num_return_sequences_multi": 16,
    "multi_k_values": [2, 4, 8, 16],
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
    "save_steps": 50,
    "save_total_limit": 8,
    "gradient_checkpointing": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "optim": "adamw_torch",
}

COMPUTE_MATCHED_TRAINING = {
    "use_lora": True,
    "max_length": 1024,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 1,
    "max_steps": 50,
    "learning_rate": 1.2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "logging_steps": 5,
    "eval_steps": 25,
    "save_steps": 25,
    "save_total_limit": 4,
    "gradient_checkpointing": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "optim": "adamw_torch",
}

MODEL_BLOCK = {
    "key": "llama31_8b_instruct",
    "trust_remote_code": True,
}

COMMON_SELECTION = {
    "strategy": "rkon",
    "k": 1,
    "quality_weight_alpha": 1.0,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "score_source": "reward_model",
    "reward_model": dict(SKYWORK_REWARD),
}

GOLD_DATA = {
    "source": "hf_gold_labelled_gens",
    "hf_dataset_id": "tlc4418/gold_labelled_gens",
    "hf_split": "validation",
    "prepared_path": "${SCRATCH_ROOT}/data/mrt/gold_labelled_gens/prompt_pools_publication_800_64_100_seed7.jsonl",
    "min_responses": 4,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "max_train_prompts": 800,
    "max_validation_prompts": 64,
    "max_test_prompts": 100,
    "max_reference_responses_per_prompt": 32,
}

NECTAR_DATA = {
    "source": "hf_nectar",
    "hf_dataset_id": "berkeley-nest/Nectar",
    "hf_split": "train",
    "prepared_path": "${SCRATCH_ROOT}/data/mrt/nectar/prompt_pools_publication_1200_128_256_seed7.jsonl",
    "min_responses": 4,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "max_train_prompts": 1200,
    "max_validation_prompts": 128,
    "max_test_prompts": 256,
    "max_reference_responses_per_prompt": 7,
}

ULTRAFEEDBACK_DATA = {
    "source": "hf_ultrafeedback",
    "hf_dataset_id": "openbmb/UltraFeedback",
    "hf_split": "train",
    "prepared_path": "${SCRATCH_ROOT}/data/mrt/ultrafeedback/prompt_pools_publication_1600_128_256_seed7.jsonl",
    "min_responses": 4,
    "train_fraction": 0.8,
    "validation_fraction": 0.1,
    "max_train_prompts": 1600,
    "max_validation_prompts": 128,
    "max_test_prompts": 256,
    "max_reference_responses_per_prompt": 4,
}

FULL_GRID_RUN_SPECS = [
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
    ("dkon", 2),
    ("dkon", 4),
    ("dkon", 8),
    ("dkon", 16),
    ("dbkon", 2),
    ("dbkon", 4),
    ("dbkon", 8),
    ("dbkon", 16),
]

N7_GRID_RUN_SPECS = [
    ("rkon", 1),
    ("rkon", 2),
    ("rkon", 4),
    ("bkon", 1),
    ("bkon", 2),
    ("bkon", 4),
    ("dkon", 2),
    ("dkon", 4),
    ("dbkon", 2),
    ("dbkon", 4),
]

N4_GRID_RUN_SPECS = [
    ("rkon", 1),
    ("rkon", 2),
    ("bkon", 1),
    ("bkon", 2),
    ("dkon", 2),
    ("dbkon", 2),
]

STAGE_SPECS = [
    {
        "stage_name": "gold_full",
        "manifest_name": "publication_gold_llama31_8b_full_v1_manifest.json",
        "dataset_label": "gold_gens",
        "data": GOLD_DATA,
        "selection_n": 64,
        "training": FULL_TRAINING,
        "run_specs": FULL_GRID_RUN_SPECS,
        "reward_multi_k_values": "2,4,8,16",
    },
    {
        "stage_name": "gold_compute_matched",
        "manifest_name": "publication_gold_llama31_8b_compute_matched_v1_manifest.json",
        "dataset_label": "gold_gens",
        "data": GOLD_DATA,
        "selection_n": 64,
        "training": COMPUTE_MATCHED_TRAINING,
        "run_specs": FULL_GRID_RUN_SPECS,
        "reward_multi_k_values": "2,4,8,16",
    },
    {
        "stage_name": "nectar_full",
        "manifest_name": "publication_nectar_llama31_8b_full_v1_manifest.json",
        "dataset_label": "nectar",
        "data": NECTAR_DATA,
        "selection_n": 7,
        "training": FULL_TRAINING,
        "run_specs": N7_GRID_RUN_SPECS,
        "reward_multi_k_values": "2,4,8,16",
    },
    {
        "stage_name": "ultrafeedback_full",
        "manifest_name": "publication_ultrafeedback_llama31_8b_full_v1_manifest.json",
        "dataset_label": "ultrafeedback",
        "data": ULTRAFEEDBACK_DATA,
        "selection_n": 4,
        "training": FULL_TRAINING,
        "run_specs": N4_GRID_RUN_SPECS,
        "reward_multi_k_values": "2,4,8,16",
    },
]

STRATEGY_COST_MULTIPLIER = {
    "rkon": 1.0,
    "bkon": 1.0,
    "dkon": 1.08,
    "dbkon": 1.12,
}

DATASET_COST_MULTIPLIER = {
    "gold_gens": 1.0,
    "nectar": 0.95,
    "ultrafeedback": 0.85,
}


def _deep_copy(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload))


def build_base_config(
    *,
    data_block: dict[str, Any],
    selection_n: int,
    training_block: dict[str, Any],
) -> dict[str, Any]:
    return {
        "data": _deep_copy(data_block),
        "selection": {
            **_deep_copy(COMMON_SELECTION),
            "n": selection_n,
        },
        "training": _deep_copy(training_block),
        "generation": _deep_copy(COMMON_GENERATION),
        "evaluation": _deep_copy(COMMON_EVALUATION),
        "model": _deep_copy(MODEL_BLOCK),
    }


def build_run_name(dataset_label: str, stage_name: str, strategy: str, k: int) -> str:
    if stage_name == "gold_compute_matched":
        return f"{dataset_label}_llama31_8b_instruct_{strategy}_k{k}_compute_matched50_publication"
    return f"{dataset_label}_llama31_8b_instruct_{strategy}_k{k}_{stage_name}"


def build_config(stage_spec: dict[str, Any], strategy: str, k: int) -> dict[str, Any]:
    config = build_base_config(
        data_block=stage_spec["data"],
        selection_n=int(stage_spec["selection_n"]),
        training_block=stage_spec["training"],
    )
    run_name = build_run_name(stage_spec["dataset_label"], stage_spec["stage_name"], strategy, k)
    config["run"] = {"name": run_name, "seed": 7}
    config["selection"]["strategy"] = strategy
    config["selection"]["k"] = k
    if strategy == "dkon":
        config["selection"]["quality_weight_alpha"] = 0.0
    elif strategy == "dbkon":
        config["selection"]["quality_weight_alpha"] = 0.75
    else:
        config["selection"]["quality_weight_alpha"] = 1.0
    return config


def should_checkpoint_eval(stage_name: str, strategy: str, k: int) -> bool:
    if stage_name == "gold_compute_matched":
        return True
    if stage_name == "gold_full":
        return (strategy in {"rkon", "bkon"} and k in {1, 16}) or (strategy in {"dkon", "dbkon"} and k == 16)
    if stage_name == "nectar_full":
        return (strategy in {"rkon", "bkon"} and k in {1, 4}) or (strategy in {"dkon", "dbkon"} and k == 4)
    if stage_name == "ultrafeedback_full":
        return k == 2
    return False


def estimate_cost(stage_name: str, dataset_label: str, strategy: str, k: int, checkpoint_eval: bool) -> float:
    stage_factor = 0.5 if stage_name == "gold_compute_matched" else 1.0
    dataset_factor = DATASET_COST_MULTIPLIER.get(dataset_label, 1.0)
    strategy_factor = STRATEGY_COST_MULTIPLIER.get(strategy, 1.0)
    selection_factor = 1.0 + 0.04 * float(k)
    checkpoint_factor = 1.25 if checkpoint_eval else 1.0
    return round(stage_factor * dataset_factor * strategy_factor * selection_factor * checkpoint_factor, 3)


def experiment_priority(stage_name: str, strategy: str, k: int) -> int:
    preferred_orders = {
        "gold_full": [
            ("rkon", 1),
            ("rkon", 16),
            ("bkon", 1),
            ("bkon", 16),
            ("dbkon", 4),
            ("dkon", 16),
            ("rkon", 4),
            ("bkon", 4),
            ("dbkon", 16),
            ("dkon", 4),
        ],
        "gold_compute_matched": [
            ("rkon", 1),
            ("bkon", 1),
            ("rkon", 16),
            ("bkon", 16),
            ("dbkon", 4),
            ("dkon", 16),
        ],
        "nectar_full": [
            ("rkon", 1),
            ("rkon", 4),
            ("bkon", 1),
            ("bkon", 4),
            ("dbkon", 4),
            ("dkon", 4),
        ],
        "ultrafeedback_full": [
            ("rkon", 1),
            ("rkon", 2),
            ("bkon", 1),
            ("bkon", 2),
            ("dbkon", 2),
            ("dkon", 2),
        ],
    }
    preferred = preferred_orders.get(stage_name, [])
    if (strategy, k) in preferred:
        return preferred.index((strategy, k))
    return 100 + int(k)


def assign_stage_to_gpus(stage_name: str, experiments: list[dict[str, Any]], num_gpus: int = 2) -> dict[int, list[dict[str, Any]]]:
    loads = [0.0 for _ in range(num_gpus)]
    queues: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ordering = sorted(
        experiments,
        key=lambda item: (
            experiment_priority(stage_name, item["strategy"], int(item["k"])),
            -float(item["estimated_cost"]),
        ),
    )
    for experiment in ordering:
        gpu_id = min(range(num_gpus), key=lambda index: loads[index])
        assigned = dict(experiment)
        assigned["gpu_id"] = gpu_id
        queues[gpu_id].append(assigned)
        loads[gpu_id] += float(experiment["estimated_cost"])
    return queues


def write_manifest(path: Path, *, suite_name: str, experiments: list[dict[str, Any]], reward_multi_k_values: str) -> None:
    manifest = {
        "suite_name": suite_name,
        "runtime_config": "configs/runtime/cluster.yaml",
        "reward_model_name": SKYWORK_REWARD["model_name"],
        "reward_batch_size": SKYWORK_REWARD["batch_size"],
        "checkpoint_multi_k_values": reward_multi_k_values,
        "experiments": experiments,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    stage_manifests: list[dict[str, Any]] = []
    combined_gpu_queues: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for stage_spec in STAGE_SPECS:
        stage_experiments: list[dict[str, Any]] = []
        for strategy, k in stage_spec["run_specs"]:
            config = build_config(stage_spec, strategy, k)
            config_path = CONFIG_DIR / f"{config['run']['name']}.yaml"
            dump_yaml(config_path, config)
            checkpoint_eval = should_checkpoint_eval(stage_spec["stage_name"], strategy, k)
            stage_experiments.append(
                {
                    "run_name": config["run"]["name"],
                    "config_path": str(config_path.relative_to(PROJECT_ROOT)),
                    "model_key": config["model"]["key"],
                    "dataset_label": stage_spec["dataset_label"],
                    "stage_name": stage_spec["stage_name"],
                    "strategy": strategy,
                    "k": k,
                    "checkpoint_eval": checkpoint_eval,
                    "estimated_cost": estimate_cost(
                        stage_spec["stage_name"],
                        stage_spec["dataset_label"],
                        strategy,
                        k,
                        checkpoint_eval,
                    ),
                }
            )

        stage_gpu_queues = assign_stage_to_gpus(stage_spec["stage_name"], stage_experiments, num_gpus=2)
        scheduled_stage: list[dict[str, Any]] = []
        for gpu_id in range(2):
            combined_gpu_queues[gpu_id].extend(stage_gpu_queues.get(gpu_id, []))
            scheduled_stage.extend(stage_gpu_queues.get(gpu_id, []))

        manifest_path = CONFIG_DIR / stage_spec["manifest_name"]
        stage_schedule_with_order: list[dict[str, Any]] = []
        for gpu_id in range(2):
            for queue_order, experiment in enumerate(stage_gpu_queues.get(gpu_id, []), start=1):
                scheduled = dict(experiment)
                scheduled["gpu_id"] = gpu_id
                scheduled["queue_order"] = queue_order
                stage_schedule_with_order.append(scheduled)
        write_manifest(
            manifest_path,
            suite_name=stage_spec["stage_name"],
            experiments=stage_schedule_with_order,
            reward_multi_k_values=stage_spec["reward_multi_k_values"],
        )
        stage_manifests.append(
            {
                "stage_name": stage_spec["stage_name"],
                "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
                "experiment_count": len(stage_schedule_with_order),
            }
        )

    combined_schedule: list[dict[str, Any]] = []
    for gpu_id in range(2):
        for queue_order, experiment in enumerate(combined_gpu_queues[gpu_id], start=1):
            scheduled = dict(experiment)
            scheduled["gpu_id"] = gpu_id
            scheduled["queue_order"] = queue_order
            combined_schedule.append(scheduled)

    combined_manifest_path = CONFIG_DIR / "publication_suite_v1_manifest.json"
    write_manifest(
        combined_manifest_path,
        suite_name="publication_suite_v1",
        experiments=combined_schedule,
        reward_multi_k_values="2,4,8,16",
    )

    print(json.dumps(
        {
            "combined_manifest": str(combined_manifest_path.relative_to(PROJECT_ROOT)),
            "stages": stage_manifests,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
