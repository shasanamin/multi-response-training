from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import load_config
from data import load_candidate_pools, pools_by_split, reference_texts_by_prompt, sample_pool_responses
from embeddings import SentenceEmbedder
from evaluation import evaluate_checkpoint
from models import resolve_model_config
from reporting import rebuild_summary_index, write_project_run_artifacts
from reward import RewardModelScorer
from runtime import apply_runtime_environment, ensure_runtime_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-config", default="configs/runtime/cluster.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reward-model-name", default=None)
    parser.add_argument("--reward-batch-size", type=int, default=4)
    parser.add_argument("--multi-k-values", default=None)
    parser.add_argument("--checkpoint-steps", default=None, help="Comma-separated checkpoint steps to evaluate.")
    return parser.parse_args()


def _load_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _checkpoint_directories(checkpoint_root: Path, requested_steps: set[int] | None = None) -> list[Path]:
    checkpoints = sorted(
        checkpoint_root.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if requested_steps:
        checkpoints = [path for path in checkpoints if int(path.name.split("-")[-1]) in requested_steps]
    return checkpoints or [checkpoint_root]


def _checkpoint_step(checkpoint_dir: Path) -> int:
    if checkpoint_dir.name.startswith("checkpoint-"):
        return int(checkpoint_dir.name.split("-")[-1])
    return 0


def _checkpoint_epoch(checkpoint_dir: Path) -> float | None:
    trainer_state = checkpoint_dir / "trainer_state.json"
    if not trainer_state.exists():
        return None
    payload = json.loads(trainer_state.read_text(encoding="utf-8"))
    epoch = payload.get("epoch")
    if isinstance(epoch, (int, float)):
        return float(epoch)
    return None


def _flatten_test_pairs(test_reference_pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pool in test_reference_pools:
        for response in pool["responses"]:
            records.append(
                {
                    "prompt_id": pool["prompt_id"],
                    "prompt": pool["prompt"],
                    "response": response["text"],
                    "score": response.get("score"),
                }
            )
    return records


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.runtime_config)
    apply_runtime_environment(config)
    ensure_runtime_directories(config)

    run_dir = Path(args.run_dir)
    summary = _load_summary(run_dir)

    if args.reward_model_name:
        config.setdefault("evaluation", {})
        config["evaluation"]["reward_model"] = {
            "model_name": args.reward_model_name,
            "batch_size": args.reward_batch_size,
            "trust_remote_code": True,
        }
    if args.multi_k_values:
        config.setdefault("generation", {})
        config["generation"]["multi_k_values"] = [int(value) for value in args.multi_k_values.split(",") if value.strip()]
    requested_steps = None
    if args.checkpoint_steps:
        requested_steps = {int(value) for value in args.checkpoint_steps.split(",") if value.strip()}

    prepared_path = Path(summary["prepared_pool_path"])
    pools = load_candidate_pools(prepared_path)
    grouped = pools_by_split(pools)
    test_pools = grouped["test"]
    max_reference_responses = config["data"].get("max_reference_responses_per_prompt")
    seed = int(config["run"].get("seed", 7))
    test_reference_pools = sample_pool_responses(
        test_pools,
        max_responses_per_prompt=max_reference_responses,
        seed=seed + 202,
    )
    flat_test_pairs = _flatten_test_pairs(test_reference_pools)

    spec, model_cfg = resolve_model_config(config)
    model_cfg.setdefault("cache_dir", config["huggingface"]["cache_dir"])

    checkpoint_root = Path(summary["checkpoint_dir"])
    checkpoint_rows: list[dict[str, Any]] = []
    evaluation_root = run_dir / "evaluation" / "checkpoint_metrics"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    reward_cfg = config.get("evaluation", {}).get("reward_model")
    reward_scorer = None
    if reward_cfg and reward_cfg.get("model_name"):
        reward_scorer = RewardModelScorer(
            reward_cfg["model_name"],
            cache_dir=model_cfg.get("cache_dir"),
            token=None,
            trust_remote_code=reward_cfg.get("trust_remote_code", True),
        )
    embedder = SentenceEmbedder(config["evaluation"].get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))

    for checkpoint_dir in _checkpoint_directories(checkpoint_root, requested_steps=requested_steps):
        metrics = evaluate_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_cfg=model_cfg,
            spec=spec,
            test_pairs=flat_test_pairs,
            test_pools=test_reference_pools,
            references_by_prompt=reference_texts_by_prompt(test_reference_pools),
            generation_cfg=config["generation"],
            evaluation_cfg=config["evaluation"],
            output_dir=evaluation_root / checkpoint_dir.name,
            reward_scorer=reward_scorer,
            embedder=embedder,
        )
        row = {
            "checkpoint": checkpoint_dir.name,
            "step": _checkpoint_step(checkpoint_dir),
            "epoch": _checkpoint_epoch(checkpoint_dir),
        }
        row.update(metrics)
        checkpoint_rows.append(row)

    jsonl_path = checkpoint_root / "checkpoint_metrics.jsonl"
    csv_path = checkpoint_root / "checkpoint_metrics.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in checkpoint_rows:
            handle.write(json.dumps(row) + "\n")

    fieldnames = sorted({key for row in checkpoint_rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in checkpoint_rows:
            writer.writerow(row)

    updated_summary = _load_summary(run_dir)
    write_project_run_artifacts(updated_summary)
    rebuild_summary_index()
    print(csv_path)


if __name__ == "__main__":
    main()
