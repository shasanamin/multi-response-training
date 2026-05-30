from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401

from evaluation import compute_reward_metrics
from jsonl_io import iter_jsonl
from config import load_yaml
from reward import RewardModelScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/experiments/paper_suite/paper_gold_gens_full_v2_manifest.json",
    )
    parser.add_argument("--reward-model-name", required=True)
    parser.add_argument("--reward-batch-size", type=int, default=8)
    parser.add_argument("--reward-model-slug", default=None)
    parser.add_argument("--run-names", default=None)
    parser.add_argument(
        "--output-dir",
        default="results/analysis/paper_suite_v2/skywork_reward_refresh",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "reward_model"


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        manifest = dict(payload)
        experiments = [dict(experiment) for experiment in manifest.get("experiments", [])]
    elif isinstance(payload, list):
        manifest = {"experiments": payload}
        experiments = [dict(experiment) for experiment in payload]
    else:
        raise TypeError(f"Unsupported manifest payload type: {type(payload)!r}")

    for experiment in experiments:
        if "model_key" not in experiment or not experiment.get("model_key"):
            config_path = experiment.get("config_path")
            if config_path:
                config = load_yaml(config_path)
                model_cfg = config.get("model", {})
                experiment["model_key"] = model_cfg.get("key") or model_cfg.get("hf_id") or "unknown_model"
        experiment.setdefault("model_key", "unknown_model")
    manifest["experiments"] = experiments
    return manifest, experiments


def _selected_run_names(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    if args.run_names:
        return [item.strip() for item in args.run_names.split(",") if item.strip()]
    return [experiment["run_name"] for experiment in manifest["experiments"]]


def _generation_files(run_dir: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    ordered_names = [
        "single_generations.jsonl",
        "multi_k2_generations.jsonl",
        "multi_k4_generations.jsonl",
        "multi_k8_generations.jsonl",
        "multi_k16_generations.jsonl",
        "multi_generations.jsonl",
    ]
    for filename in ordered_names:
        path = run_dir / filename
        if path.exists():
            metric_prefix = filename.replace("_generations.jsonl", "")
            files.append((metric_prefix, path))
    return files


def _load_records(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def main() -> None:
    args = parse_args()
    manifest_path = PROJECT_ROOT / args.manifest
    manifest, experiments = _load_manifest(manifest_path)
    experiments_by_name = {experiment["run_name"]: experiment for experiment in experiments}
    run_names = _selected_run_names(args, manifest)

    reward_slug = args.reward_model_slug or _slugify(args.reward_model_name)
    output_dir = PROJECT_ROOT / args.output_dir / reward_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    init_started_at = time.perf_counter()
    scorer = RewardModelScorer(
        args.reward_model_name,
        trust_remote_code=True,
    )
    model_init_elapsed_seconds = time.perf_counter() - init_started_at

    suite_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for index, run_name in enumerate(run_names, start=1):
        experiment = experiments_by_name[run_name]
        run_dir = PROJECT_ROOT / "results" / "runs" / run_name
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        old_metrics = summary.get("evaluation_metrics", {})

        run_started_at = time.perf_counter()
        run_row: dict[str, Any] = {
            "run_name": run_name,
            "model_key": experiment["model_key"],
            "strategy": experiment["strategy"],
            "k": int(experiment["k"]),
            "count_train_pairs": summary["counts"]["train_pairs"],
        }

        timing_rows: list[dict[str, Any]] = []
        total_records = 0
        for metric_prefix, path in _generation_files(run_dir):
            records = _load_records(path)
            if not records:
                continue
            total_records += len(records)
            prompts = [record["prompt"] for record in records]
            responses = [record["response"] for record in records]
            scoring_started_at = time.perf_counter()
            scores = scorer.score(prompts, responses, batch_size=args.reward_batch_size)
            elapsed_seconds = time.perf_counter() - scoring_started_at

            scored_records: list[dict[str, Any]] = []
            for record, score in zip(records, scores):
                enriched = dict(record)
                enriched["reward_score"] = float(score)
                scored_records.append(enriched)

            reward_metrics = compute_reward_metrics(scored_records)
            run_row.update({f"{metric_prefix}_{key}": value for key, value in reward_metrics.items()})
            timing_rows.append(
                {
                    "metric_prefix": metric_prefix,
                    "records": len(records),
                    "elapsed_seconds": elapsed_seconds,
                    "records_per_second": len(records) / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                }
            )

        run_elapsed_seconds = time.perf_counter() - run_started_at
        run_row["reward_model_name"] = args.reward_model_name
        run_row["reward_model_slug"] = reward_slug
        run_row["rescoring_elapsed_seconds"] = run_elapsed_seconds
        run_row["rescoring_records"] = total_records
        run_row["rescoring_records_per_second"] = total_records / run_elapsed_seconds if run_elapsed_seconds > 0 else 0.0

        for key, value in old_metrics.items():
            if "reward" not in key:
                continue
            run_row[f"old_{key}"] = value
            if key in run_row and isinstance(value, (int, float)):
                run_row[f"delta_{key}"] = float(run_row[key]) - float(value)

        per_run_payload = {
            "run_name": run_name,
            "reward_model_name": args.reward_model_name,
            "reward_model_slug": reward_slug,
            "metrics": {
                key: value
                for key, value in run_row.items()
                if key not in {"run_name", "model_key", "strategy", "k", "count_train_pairs"}
            },
            "timings": timing_rows,
        }
        (per_run_dir / f"{run_name}.json").write_text(json.dumps(per_run_payload, indent=2), encoding="utf-8")
        suite_rows.append(run_row)
        print(
            f"[{index}/{len(run_names)}] Rescored {run_name}: "
            f"{total_records} records in {run_elapsed_seconds:.1f}s"
        )

    suite_frame_path = output_dir / "reward_rescoring_summary.csv"
    import pandas as pd

    suite_frame = pd.DataFrame(suite_rows).sort_values(["model_key", "strategy", "k"]).reset_index(drop=True)
    suite_frame.to_csv(suite_frame_path, index=False)

    total_scoring_elapsed_seconds = time.perf_counter() - started_at
    metadata = {
        "manifest": args.manifest,
        "reward_model_name": args.reward_model_name,
        "reward_model_slug": reward_slug,
        "reward_batch_size": args.reward_batch_size,
        "reward_model_init_elapsed_seconds": model_init_elapsed_seconds,
        "num_runs": len(run_names),
        "total_scoring_elapsed_seconds": total_scoring_elapsed_seconds,
        "total_elapsed_including_init_seconds": model_init_elapsed_seconds + total_scoring_elapsed_seconds,
        "suite_summary_csv": str(suite_frame_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(suite_frame_path)


if __name__ == "__main__":
    main()
