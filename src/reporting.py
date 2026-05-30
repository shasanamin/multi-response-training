from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from paths import PROJECT_ROOT


def project_results_root() -> Path:
    path = PROJECT_ROOT / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_logs_root() -> Path:
    path = PROJECT_ROOT / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_results_dir(run_name: str) -> Path:
    path = project_results_root() / "runs" / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy_if_exists(source: str | Path | None, destination: Path) -> None:
    if not source:
        return
    source_path = Path(source)
    if not source_path.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(next_prefix, item, output)
        return
    output[prefix] = value


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    _flatten("", summary, flattened)
    return flattened


def _sorted_checkpoint_state_paths(checkpoint_dir: str | Path) -> list[Path]:
    root = Path(checkpoint_dir)
    state_paths = list(root.glob("checkpoint-*/trainer_state.json"))
    return sorted(
        state_paths,
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )


def load_training_history(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    state_paths = _sorted_checkpoint_state_paths(checkpoint_dir)
    if not state_paths:
        return []
    latest_state = state_paths[-1]
    payload = json.loads(latest_state.read_text(encoding="utf-8"))
    history = payload.get("log_history", [])
    if isinstance(history, list):
        return [item for item in history if isinstance(item, dict)]
    return []


def _write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _maybe_plot_training_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    train_rows = [row for row in history if "loss" in row and "step" in row]
    eval_rows = [row for row in history if "eval_loss" in row and "step" in row]
    if not train_rows and not eval_rows:
        return

    plt.figure(figsize=(7, 4))
    if train_rows:
        plt.plot(
            [row["step"] for row in train_rows],
            [row["loss"] for row in train_rows],
            marker="o",
            label="train_loss",
        )
    if eval_rows:
        plt.plot(
            [row["step"] for row in eval_rows],
            [row["eval_loss"] for row in eval_rows],
            marker="s",
            label="eval_loss",
        )
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()


def _maybe_plot_checkpoint_metrics(checkpoint_metrics_csv: Path, output_dir: Path) -> None:
    if not checkpoint_metrics_csv.exists():
        return
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    frame = pd.read_csv(checkpoint_metrics_csv)
    if frame.empty or "step" not in frame.columns:
        return

    metric_specs = [
        ("reference_loss", "checkpoint_reference_loss.png", "Reference Loss"),
        ("single_reward_mean", "checkpoint_single_reward.png", "Single-Sample Reward"),
        ("multi_reward_best_mean", "checkpoint_multi_reward.png", "Multi-Sample Reward"),
        ("multi_k4_reward_best_mean", "checkpoint_multi_k4_reward.png", "Multi-k=4 Reward"),
    ]
    for column, filename, title in metric_specs:
        if column not in frame.columns:
            continue
        series = frame[["step", column]].dropna()
        if series.empty:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(series["step"], series[column], marker="o")
        plt.xlabel("Checkpoint Step")
        plt.ylabel(column)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


def write_project_run_artifacts(summary: dict[str, Any]) -> Path:
    run_name = str(summary["run_name"])
    output_dir = _run_results_dir(run_name)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _copy_if_exists(summary.get("selection_manifest_path"), output_dir / "selection_manifest.jsonl")
    _copy_if_exists(summary.get("config_path"), output_dir / "experiment_config.yaml")

    merged_config_path = Path(summary["checkpoint_dir"]).parent / "merged_config.yaml"
    _copy_if_exists(merged_config_path, output_dir / "merged_config.yaml")

    evaluation_dir = Path(summary["checkpoint_dir"]).parent / "evaluation"
    for artifact_name in (
        "single_generations.jsonl",
        "multi_generations.jsonl",
        "multi_k2_generations.jsonl",
        "multi_k4_generations.jsonl",
        "multi_k8_generations.jsonl",
        "multi_k16_generations.jsonl",
    ):
        _copy_if_exists(evaluation_dir / artifact_name, output_dir / artifact_name)

    history = load_training_history(summary["checkpoint_dir"])
    if history:
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        _write_history_csv(history, output_dir / "training_history.csv")
        _maybe_plot_training_history(history, output_dir)
        _copy_if_exists(Path(summary["checkpoint_dir"]) / "checkpoint_metrics.csv", output_dir / "checkpoint_metrics.csv")
        _copy_if_exists(Path(summary["checkpoint_dir"]) / "checkpoint_metrics.jsonl", output_dir / "checkpoint_metrics.jsonl")
        _maybe_plot_checkpoint_metrics(output_dir / "checkpoint_metrics.csv", output_dir)

    return output_dir


def rebuild_summary_index() -> Path:
    runs_root = project_results_root() / "runs"
    summaries = sorted(runs_root.glob("*/summary.json"))
    rows = [flatten_summary(json.loads(path.read_text(encoding="utf-8"))) for path in summaries]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    index_path = project_results_root() / "run_summaries.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return index_path
