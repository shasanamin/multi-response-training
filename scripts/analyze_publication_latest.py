from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

TMP_MPL_DIR = Path("/tmp/mrt_matplotlib")
TMP_XDG_DIR = Path("/tmp/mrt_xdg_cache")
TMP_MPL_DIR.mkdir(parents=True, exist_ok=True)
TMP_XDG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(TMP_MPL_DIR)
os.environ["XDG_CACHE_HOME"] = str(TMP_XDG_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import load_yaml


STRATEGY_LABELS = {
    "rkon": "RKoN",
    "bkon": "BKoN",
    "dkon": "DKoN",
    "dbkon": "DBKoN",
}

STRATEGY_COLORS = {
    "rkon": "#1f77b4",
    "bkon": "#d62728",
    "dkon": "#2ca02c",
    "dbkon": "#ff7f0e",
}

FINAL_METRICS = [
    ("reference_loss", "Reference loss", "Reference loss", "lower"),
    ("single_reward_mean", "Single reward", "Reward", "higher"),
    ("multi_k16_reward_best_mean", "Best-of-16 reward", "Reward", "higher"),
    ("multi_k16_reference_coverage_mean", "Reference coverage", "Coverage", "higher"),
    ("multi_k16_semantic_diversity_mean", "Semantic diversity", "Diversity", "higher"),
    ("multi_k16_response_words_mean", "Response length", "Words", "neutral"),
]

CHECKPOINT_METRICS = [
    ("training_loss", "Training loss", "Loss", "training"),
    ("reference_loss", "Reference loss", "Reference loss", "checkpoint"),
    ("single_reward_mean", "Single reward", "Reward", "checkpoint"),
    ("multi_k16_reward_best_mean", "Best-of-16 reward", "Reward", "checkpoint"),
    ("multi_k16_reference_coverage_mean", "Reference coverage", "Coverage", "checkpoint"),
    ("multi_k16_semantic_diversity_mean", "Semantic diversity", "Diversity", "checkpoint"),
]

STAGE_ORDER = {
    "gold_full": 0,
    "nectar_full": 1,
    "ultrafeedback_full": 2,
    "gold_compute_matched_equal_examples_v2": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publication-manifest",
        default="configs/experiments/publication_suite/publication_suite_v1_manifest.json",
    )
    parser.add_argument(
        "--compute-matched-manifest",
        default=(
            "configs/experiments/publication_compute_matched_v2/"
            "publication_gold_compute_matched_equal_examples_v2_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/publication_latest_v1",
    )
    return parser.parse_args()


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
        elif pd.api.types.is_integer_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: str(int(value)))
        else:
            formatted[column] = formatted[column].astype(str)

    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_csv_and_markdown(df: pd.DataFrame, csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    markdown_path.write_text(_dataframe_to_markdown(df), encoding="utf-8")


def _flatten_mapping(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        new_prefix = f"{prefix}_{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(value, new_prefix))
        elif isinstance(value, list):
            flattened[new_prefix] = json.dumps(value)
        else:
            flattened[new_prefix] = value
    return flattened


def _load_manifest_experiments(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiments = [dict(experiment) for experiment in manifest["experiments"]]
    for experiment in experiments:
        experiment["manifest_path"] = str(manifest_path.relative_to(PROJECT_ROOT))
        experiment["suite_name"] = manifest.get("suite_name")
    return manifest, experiments


def _load_run_record(experiment: dict[str, Any]) -> dict[str, Any]:
    summary_path = PROJECT_ROOT / "results" / "runs" / experiment["run_name"] / "summary.json"
    checkpoint_path = PROJECT_ROOT / "results" / "runs" / experiment["run_name"] / "checkpoint_metrics.csv"
    training_history_path = PROJECT_ROOT / "results" / "runs" / experiment["run_name"] / "training_history.csv"
    row: dict[str, Any] = dict(experiment)
    row["summary_path"] = str(summary_path)
    row["status"] = "completed" if summary_path.exists() else "missing"
    row["has_checkpoint_metrics"] = bool(checkpoint_path.exists())
    row["has_training_history"] = bool(training_history_path.exists())

    config = load_yaml(experiment["config_path"])
    row.update(_flatten_mapping(config, "cfg"))
    if not summary_path.exists():
        return row

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row["created_at"] = summary.get("created_at")
    row["checkpoint_dir"] = summary.get("checkpoint_dir")
    row.update({f"train_{key}": value for key, value in summary.get("train_metrics", {}).items()})
    row.update(summary.get("evaluation_metrics", {}))
    row.update({f"count_{key}": value for key, value in summary.get("counts", {}).items()})
    return row


def _sort_stage(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["stage_order"] = working["stage_name"].map(lambda value: STAGE_ORDER.get(str(value), 99))
    working = working.sort_values(["stage_order", "dataset_label", "strategy", "k", "gpu_id", "queue_order"]).reset_index(
        drop=True
    )
    return working.drop(columns=["stage_order"])


def _plot_strategy_metric(axis: plt.Axes, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    for strategy, group in df.groupby("strategy"):
        ordered = group.sort_values("k")
        axis.plot(
            ordered["k"],
            ordered[metric],
            marker="o",
            linewidth=2,
            color=STRATEGY_COLORS.get(str(strategy), "#333333"),
            label=STRATEGY_LABELS.get(str(strategy), str(strategy).upper()),
        )
    axis.set_title(title)
    axis.set_xlabel("Training k")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.set_xticks(sorted(df["k"].unique()))


def _plot_stage_bundle(stage_df: pd.DataFrame, output_path: Path, title: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat_axes = axes.flatten()
    for axis, (metric, metric_title, ylabel, _) in zip(flat_axes, FINAL_METRICS):
        if metric not in stage_df.columns:
            axis.set_visible(False)
            continue
        _plot_strategy_metric(axis, stage_df, metric, metric_title, ylabel)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    figure.suptitle(title, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_strategy_bundle(stage_df: pd.DataFrame, strategy: str, output_path: Path, title: str) -> None:
    subset = stage_df[stage_df["strategy"] == strategy].sort_values("k")
    if subset.empty:
        return
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat_axes = axes.flatten()
    for axis, (metric, metric_title, ylabel, _) in zip(flat_axes, FINAL_METRICS):
        if metric not in subset.columns:
            axis.set_visible(False)
            continue
        axis.plot(subset["k"], subset[metric], marker="o", linewidth=2, color=STRATEGY_COLORS.get(strategy, "#333333"))
        axis.set_title(metric_title)
        axis.set_xlabel("Training k")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.set_xticks(sorted(subset["k"].unique()))
    figure.suptitle(title, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_k_bundle(stage_df: pd.DataFrame, k: int, output_path: Path, title: str) -> None:
    subset = stage_df[stage_df["k"] == k].sort_values("strategy")
    if subset.empty:
        return
    strategies = list(subset["strategy"])
    labels = [STRATEGY_LABELS.get(strategy, strategy.upper()) for strategy in strategies]

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat_axes = axes.flatten()
    for axis, (metric, metric_title, ylabel, _) in zip(flat_axes, FINAL_METRICS):
        if metric not in subset.columns:
            axis.set_visible(False)
            continue
        axis.bar(
            labels,
            subset[metric],
            color=[STRATEGY_COLORS.get(strategy, "#333333") for strategy in strategies],
        )
        axis.set_title(metric_title)
        axis.set_xlabel("Recipe")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, axis="y")
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle(title, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _checkpoint_history_for_run(run_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "runs" / run_name / "checkpoint_metrics.csv"
    frame = pd.read_csv(path)
    frame["run_name"] = run_name
    return frame


def _training_history_for_run(run_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / "runs" / run_name / "training_history.csv"
    frame = pd.read_csv(path)
    frame = frame.dropna(subset=["loss"]).copy()
    frame["run_name"] = run_name
    frame = frame.rename(columns={"loss": "training_loss"})
    return frame


def _plot_compute_strategy_checkpoint_bundle(
    completed_stage_df: pd.DataFrame,
    strategy: str,
    output_path: Path,
    title: str,
) -> None:
    stage_runs = completed_stage_df[
        (completed_stage_df["strategy"] == strategy)
        & completed_stage_df["has_checkpoint_metrics"]
        & completed_stage_df["has_training_history"]
    ].copy()
    if stage_runs.empty:
        return

    training_frames: list[pd.DataFrame] = []
    checkpoint_frames: list[pd.DataFrame] = []
    for row in stage_runs.itertuples(index=False):
        training = _training_history_for_run(row.run_name)
        training["k"] = int(row.k)
        training_frames.append(training)

        checkpoints = _checkpoint_history_for_run(row.run_name)
        checkpoints["k"] = int(row.k)
        checkpoint_frames.append(checkpoints)

    training_df = pd.concat(training_frames, ignore_index=True)
    checkpoint_df = pd.concat(checkpoint_frames, ignore_index=True)

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat_axes = axes.flatten()
    for axis, (metric, metric_title, ylabel, source_kind) in zip(flat_axes, CHECKPOINT_METRICS):
        source_df = training_df if source_kind == "training" else checkpoint_df
        if metric not in source_df.columns:
            axis.set_visible(False)
            continue
        for k_value, group in source_df.groupby("k"):
            ordered = group.sort_values("step")
            axis.plot(
                ordered["step"],
                ordered[metric],
                marker="o",
                linewidth=2,
                label=f"k={int(k_value)}",
            )
        axis.set_title(metric_title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    figure.suptitle(title, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _best_row(stage_df: pd.DataFrame, metric: str, direction: str) -> pd.Series | None:
    if stage_df.empty or metric not in stage_df.columns:
        return None
    if direction == "lower":
        return stage_df.loc[stage_df[metric].idxmin()]
    if direction == "higher":
        return stage_df.loc[stage_df[metric].idxmax()]
    return None


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    publication_manifest, publication_experiments = _load_manifest_experiments(PROJECT_ROOT / args.publication_manifest)
    compute_manifest, compute_experiments = _load_manifest_experiments(PROJECT_ROOT / args.compute_matched_manifest)

    experiments = publication_experiments + compute_experiments
    all_rows = [_load_run_record(experiment) for experiment in experiments]
    all_df = _sort_stage(pd.DataFrame(all_rows))
    completed_df = all_df[all_df["status"] == "completed"].copy()

    status_df = (
        all_df.groupby(["stage_name", "dataset_label", "status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["stage_name", "dataset_label", "status"])
        .reset_index(drop=True)
    )
    _write_csv_and_markdown(
        status_df,
        tables_dir / "run_status.csv",
        tables_dir / "run_status.md",
    )

    _write_csv_and_markdown(
        all_df,
        tables_dir / "all_run_metrics.csv",
        tables_dir / "all_run_metrics.md",
    )

    report_lines = [
        "# Latest Publication Analysis",
        "",
        f"- Publication manifest: `{args.publication_manifest}`",
        f"- Corrected compute-matched manifest: `{args.compute_matched_manifest}`",
        f"- Publication-suite rows: `{len(publication_experiments)}`",
        f"- Compute-matched rows: `{len(compute_experiments)}`",
        f"- Completed rows currently available: `{len(completed_df)}`",
        "",
        "## Stage Status",
        "",
        _dataframe_to_markdown(status_df),
        "",
    ]

    for stage_name, stage_df in completed_df.groupby("stage_name"):
        stage_df = stage_df.sort_values(["strategy", "k"]).reset_index(drop=True)
        stage_table = stage_df[
            [
                "run_name",
                "dataset_label",
                "strategy",
                "k",
                "reference_loss",
                "single_reward_mean",
                "multi_k16_reward_best_mean",
                "multi_k16_reference_coverage_mean",
                "multi_k16_semantic_diversity_mean",
                "multi_k16_response_words_mean",
                "has_checkpoint_metrics",
            ]
        ].copy()
        _write_csv_and_markdown(
            stage_table,
            tables_dir / f"{stage_name}_metrics.csv",
            tables_dir / f"{stage_name}_metrics.md",
        )

        stage_title = f"{stage_name} final metrics"
        _plot_stage_bundle(stage_df, plots_dir / f"{stage_name}_final_bundle.png", stage_title)

        for strategy in sorted(stage_df["strategy"].unique()):
            _plot_strategy_bundle(
                stage_df,
                strategy,
                plots_dir / f"{stage_name}_{strategy}_bundle.png",
                f"{stage_name} {STRATEGY_LABELS.get(strategy, strategy.upper())} k-sweep",
            )

        for k_value in sorted(stage_df["k"].unique()):
            _plot_k_bundle(
                stage_df,
                int(k_value),
                plots_dir / f"{stage_name}_k{k_value}_bundle.png",
                f"{stage_name} k={int(k_value)} recipe comparison",
            )

        if stage_name == "gold_compute_matched_equal_examples_v2":
            for strategy in sorted(stage_df["strategy"].unique()):
                _plot_compute_strategy_checkpoint_bundle(
                    stage_df,
                    strategy,
                    plots_dir / f"{stage_name}_{strategy}_checkpoint_bundle.png",
                    f"{stage_name} {STRATEGY_LABELS.get(strategy, strategy.upper())} checkpoint dynamics",
                )

        report_lines.extend(
            [
                f"## {stage_name}",
                "",
                f"- Completed runs in this stage: `{len(stage_df)}`",
            ]
        )
        for metric, _, _, direction in FINAL_METRICS:
            best = _best_row(stage_df, metric, direction)
            if best is None:
                continue
            relation = "lowest" if direction == "lower" else "highest"
            report_lines.append(
                f"- {metric}: {relation} at `{best['run_name']}` with `{best[metric]:.4f}`."
            )
        report_lines.extend(
            [
                "",
                _dataframe_to_markdown(stage_table),
                "",
            ]
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir.relative_to(PROJECT_ROOT)), "completed_runs": len(completed_df)}, indent=2))


if __name__ == "__main__":
    main()
