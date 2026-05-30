from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

TMP_MPL_DIR = Path("/tmp/mrt_matplotlib_reward_compare")
TMP_XDG_DIR = Path("/tmp/mrt_xdg_reward_compare")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rescore-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-filter", default=None)
    parser.add_argument("--dataset-filter", default=None)
    parser.add_argument("--title", default="Alternative reward-model comparison")
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


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        experiments = [dict(experiment) for experiment in payload.get("experiments", [])]
    elif isinstance(payload, list):
        experiments = [dict(experiment) for experiment in payload]
    else:
        raise TypeError(f"Unsupported manifest type: {type(payload)!r}")

    for experiment in experiments:
        config = load_yaml(experiment["config_path"])
        experiment.setdefault("model_key", config.get("model", {}).get("key") or config.get("model", {}).get("hf_id"))
        experiment.setdefault("dataset_label", config.get("data", {}).get("source", "unknown_dataset"))
        experiment.setdefault("stage_name", config.get("run", {}).get("name", "unknown_stage"))
    return experiments


def _load_summary_row(experiment: dict[str, Any]) -> dict[str, Any]:
    summary_path = PROJECT_ROOT / "results" / "runs" / experiment["run_name"] / "summary.json"
    row = dict(experiment)
    row["summary_path"] = str(summary_path)
    row["status"] = "completed" if summary_path.exists() else "missing"
    if not summary_path.exists():
        return row
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("evaluation_metrics", {})
    row["old_reward_model_name"] = (
        load_yaml(experiment["config_path"]).get("evaluation", {}).get("reward_model", {}).get("model_name")
    )
    for key in [
        "single_reward_mean",
        "single_reward_best_mean",
        "multi_k16_reward_mean",
        "multi_k16_reward_best_mean",
        "multi_reward_mean",
        "multi_reward_best_mean",
    ]:
        row[f"old_{key}"] = metrics.get(key)
    return row


def _plot_reward_scatter(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    for strategy, group in frame.groupby("strategy"):
        axis.scatter(
            group["old_multi_k16_reward_best_mean"],
            group["multi_k16_reward_best_mean"],
            color=STRATEGY_COLORS.get(str(strategy), "#333333"),
            label=STRATEGY_LABELS.get(str(strategy), str(strategy).upper()),
            s=60,
            alpha=0.9,
        )
        for row in group.itertuples(index=False):
            axis.annotate(f"{STRATEGY_LABELS.get(row.strategy, row.strategy)}-{int(row.k)}", (row.old_multi_k16_reward_best_mean, row.multi_k16_reward_best_mean), fontsize=7, xytext=(4, 4), textcoords="offset points")
    lower = min(frame["old_multi_k16_reward_best_mean"].min(), frame["multi_k16_reward_best_mean"].min())
    upper = max(frame["old_multi_k16_reward_best_mean"].max(), frame["multi_k16_reward_best_mean"].max())
    axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1, color="#444444")
    axis.set_xlabel("Original best-of-16 reward")
    axis.set_ylabel("Alt-RM best-of-16 reward")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    experiments = _load_manifest(PROJECT_ROOT / args.manifest)
    rows = [_load_summary_row(experiment) for experiment in experiments]
    summary_df = pd.DataFrame(rows)
    rescore_df = pd.read_csv(PROJECT_ROOT / args.rescore_csv)
    metadata_columns = [
        "run_name",
        "dataset_label",
        "stage_name",
        "status",
        "old_reward_model_name",
    ]
    # The rescoring summary already contains strategy/k and the old/new reward
    # values, so we only merge in lightweight manifest metadata here.
    for column in ["model_key", "strategy", "k"]:
        if column not in rescore_df.columns and column in summary_df.columns:
            metadata_columns.append(column)
    for column in ["old_single_reward_mean", "old_multi_k16_reward_best_mean", "old_multi_reward_best_mean"]:
        if column not in rescore_df.columns and column in summary_df.columns:
            metadata_columns.append(column)

    metadata_df = summary_df[[column for column in metadata_columns if column in summary_df.columns]].copy()
    merged = metadata_df.merge(rescore_df, on="run_name", how="inner")
    merged = merged[merged["status"] == "completed"].copy()
    if args.stage_filter:
        merged = merged[merged["stage_name"] == args.stage_filter].copy()
    if args.dataset_filter:
        merged = merged[merged["dataset_label"] == args.dataset_filter].copy()

    output_dir = PROJECT_ROOT / args.output_dir
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if merged.empty:
        (output_dir / "report.md").write_text("# Reward-Model Robustness\n\nNo matching runs found.\n", encoding="utf-8")
        return

    if "old_multi_k16_reward_best_mean" not in merged.columns and "old_multi_reward_best_mean" in merged.columns:
        merged["old_multi_k16_reward_best_mean"] = merged["old_multi_reward_best_mean"]
    if "multi_k16_reward_best_mean" not in merged.columns and "multi_reward_best_mean" in merged.columns:
        merged["multi_k16_reward_best_mean"] = merged["multi_reward_best_mean"]

    required_columns = [
        "dataset_label",
        "stage_name",
        "strategy",
        "k",
        "old_reward_model_name",
        "reward_model_name",
        "old_single_reward_mean",
        "single_reward_mean",
        "old_multi_k16_reward_best_mean",
        "multi_k16_reward_best_mean",
    ]
    missing_columns = [column for column in required_columns if column not in merged.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns for robustness analysis: {missing_columns}. "
            f"Available columns: {sorted(merged.columns.tolist())}"
        )

    comparison = merged[
        [
            "run_name",
            "dataset_label",
            "stage_name",
            "strategy",
            "k",
            "old_reward_model_name",
            "reward_model_name",
            "old_single_reward_mean",
            "single_reward_mean",
            "old_multi_k16_reward_best_mean",
            "multi_k16_reward_best_mean",
        ]
    ].copy()
    comparison["delta_single_reward_mean"] = comparison["single_reward_mean"] - comparison["old_single_reward_mean"]
    comparison["delta_multi_k16_reward_best_mean"] = (
        comparison["multi_k16_reward_best_mean"] - comparison["old_multi_k16_reward_best_mean"]
    )
    comparison["strategy"] = comparison["strategy"].map(STRATEGY_LABELS).fillna(comparison["strategy"])
    comparison = comparison.sort_values(["stage_name", "strategy", "k"]).reset_index(drop=True)
    _write_csv_and_markdown(
        comparison,
        tables_dir / "reward_model_comparison.csv",
        tables_dir / "reward_model_comparison.md",
    )

    scatter_frame = merged[
        [
            "run_name",
            "strategy",
            "k",
            "old_multi_k16_reward_best_mean",
            "multi_k16_reward_best_mean",
        ]
    ].dropna()
    _plot_reward_scatter(scatter_frame, plots_dir / "reward_model_scatter.png", args.title)

    report_lines = [
        "# Reward-Model Robustness",
        "",
        f"- Compared `{len(comparison)}` completed runs.",
        f"- Original evaluator: `{comparison['old_reward_model_name'].iloc[0]}`.",
        f"- Alternative evaluator: `{comparison['reward_model_name'].iloc[0]}`.",
    ]
    if len(comparison) > 1:
        corr = comparison["old_multi_k16_reward_best_mean"].corr(
            comparison["multi_k16_reward_best_mean"], method="spearman"
        )
        if pd.notna(corr):
            report_lines.append(f"- Spearman correlation on best-of-16 reward: `{corr:.4f}`.")
    report_lines.extend(
        [
            "",
            "Key artifacts:",
            f"- `{(tables_dir / 'reward_model_comparison.csv').relative_to(PROJECT_ROOT)}`",
            f"- `{(plots_dir / 'reward_model_scatter.png').relative_to(PROJECT_ROOT)}`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
