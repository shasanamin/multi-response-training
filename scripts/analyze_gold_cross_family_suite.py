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


MODEL_LABELS = {
    "llama32_1b_instruct": "Llama-3.2-1B-Instruct",
    "llama32_3b_instruct": "Llama-3.2-3B-Instruct",
    "qwen35_2b": "Qwen-3.5-2B",
    "qwen35_4b": "Qwen-3.5-4B",
    "gemma4_e2b_it": "Gemma-4-E2B-it",
    "gemma4_e4b_it": "Gemma-4-E4B-it",
}

MODEL_COLORS = {
    "llama32_1b_instruct": "#1f77b4",
    "llama32_3b_instruct": "#0d3b66",
    "qwen35_2b": "#2ca02c",
    "qwen35_4b": "#1b7f3b",
    "gemma4_e2b_it": "#d62728",
    "gemma4_e4b_it": "#8c1d18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/experiments/gold_cross_family_suite/gold_cross_family_rkon_v2_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis/gold_cross_family_suite_v2",
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


def _load_run_record(experiment: dict[str, Any]) -> dict[str, Any]:
    summary_path = PROJECT_ROOT / "results" / "runs" / experiment["run_name"] / "summary.json"
    row: dict[str, Any] = dict(experiment)
    row["summary_path"] = str(summary_path)
    row["status"] = "completed" if summary_path.exists() else "missing"

    config = load_yaml(experiment["config_path"])
    row.update(_flatten_mapping(config, "cfg"))

    if not summary_path.exists():
        return row

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row["created_at"] = summary.get("created_at")
    row.update({f"train_{key}": value for key, value in summary.get("train_metrics", {}).items()})
    row.update(summary.get("evaluation_metrics", {}))
    row.update({f"count_{key}": value for key, value in summary.get("counts", {}).items()})
    return row


def _plot_metric_by_model(
    *,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for model_key, group in df.groupby("model_key"):
        ordered = group.sort_values("k")
        axis.plot(
            ordered["k"],
            ordered[metric],
            marker="o",
            linewidth=2,
            color=MODEL_COLORS.get(model_key, "#333333"),
            label=MODEL_LABELS.get(model_key, model_key),
        )
    axis.set_xlabel("Training k")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xticks(sorted(df["k"].unique()))
    axis.grid(alpha=0.25)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_best_k_improvement(df: pd.DataFrame, output_path: Path) -> None:
    improvement_rows = []
    for model_label, group in df.groupby("model_label"):
        pair = group.set_index("k").sort_index()
        if 1 not in pair.index:
            continue
        best_row = pair.loc[pair["reference_loss"].idxmin()]
        improvement_rows.append(
            {
                "model_label": model_label,
                "best_k": int(best_row.name),
                "reference_loss_improvement": float(pair.loc[1, "reference_loss"] - best_row["reference_loss"]),
            }
        )
    if not improvement_rows:
        return
    improvement_frame = pd.DataFrame(improvement_rows).sort_values("reference_loss_improvement", ascending=False)

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.bar(improvement_frame["model_label"], improvement_frame["reference_loss_improvement"], color="#1f77b4")
    axis.axhline(0.0, color="#444444", linewidth=1)
    axis.set_ylabel("Reference-loss improvement (k=1 - best k)")
    axis.set_title("Gold RKoN improvement from k=1 to best observed k")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    manifest = json.loads((PROJECT_ROOT / args.manifest).read_text(encoding="utf-8"))
    output_dir = PROJECT_ROOT / args.output_dir
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = [_load_run_record(experiment) for experiment in manifest["experiments"]]
    frame = pd.DataFrame(rows)
    frame["model_label"] = frame["model_key"].map(MODEL_LABELS).fillna(frame["model_key"])

    status_table = frame[
        ["run_name", "model_key", "k", "stage", "status", "summary_path"]
    ].sort_values(["model_key", "k"])
    _write_csv_and_markdown(
        status_table,
        tables_dir / "run_status.csv",
        tables_dir / "run_status.md",
    )

    completed = frame[frame["status"] == "completed"].copy()
    if completed.empty:
        report = (
            "# Gold Cross-Family Suite Report\n\n"
            "No completed runs were found yet. Start the suite first, then rerun this analysis.\n"
        )
        (output_dir / "report.md").write_text(report, encoding="utf-8")
        return

    main_table = completed[
        [
            "model_label",
            "model_key",
            "k",
            "reference_loss",
            "multi_k16_reference_coverage_mean",
            "multi_k16_semantic_diversity_mean",
            "multi_k16_reward_best_mean",
            "train_eval_loss",
        ]
    ].rename(
        columns={
            "model_label": "Model",
            "model_key": "model_key",
            "k": "k",
            "reference_loss": "reference_loss",
            "multi_k16_reference_coverage_mean": "coverage_at_16",
            "multi_k16_semantic_diversity_mean": "diversity_at_16",
            "multi_k16_reward_best_mean": "best_of_16_reward",
            "train_eval_loss": "trainer_eval_loss",
        }
    ).sort_values(["Model", "k"])
    _write_csv_and_markdown(
        main_table,
        tables_dir / "cross_family_metrics.csv",
        tables_dir / "cross_family_metrics.md",
    )

    pivot_reference = completed.pivot(index="model_label", columns="k", values="reference_loss").reset_index()
    pivot_reference.columns = [
        "model_label",
        *[f"reference_loss_k{column}" for column in pivot_reference.columns[1:]],
    ]
    _write_csv_and_markdown(
        pivot_reference,
        tables_dir / "reference_loss_by_model.csv",
        tables_dir / "reference_loss_by_model.md",
    )

    delta_rows = []
    k1_to_k4_rows = []
    for model_key, group in completed.groupby("model_key"):
        pair = group.set_index("k")
        if 1 not in pair.index:
            continue
        best_k = int(pair["reference_loss"].idxmin())
        best_row = pair.loc[best_k]
        delta_rows.append(
            {
                "model_key": model_key,
                "model_label": MODEL_LABELS.get(model_key, model_key),
                "best_k_reference_loss": best_k,
                "reference_loss_delta_k1_to_best": float(pair.loc[1, "reference_loss"] - best_row["reference_loss"]),
                "coverage_delta_k1_to_best": float(
                    best_row["multi_k16_reference_coverage_mean"] - pair.loc[1, "multi_k16_reference_coverage_mean"]
                ),
                "diversity_delta_k1_to_best": float(
                    best_row["multi_k16_semantic_diversity_mean"] - pair.loc[1, "multi_k16_semantic_diversity_mean"]
                ),
            }
        )
        if 4 in pair.index:
            k1_to_k4_rows.append(
                {
                    "model_key": model_key,
                    "model_label": MODEL_LABELS.get(model_key, model_key),
                    "reference_loss_delta_k1_to_k4": float(pair.loc[1, "reference_loss"] - pair.loc[4, "reference_loss"]),
                    "coverage_delta_k1_to_k4": float(
                        pair.loc[4, "multi_k16_reference_coverage_mean"] - pair.loc[1, "multi_k16_reference_coverage_mean"]
                    ),
                    "diversity_delta_k1_to_k4": float(
                        pair.loc[4, "multi_k16_semantic_diversity_mean"] - pair.loc[1, "multi_k16_semantic_diversity_mean"]
                    ),
                }
            )
    delta_table = pd.DataFrame(delta_rows).sort_values("reference_loss_delta_k1_to_best", ascending=False)
    if not delta_table.empty:
        _write_csv_and_markdown(
            delta_table,
            tables_dir / "k1_to_best_deltas.csv",
            tables_dir / "k1_to_best_deltas.md",
        )
    k1_to_k4_table = pd.DataFrame(k1_to_k4_rows).sort_values("reference_loss_delta_k1_to_k4", ascending=False)
    if not k1_to_k4_table.empty:
        _write_csv_and_markdown(
            k1_to_k4_table,
            tables_dir / "k1_to_k4_deltas.csv",
            tables_dir / "k1_to_k4_deltas.md",
        )

    _plot_metric_by_model(
        df=completed,
        metric="reference_loss",
        ylabel="Reference loss",
        title="Gold cross-family reference loss",
        output_path=plots_dir / "reference_loss_by_model.png",
    )
    _plot_metric_by_model(
        df=completed,
        metric="multi_k16_reference_coverage_mean",
        ylabel="Reference coverage @ 16",
        title="Gold cross-family reference coverage",
        output_path=plots_dir / "coverage_by_model.png",
    )
    if len(delta_rows) > 0:
        _plot_best_k_improvement(completed, plots_dir / "reference_loss_k1_to_best_improvement.png")

    best_run = completed.loc[completed["reference_loss"].idxmin()]
    report_lines = [
        "# Gold Cross-Family Suite Report",
        "",
        f"- Completed runs: `{len(completed)}/{len(frame)}`.",
        f"- Best reference-loss run so far: `{best_run['run_name']}` at `{best_run['reference_loss']:.4f}`.",
    ]
    if not delta_table.empty:
        for row in delta_table.itertuples(index=False):
            report_lines.append(
                f"- `{row.model_label}`: best k is `{int(row.best_k_reference_loss)}` with reference-loss improvement "
                f"`{row.reference_loss_delta_k1_to_best:.4f}` over k=1 and coverage change `{row.coverage_delta_k1_to_best:.4f}`."
            )
    report_lines.extend(
        [
            "",
            "Key artifacts:",
            f"- `{(tables_dir / 'cross_family_metrics.csv').relative_to(PROJECT_ROOT)}`",
            f"- `{(tables_dir / 'k1_to_best_deltas.csv').relative_to(PROJECT_ROOT)}`" if not delta_table.empty else "",
            f"- `{(plots_dir / 'reference_loss_by_model.png').relative_to(PROJECT_ROOT)}`",
            f"- `{(plots_dir / 'coverage_by_model.png').relative_to(PROJECT_ROOT)}`",
        ]
    )
    report = "\n".join(line for line in report_lines if line)
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
