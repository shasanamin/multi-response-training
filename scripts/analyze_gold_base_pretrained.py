from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT  # noqa: F401


OUT_DIR = PROJECT_ROOT / "paper" / "results" / "gold_base_pretrained"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
STRATEGIES = ["rkon", "bkon", "dkon", "dbkon"]
STRATEGY_LABELS = {
    "rkon": "RKoN",
    "bkon": "BKoN",
    "dkon": "DKoN",
    "dbkon": "DBKoN",
}
K_VALUES = [1, 2, 4, 8, 16]


def load_summary(run_name: str) -> dict[str, Any]:
    path = RUNS_DIR / run_name / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_name(model_label: str, strategy: str, k: int) -> str:
    return f"gold_gens_llama31_8b_{model_label}_{strategy}_k{k}_gold_full"


def metric(summary: dict[str, Any], name: str) -> float:
    return float(summary["evaluation_metrics"][name])


def row_for(model_label: str, strategy: str, k: int) -> dict[str, Any]:
    summary = load_summary(run_name(model_label, strategy, k))
    metrics = summary["evaluation_metrics"]
    counts = summary["counts"]
    return {
        "model": model_label,
        "strategy": STRATEGY_LABELS[strategy],
        "k": k,
        "train_pairs": counts["train_pairs"],
        "reference_loss": metrics["reference_loss"],
        "coverage_at_16": metrics["multi_k16_reference_coverage_mean"],
        "diversity_at_16": metrics["multi_k16_semantic_diversity_mean"],
        "mean_reward_at_16": metrics["multi_k16_reward_mean"],
        "best_reward_at_16": metrics["multi_k16_reward_best_mean"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_model_grid(model_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for k in K_VALUES:
            # DKoN/DBKoN k=1 is equivalent to a single retained response and was
            # run for the base model, but not for the instruct main table.
            if model_label == "instruct" and strategy in {"dkon", "dbkon"} and k == 1:
                continue
            rows.append(row_for(model_label, strategy, k))
    return rows


def build_base_vs_instruct_comparison() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for k in K_VALUES:
            base = row_for("base", strategy, k)
            instruct: dict[str, Any] | None = None
            if not (strategy in {"dkon", "dbkon"} and k == 1):
                instruct = row_for("instruct", strategy, k)
            rows.append(
                {
                    "strategy": STRATEGY_LABELS[strategy],
                    "k": k,
                    "base_reference_loss": base["reference_loss"],
                    "instruct_reference_loss": None if instruct is None else instruct["reference_loss"],
                    "delta_loss_base_minus_instruct": None
                    if instruct is None
                    else base["reference_loss"] - instruct["reference_loss"],
                    "base_coverage_at_16": base["coverage_at_16"],
                    "instruct_coverage_at_16": None if instruct is None else instruct["coverage_at_16"],
                    "base_diversity_at_16": base["diversity_at_16"],
                    "instruct_diversity_at_16": None if instruct is None else instruct["diversity_at_16"],
                    "base_best_reward_at_16": base["best_reward_at_16"],
                    "instruct_best_reward_at_16": None if instruct is None else instruct["best_reward_at_16"],
                }
            )
    return rows


def best_row(rows: list[dict[str, Any]], selector: str, metric_name: str, *, higher: bool) -> dict[str, Any]:
    subset = [row for row in rows if row["strategy"] == selector]
    return sorted(subset, key=lambda row: row[metric_name], reverse=higher)[0]


def build_selector_highlights(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for selector in [STRATEGY_LABELS[strategy] for strategy in STRATEGIES]:
        best_loss = best_row(base_rows, selector, "reference_loss", higher=False)
        best_coverage = best_row(base_rows, selector, "coverage_at_16", higher=True)
        best_diversity = best_row(base_rows, selector, "diversity_at_16", higher=True)
        best_reward = best_row(base_rows, selector, "best_reward_at_16", higher=True)
        highlights.append(
            {
                "strategy": selector,
                "best_loss_k": best_loss["k"],
                "best_reference_loss": best_loss["reference_loss"],
                "best_coverage_k": best_coverage["k"],
                "best_coverage_at_16": best_coverage["coverage_at_16"],
                "best_diversity_k": best_diversity["k"],
                "best_diversity_at_16": best_diversity["diversity_at_16"],
                "best_reward_k": best_reward["k"],
                "best_reward_at_16": best_reward["best_reward_at_16"],
            }
        )
    return highlights


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_rows = build_model_grid("base")
    instruct_rows = build_model_grid("instruct")
    base_columns = [
        "model",
        "strategy",
        "k",
        "train_pairs",
        "reference_loss",
        "coverage_at_16",
        "diversity_at_16",
        "mean_reward_at_16",
        "best_reward_at_16",
    ]
    write_csv(OUT_DIR / "llama31_base_gold_full_summary.csv", base_rows, base_columns)
    write_markdown(OUT_DIR / "llama31_base_gold_full_summary.md", base_rows, base_columns)
    write_csv(OUT_DIR / "llama31_instruct_gold_full_summary.csv", instruct_rows, base_columns)
    write_markdown(OUT_DIR / "llama31_instruct_gold_full_summary.md", instruct_rows, base_columns)

    comparison_rows = build_base_vs_instruct_comparison()
    comparison_columns = [
        "strategy",
        "k",
        "base_reference_loss",
        "instruct_reference_loss",
        "delta_loss_base_minus_instruct",
        "base_coverage_at_16",
        "instruct_coverage_at_16",
        "base_diversity_at_16",
        "instruct_diversity_at_16",
        "base_best_reward_at_16",
        "instruct_best_reward_at_16",
    ]
    write_csv(OUT_DIR / "llama31_base_vs_instruct_gold_full.csv", comparison_rows, comparison_columns)
    write_markdown(OUT_DIR / "llama31_base_vs_instruct_gold_full.md", comparison_rows, comparison_columns)

    highlight_rows = build_selector_highlights(base_rows)
    highlight_columns = [
        "strategy",
        "best_loss_k",
        "best_reference_loss",
        "best_coverage_k",
        "best_coverage_at_16",
        "best_diversity_k",
        "best_diversity_at_16",
        "best_reward_k",
        "best_reward_at_16",
    ]
    write_csv(OUT_DIR / "llama31_base_selector_highlights.csv", highlight_rows, highlight_columns)
    write_markdown(OUT_DIR / "llama31_base_selector_highlights.md", highlight_rows, highlight_columns)

    print(OUT_DIR / "llama31_base_gold_full_summary.csv")
    print(OUT_DIR / "llama31_instruct_gold_full_summary.csv")
    print(OUT_DIR / "llama31_base_vs_instruct_gold_full.csv")
    print(OUT_DIR / "llama31_base_selector_highlights.csv")


if __name__ == "__main__":
    main()
