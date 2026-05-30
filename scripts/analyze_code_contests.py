"""Analyze code_contests experiment results and generate publication figures.

Runs:
  - RKoN k=1 (SRT baseline)
  - RKoN k=4, BKoN k=4, DKoN k=4, DBKoN k=4
"""
from __future__ import annotations

import os
from pathlib import Path

TMP_MPL_DIR = Path("/tmp/mrt_matplotlib_cc")
TMP_XDG_DIR = Path("/tmp/mrt_xdg_cc")
TMP_MPL_DIR.mkdir(parents=True, exist_ok=True)
TMP_XDG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(TMP_MPL_DIR)
os.environ["XDG_CACHE_HOME"] = str(TMP_XDG_DIR)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = PROJECT_ROOT / "results" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "code_contests"
PAPER_DIR = PROJECT_ROOT / "paper" / "results"
PASSK_DIR = OUTPUT_DIR / "passk_v1"

RUNS = {
    "RKoN k=1": "code_contests_rkon_k1",
    "BKoN k=1": "code_contests_bkon_k1",
    "RKoN k=2": "code_contests_rkon_k2",
    "BKoN k=2": "code_contests_bkon_k2",
    "DKoN k=2": "code_contests_dkon_k2",
    "DBKoN k=2": "code_contests_dbkon_k2",
    "RKoN k=4": "code_contests_rkon_k4",
    "BKoN k=4": "code_contests_bkon_k4",
    "DKoN k=4": "code_contests_dkon_k4",
    "DBKoN k=4": "code_contests_dbkon_k4",
}

STRATEGY_COLORS = {
    "RKoN": "#4C72B0",
    "BKoN": "#DD8452",
    "DKoN": "#55A868",
    "DBKoN": "#C44E52",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_all() -> dict[str, dict]:
    results = {}
    for label, run_name in RUNS.items():
        path = RESULTS_DIR / run_name / "summary.json"
        if not path.exists():
            print(f"  WARNING: missing {run_name}")
            continue
        with open(path) as f:
            s = json.load(f)
        results[label] = s["evaluation_metrics"]
    return results


def load_training_curves() -> dict[str, list[dict]]:
    """Load training loss history from training_history.json."""
    curves = {}
    for label, run_name in RUNS.items():
        hist_path = RESULTS_DIR / run_name / "training_history.json"
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            curves[label] = json.load(f)
    return curves


def load_passk() -> pd.DataFrame:
    path = PASSK_DIR / "passk_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_main_figure(
    results: dict[str, dict],
    curves: dict[str, list[dict]],
    passk_df: pd.DataFrame,
) -> None:
    """Figure with Code Contests likelihood, dynamics, frontier, and pass@k."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    if passk_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    else:
        fig, axes = plt.subplots(1, 4, figsize=(9.3, 2.8))
    ax1, ax2, ax3 = axes[:3]

    MARKERS = {"RKoN": "o", "BKoN": "s", "DKoN": "D", "DBKoN": "^"}

    # Panel (a): Reference loss vs k per strategy (line plot)
    strat_k_data = {}  # {strat: [(k, ref_loss), ...]}
    for label, em in results.items():
        parts = label.split(" k=")
        if len(parts) != 2:
            continue
        strat = parts[0]
        k = int(parts[1])
        strat_k_data.setdefault(strat, []).append((k, em["reference_loss"]))

    for strat, points in sorted(strat_k_data.items()):
        points.sort()
        ks, losses = zip(*points)
        color = STRATEGY_COLORS.get(strat, "gray")
        marker = MARKERS.get(strat, "o")
        ax1.plot(ks, losses, "-", color=color, marker=marker, markersize=5,
                 label=strat, linewidth=1.5)

    ax1.set_xlabel("Training $k$")
    ax1.set_ylabel("Reference Loss $\\downarrow$")
    ax1.set_title("(a) Ref. Loss vs $k$")
    ax1.set_xticks([1, 2, 4])
    ax1.legend(frameon=False, fontsize=7)

    # Panel (b): Training dynamics — show RKoN k=1,2,4 + all strategies k=4
    show_curves = ["RKoN k=1", "RKoN k=2", "RKoN k=4", "BKoN k=4", "DKoN k=4", "DBKoN k=4"]
    k_styles = {"k=1": ("--", 1.0), "k=2": ("-.", 1.2), "k=4": ("-", 1.5)}
    for label in show_curves:
        if label not in curves:
            continue
        history = curves[label]
        eval_pts = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
        if not eval_pts:
            continue
        steps, losses = zip(*eval_pts)
        strat = label.split(" k=")[0]
        k_str = "k=" + label.split(" k=")[1]
        color = STRATEGY_COLORS.get(strat, "gray")
        ls, lw = k_styles.get(k_str, ("-", 1.5))
        marker = MARKERS.get(strat, "o")
        ax2.plot(steps, losses, ls, color=color, marker=marker, markersize=3,
                 label=label, linewidth=lw)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("(b) Training Dynamics")
    ax2.legend(frameon=False, fontsize=5, loc="upper right", ncol=2)

    # Panel (c): Coverage vs Diversity at k=4 (+ k=1 baselines)
    show_scatter = ["RKoN k=1", "BKoN k=1",
                    "RKoN k=4", "BKoN k=4", "DKoN k=4", "DBKoN k=4"]
    for label in show_scatter:
        if label not in results:
            continue
        em = results[label]
        cov = em["multi_k16_reference_coverage_mean"]
        div = em["multi_k16_semantic_diversity_mean"]
        strat = label.split(" k=")[0]
        k = int(label.split(" k=")[1])
        color = STRATEGY_COLORS.get(strat, "gray")
        marker = MARKERS.get(strat, "o")
        alpha = 0.4 if k == 1 else 0.9
        size = 50 if k == 1 else 80
        ax3.scatter(div, cov, color=color, s=size, marker=marker, zorder=5,
                    edgecolors="white", linewidths=0.5, alpha=alpha)
        short = f"{strat}-{k}"
        ax3.annotate(short, (div, cov), fontsize=5,
                     xytext=(4, 4), textcoords="offset points")

    ax3.set_xlabel("Semantic Diversity $\\uparrow$")
    ax3.set_ylabel("Reference Coverage $\\uparrow$")
    ax3.set_title("(c) Coverage vs Diversity")

    if not passk_df.empty:
        ax4 = axes[3]
        k_columns = [
            ("pass_at_1", 1),
            ("pass_at_2", 2),
            ("pass_at_4", 4),
            ("pass_at_8", 8),
            ("pass_at_16", 16),
        ]
        selected = passk_df[passk_df["run_name"].isin([RUNS[label] for label in ["RKoN k=1", "RKoN k=4", "BKoN k=4", "DKoN k=4", "DBKoN k=4"]])]
        label_to_run = {run_name: label for label, run_name in RUNS.items()}
        for _, row in selected.sort_values(["k", "strategy"]).iterrows():
            xs = [k_value for column, k_value in k_columns if column in row and not pd.isna(row[column])]
            ys = [float(row[column]) for column, _ in k_columns if column in row and not pd.isna(row[column])]
            label = label_to_run.get(row["run_name"], row["run_name"])
            strategy = label.split(" k=")[0]
            ax4.plot(
                xs,
                ys,
                marker=MARKERS.get(strategy, "o"),
                linewidth=1.5,
                color=STRATEGY_COLORS.get(strategy, "gray"),
                label=label,
            )
        ax4.set_xlabel("Inference $k$")
        ax4.set_ylabel("pass@$k$ $\\uparrow$")
        ax4.set_title("(d) Preliminary pass@$k$")
        ax4.set_xticks([1, 2, 4, 8, 16])
        ax4.legend(frameon=False, fontsize=5, loc="lower right", ncol=1)

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(OUTPUT_DIR / f"code_contests_results.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        fig.savefig(PAPER_DIR / f"code_contests_results.{ext}", bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"Figure saved to {OUTPUT_DIR} and {PAPER_DIR}")


def save_csv(results: dict[str, dict], passk_df: pd.DataFrame) -> None:
    """Save results as CSV for reproducibility."""
    import csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        "reference_loss", "reference_perplexity",
        "single_reward_mean", "single_response_words_mean",
        "multi_k16_reward_best_mean", "multi_k16_reward_mean",
        "multi_k16_reference_coverage_mean", "multi_k16_semantic_diversity_mean",
    ]
    passk_columns = [column for column in ["pass_at_1", "pass_at_2", "pass_at_4", "pass_at_8", "pass_at_16"] if column in passk_df.columns]
    passk_by_run = passk_df.set_index("run_name").to_dict(orient="index") if not passk_df.empty else {}

    rows = []
    for label, run_name in RUNS.items():
        if label not in results:
            continue
        em = results[label]
        row = {"run": run_name, "label": label}
        for m in metrics:
            row[m] = em.get(m)
        for column in passk_columns:
            row[column] = passk_by_run.get(run_name, {}).get(column)
        rows.append(row)

    csv_path = OUTPUT_DIR / "code_contests_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "label"] + metrics + passk_columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {csv_path}")


def main():
    print("=== Code Contests Analysis ===\n")
    results = load_all()
    curves = load_training_curves()
    passk_df = load_passk()

    if not results:
        print("No results found.")
        return

    plot_main_figure(results, curves, passk_df)
    save_csv(results, passk_df)
    print("\nDone!")


if __name__ == "__main__":
    main()
