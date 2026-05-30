"""Reproduce every figure in the paper from bundled result artifacts.

This single script regenerates the five figures used in the paper, writing them
(by default into ``results/figures/``) with the exact filenames referenced by the
LaTeX source:

    sim_variance.pdf                     (Fig. "Controlled validation", panel a)
    sim_budget.pdf                       (Fig. "Controlled validation", panel b)
    sim_emse.pdf                         (Fig. "Controlled validation", panel c)
    rkon_gold_cross_family.pdf           (RKoN cross-family scaling)
    publication_gold_full_grouped_bars.pdf (Gold full-data selector grid)

Inputs (all bundled in this archive, no GPU or network required):
    controlled-val/results/exp1_variance.json
    controlled-val/results/exp2_budget.json
    controlled-val/results/exp5_gradient_emse.json
    results/analysis/cross_family_metrics.csv
    results/analysis/publication_all_run_metrics.csv

The simulation JSONs are produced by ``controlled-val/mrt_experiments.py``; the
two CSVs are the aggregated metric tables for the Gold cross-family and Gold
full-data suites (see README for the suite configs that generate them).

Usage:
    python scripts/make_paper_figures.py [--out results/figures]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mrt_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = PROJECT_ROOT / "controlled-val" / "results"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
CROSS_FAMILY_CSV = ANALYSIS_DIR / "cross_family_metrics.csv"
PUBLICATION_CSV = ANALYSIS_DIR / "publication_all_run_metrics.csv"

PALETTE = {
    "blue": "#0072B2",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "gray": "#7A7A7A",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "lines.linewidth": 1.7,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.5,
        "lines.markeredgecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ---------------------------------------------------------------------------
# Controlled-validation figures (panels a, b, c)
# ---------------------------------------------------------------------------
def sim_variance(out_dir: Path) -> None:
    raw = json.loads((SIM_DIR / "exp1_variance.json").read_text())
    ks = np.array(raw["k_values"])
    var = np.array(raw["variance"])
    theory = var[0] * ks[0] / ks
    floor = float(min(var))

    fig, ax = plt.subplots(figsize=(2.5, 2.2), constrained_layout=True)
    ax.plot(ks, var, "o-", color=PALETTE["blue"], label="Empirical")
    ax.plot(ks, theory, "s--", color=PALETTE["vermillion"], alpha=0.85,
            label=r"$\mathcal{O}(1/K)$ ref.")
    ax.axhline(floor, color=PALETTE["gray"], linestyle=":", linewidth=1,
               label=r"$V_{\mathbf{x}}/N_p$ floor")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"$K$")
    ax.set_ylabel(r"$\mathrm{Var}\!\left[\hat{\mathcal{L}}_{\mathrm{MRT}}\right]$")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4,
              labelspacing=0.3, borderpad=0.2)
    _save(fig, out_dir, "sim_variance")


def sim_budget(out_dir: Path) -> None:
    raw = json.loads((SIM_DIR / "exp2_budget.json").read_text())
    ks = np.array(raw["k_values"])
    rm, rs = np.array(raw["random_mean"]), np.array(raw["random_std"])
    dm, ds = np.array(raw["grades_mean"]), np.array(raw["grades_std"])
    k_star = float(raw["k_star"])

    fig, ax = plt.subplots(figsize=(2.5, 2.2), constrained_layout=True)
    ax.errorbar(ks, rm, yerr=rs, marker="o", capsize=2.5, lw=1.4,
                color=PALETTE["blue"], label="RKoN")
    ax.errorbar(ks, dm, yerr=ds, marker="s", capsize=2.5, lw=1.4,
                color=PALETTE["green"], label="GRADES")
    ax.axvline(k_star, color=PALETTE["vermillion"], linestyle="--", linewidth=1,
               label=fr"$K^\star{{=}}{k_star:.1f}$")
    ax.set_xlabel(r"$K$")
    ax.set_ylabel("Test NLL")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4,
              labelspacing=0.3, borderpad=0.2)
    _save(fig, out_dir, "sim_budget")


def sim_emse(out_dir: Path) -> None:
    raw = json.loads((SIM_DIR / "exp5_gradient_emse.json").read_text())
    summary = raw["summary"]
    display = {"Random": "RKoN", "Top-$k$": "BKoN", "GRADES": "GRADES"}
    order = ["Random", "Top-$k$", "GRADES"]
    names = [display[s] for s in order]
    bias_sq = [float(summary[s]["bias_sq"]) for s in order]
    var = [float(summary[s]["var"]) for s in order]
    color = [PALETTE["blue"], PALETTE["vermillion"], PALETTE["green"]]

    fig, ax = plt.subplots(figsize=(2.5, 2.2), constrained_layout=True)
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w / 2, bias_sq, w, color=color, alpha=0.55,
           edgecolor="#222222", linewidth=0.6, label=r"Bias$^2$")
    ax.bar(x + w / 2, var, w, color=color, alpha=0.95,
           edgecolor="#222222", linewidth=0.6, hatch="//", label="Variance")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("EMSE component")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4,
              labelspacing=0.3, borderpad=0.2)
    for xi, s in zip(x, order):
        gc = float(summary[s]["grad_cos"])
        top = max(bias_sq[xi], var[xi])
        ax.text(xi, top * 1.03, fr"$\bar c{{=}}{gc:+.2f}$",
                ha="center", va="bottom", fontsize=6.8)
    ax.set_ylim(0, max(max(bias_sq), max(var)) * 1.25)
    _save(fig, out_dir, "sim_emse")


# ---------------------------------------------------------------------------
# Cross-family RKoN scaling
# ---------------------------------------------------------------------------
FAMILY_COLORS = {
    "Llama-3.2-1B": "#F5D78E",
    "Llama-3.2-3B": "#C8900A",
    "Llama-3.1-8B": "#7A4400",
    "Qwen-3.5-2B": "#C9A8E8",
    "Qwen-3.5-4B": "#7B2D8B",
    "Gemma-4-2B": "#7FCC8A",
    "Gemma-4-4B": "#1A6E2E",
}
FAMILY_MARKERS = {
    "Llama-3.2-1B": "o", "Llama-3.2-3B": "o", "Llama-3.1-8B": "o",
    "Qwen-3.5-2B": "^", "Qwen-3.5-4B": "^",
    "Gemma-4-2B": "D", "Gemma-4-4B": "D",
}
RAW_TO_DISPLAY = {
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen-3.5-2B": "Qwen-3.5-2B",
    "Qwen-3.5-4B": "Qwen-3.5-4B",
    "Gemma-4-E2B-it": "Gemma-4-2B",
    "Gemma-4-E4B-it": "Gemma-4-4B",
}
LEGEND_ROW1 = ["Qwen-3.5-2B", "Qwen-3.5-4B", "Gemma-4-2B", "Gemma-4-4B"]
LEGEND_ROW2 = ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B"]
PLOT_ORDER = LEGEND_ROW1 + LEGEND_ROW2


def rkon_gold_cross_family(out_dir: Path) -> None:
    df_raw = pd.read_csv(CROSS_FAMILY_CSV)[
        ["Model", "k", "reference_loss", "coverage_at_16"]
    ]

    # The Llama-3.1-8B-Instruct RKoN curve comes from the Gold full-data suite.
    pub = pd.read_csv(PUBLICATION_CSV)
    rkon8b = pub[(pub["stage_name"] == "gold_full") & (pub["strategy"] == "rkon")][
        ["k", "reference_loss", "multi_k16_reference_coverage_mean"]
    ].copy()
    rkon8b.rename(columns={"multi_k16_reference_coverage_mean": "coverage_at_16"},
                  inplace=True)
    rkon8b["Model"] = "Llama-3.1-8B-Instruct"
    rkon8b = rkon8b[["Model", "k", "reference_loss", "coverage_at_16"]]

    df_all = pd.concat([df_raw, rkon8b], ignore_index=True)
    df_all["Model_display"] = df_all["Model"].map(RAW_TO_DISPLAY)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), constrained_layout=True)
    ks = [1, 2, 4, 8, 16]
    for ax, ycol, ylabel in [
        (axes[0], "reference_loss", "Loss"),
        (axes[1], "coverage_at_16", "Coverage"),
    ]:
        for model in PLOT_ORDER:
            sub = df_all[df_all["Model_display"] == model].sort_values("k")
            if sub.empty:
                continue
            ax.plot(sub["k"], sub[ycol], color=FAMILY_COLORS[model],
                    marker=FAMILY_MARKERS[model], label=model)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel(r"Responses Per Prompt ($K$)")
        ax.set_ylabel(ylabel)

    handle_map = {l: h for h, l in zip(*axes[0].get_legend_handles_labels())}
    row1 = list(LEGEND_ROW1)
    row2 = list(LEGEND_ROW2) + [""] * (4 - len(LEGEND_ROW2))
    invisible = Line2D([], [], color="none", marker="None", label="")
    ordered_handles, ordered_labels = [], []
    for col in range(4):
        for lbl in (row1[col], row2[col]):
            if lbl and lbl in handle_map:
                ordered_handles.append(handle_map[lbl])
                ordered_labels.append(lbl)
            else:
                ordered_handles.append(invisible)
                ordered_labels.append("")
    fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.20), frameon=False, columnspacing=1.1,
               handletextpad=0.3)
    _save(fig, out_dir, "rkon_gold_cross_family")


# ---------------------------------------------------------------------------
# Gold full-data selector grid (grouped bars)
# ---------------------------------------------------------------------------
STRATEGY_ORDER = ["rkon", "bkon", "dkon", "dbkon"]
STRATEGY_LABELS = {"rkon": "RKoN", "bkon": "BKoN", "dkon": "DKoN", "dbkon": "DBKoN"}
STRATEGY_COLORS = {
    "rkon": "#1f77b4",
    "bkon": "#d62728",
    "dkon": "#e69f00",
    "dbkon": "#2ca02c",
}
FINAL_METRICS = [
    ("reference_loss", "Reference Loss"),
    ("multi_k16_reward_best_mean", "Best-of-16 Reward"),
    ("multi_k16_reference_coverage_mean", "Reference Coverage"),
    ("multi_k16_semantic_diversity_mean", "Semantic Diversity"),
]


def _shade_series(hex_color: str, count: int) -> list[tuple]:
    base = to_rgb(hex_color)
    if count <= 1:
        return [base]
    shades = []
    for i in range(count):
        frac = max(0.0, min(0.8, 0.60 - 0.60 * i / (count - 1)))
        shades.append(tuple(base[c] + (1 - base[c]) * frac for c in range(3)))
    return shades


def _metric_limits(values: pd.Series, pad_frac: float = 0.12) -> tuple[float, float]:
    nums = pd.to_numeric(values, errors="coerce").dropna()
    if nums.empty:
        return (0, 1)
    lo, hi = float(nums.min()), float(nums.max())
    if lo == hi:
        pad = max(abs(lo) * 0.05, 0.01)
        return (lo - pad, hi + pad)
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def publication_gold_full_grouped_bars(out_dir: Path) -> None:
    df = pd.read_csv(PUBLICATION_CSV)
    stage = df[df["stage_name"] == "gold_full"].copy()
    strategies = [s for s in STRATEGY_ORDER if s in set(stage["strategy"])]
    all_k = sorted(stage["k"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (metric, label) in zip(axes, FINAL_METRICS):
        ax.set_ylabel(label)
        ax.set_ylim(*_metric_limits(stage[metric]))
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], fontsize=9)
        for si, strat in enumerate(strategies):
            sdf = stage[stage["strategy"] == strat].sort_values("k")
            k_vals = sdf["k"].tolist()
            shades = _shade_series(STRATEGY_COLORS[strat], len(k_vals))
            bar_w = 0.14 if len(k_vals) >= 4 else 0.18
            offsets = [(j - (len(k_vals) - 1) / 2.0) * bar_w for j in range(len(k_vals))]
            for off, (_, row), shade in zip(offsets, sdf.iterrows(), shades):
                ax.bar(si + off, row[metric], width=bar_w * 0.92, color=shade,
                       edgecolor=STRATEGY_COLORS[strat], linewidth=0.6)

    shade_patches = []
    for k in all_k:
        frac = 0.60 - 0.60 * all_k.index(k) / max(1, len(all_k) - 1)
        shade_patches.append(
            Patch(facecolor=(frac + 0.4 * (1 - frac),) * 3, edgecolor="k",
                  linewidth=0.5, label=f"$k$={int(k)}")
        )
    fig.legend(handles=shade_patches, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(all_k), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_dir, "publication_gold_full_grouped_bars")


# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"[OK] {stem}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "figures"),
                        help="Output directory for the generated PDFs.")
    args = parser.parse_args()
    out_dir = Path(args.out)

    sim_variance(out_dir)
    sim_budget(out_dir)
    sim_emse(out_dir)
    rkon_gold_cross_family(out_dir)
    publication_gold_full_grouped_bars(out_dir)
    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()
