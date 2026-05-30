"""Aggregate alpha-sweep results into a CSV + plots.

Combines the 8 newly-trained DBKoN runs at alpha in {0.25,0.5,1.0,2.0} for k in {4,8}
with the existing endpoints already in the training runs directory
(``$MRT_RUNS_DIR``, default ``results/runs``):
  - DKoN  (alpha = 0   => pure diversity)         instruct_dkon_k{4,8}_gold_full
  - DBKoN (alpha = 0.75, publication default)     instruct_dbkon_k{4,8}_gold_full
  - BKoN  (alpha = +inf, reward only)             instruct_bkon_k{4,8}_gold_full
Also includes RKoN (random) for reference.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path(os.environ.get("MRT_RUNS_DIR", "results/runs"))
OUT_DIR = Path("results/analysis/alpha_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_ALPHAS = [0.25, 0.5, 1.0, 2.0]
KS = [4, 8]

# Map metric prefixes per k. We always read multi_k{k}_* metrics so all entries
# refer to the same test-time generation budget.
METRIC_KEYS = [
    "reference_loss",
    "reference_perplexity",
    "single_reward_mean",
    "multi_k{k}_reward_mean",
    "multi_k{k}_reward_best_mean",
    "multi_k{k}_semantic_diversity_mean",
    "multi_k{k}_reference_alignment_mean",
    "multi_k{k}_reference_coverage_mean",
    "multi_k{k}_response_words_mean",
]


def alpha_label(a: float) -> str:
    s = f"{a:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def load_run(name: str, k: int) -> dict | None:
    p = RUNS / name / "summary.json"
    if not p.exists():
        return None
    with p.open() as fh:
        d = json.load(fh)
    out = {"run": name}
    eval_metrics = d.get("evaluation_metrics", {})
    train_metrics = d.get("train_metrics", {})
    out["eval_loss_train"] = train_metrics.get("eval_loss")
    for key_tpl in METRIC_KEYS:
        key = key_tpl.format(k=k)
        out[key.replace(f"multi_k{k}_", "multi_")] = eval_metrics.get(key)
    return out


def main() -> None:
    rows = []
    for k in KS:
        # New alpha-sweep runs
        for a in NEW_ALPHAS:
            name = f"gold_gens_llama31_8b_instruct_dbkon_k{k}_alpha{alpha_label(a)}_alpha_sweep"
            r = load_run(name, k)
            if r is None:
                print(f"[agg] missing {name}")
                continue
            r.update({"k": k, "alpha": float(a), "strategy": "dbkon", "source": "alpha_sweep"})
            rows.append(r)
        # Endpoints
        endpoints = [
            ("dkon",  0.0,   f"gold_gens_llama31_8b_instruct_dkon_k{k}_gold_full"),
            ("dbkon", 0.75,  f"gold_gens_llama31_8b_instruct_dbkon_k{k}_gold_full"),
            ("bkon",  math.inf, f"gold_gens_llama31_8b_instruct_bkon_k{k}_gold_full"),
            ("rkon",  float("nan"), f"gold_gens_llama31_8b_instruct_rkon_k{k}_gold_full"),
        ]
        for strat, a, name in endpoints:
            r = load_run(name, k)
            if r is None:
                print(f"[agg] missing {name}")
                continue
            r.update({"k": k, "alpha": a, "strategy": strat, "source": "existing"})
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "alpha_sweep_raw.csv", index=False)

    # Compose a tidy view.
    keep = [
        "k", "alpha", "strategy", "source",
        "eval_loss_train", "reference_loss", "reference_perplexity",
        "single_reward_mean", "multi_reward_mean", "multi_reward_best_mean",
        "multi_semantic_diversity_mean", "multi_reference_alignment_mean",
        "multi_reference_coverage_mean", "multi_response_words_mean",
    ]
    tidy = df[keep].sort_values(["k", "alpha"], na_position="last")
    tidy.to_csv(OUT_DIR / "alpha_sweep_tidy.csv", index=False)
    print(tidy.to_string(index=False))

    # Build per-k plots: 4-panel (reward, BoN reward, diversity, ref-loss).
    for k in KS:
        sub_dbkon = df[(df["k"] == k) & (df["strategy"] == "dbkon")].copy()
        # sweep curve = dkon at alpha=0 + dbkon at alphas + bkon at right edge.
        dkon_row = df[(df["k"] == k) & (df["strategy"] == "dkon")]
        bkon_row = df[(df["k"] == k) & (df["strategy"] == "bkon")]
        rkon_row = df[(df["k"] == k) & (df["strategy"] == "rkon")]

        # alpha values to plot on log axis: 0 -> placeholder, inf -> placeholder
        eps_low = 0.05  # placeholder for alpha=0 on log axis
        eps_high = 8.0  # placeholder for alpha=infinity
        sub_dbkon = sub_dbkon.sort_values("alpha")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        panel_specs = [
            ("multi_reward_mean", f"Reward (multi-k={k} mean)"),
            ("multi_reward_best_mean", f"BoN reward (best of {k})"),
            ("multi_semantic_diversity_mean", f"Semantic diversity (k={k})"),
            ("reference_loss", "Reference loss (test split)"),
        ]
        for ax, (col, title) in zip(axes.flat, panel_specs):
            xs = sub_dbkon["alpha"].to_numpy(dtype=float)
            ys = sub_dbkon[col].to_numpy(dtype=float)
            # add dkon endpoint as alpha=eps_low marker
            extra_x, extra_y, extra_lbl = [], [], []
            if len(dkon_row):
                extra_x.append(eps_low); extra_y.append(float(dkon_row[col].iloc[0])); extra_lbl.append("DKoN (\u03b1=0)")
            if len(bkon_row):
                extra_x.append(eps_high); extra_y.append(float(bkon_row[col].iloc[0])); extra_lbl.append("BKoN (\u03b1\u2192\u221e)")
            ax.plot(np.r_[extra_x[:1], xs, extra_x[1:2]],
                    np.r_[extra_y[:1], ys, extra_y[1:2]],
                    "-o", color="C0", label="DBKoN sweep")
            for ex, ey, el in zip(extra_x, extra_y, extra_lbl):
                ax.scatter([ex], [ey], color="C3", zorder=5, s=60, marker="s")
                ax.annotate(el, (ex, ey), textcoords="offset points", xytext=(5, 6), fontsize=8)
            if len(rkon_row):
                ax.axhline(float(rkon_row[col].iloc[0]), color="grey", linestyle="--", linewidth=1, label="RKoN baseline")
            ax.set_xscale("log")
            ax.set_xlabel(r"GRADES quality exponent $\alpha$")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8, loc="best")
        fig.suptitle(f"DBKoN \u03b1-sweep at k={k} (Llama-3.1-8B-Instruct, Gold-full)")
        fig.tight_layout()
        out = OUT_DIR / f"alpha_sweep_k{k}.png"
        fig.savefig(out, dpi=150)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        print(f"[agg] wrote {out}")

    # Combined two-panel summary: reward vs alpha and diversity vs alpha for both k.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for k, color in zip(KS, ["C0", "C2"]):
        sub = df[(df["k"] == k) & (df["strategy"] == "dbkon")].sort_values("alpha")
        axes[0].plot(sub["alpha"], sub["multi_reward_mean"], "-o", color=color, label=f"k={k}")
        axes[1].plot(sub["alpha"], sub["multi_semantic_diversity_mean"], "-o", color=color, label=f"k={k}")
    for ax, ylab in zip(axes, ["Mean reward (test, multi-k)", "Semantic diversity (test, multi-k)"]):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "alpha_sweep_summary.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[agg] wrote {out}")


if __name__ == "__main__":
    main()


def pareto_plot() -> None:
    df = pd.read_csv(OUT_DIR / "alpha_sweep_raw.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, k in zip(axes, KS):
        sub = df[df["k"] == k].copy()
        sub = sub.sort_values("alpha", na_position="last")
        # DBKoN points
        d = sub[sub["strategy"] == "dbkon"]
        ax.scatter(d["multi_reward_mean"], d["reference_loss"], s=70, c=range(len(d)), cmap="viridis", zorder=3)
        for _, r in d.iterrows():
            lbl = "0" if r["alpha"] == 0 else (r"$\infty$" if math.isinf(r["alpha"]) else f"{r['alpha']:g}")
            ax.annotate(lbl, (r["multi_reward_mean"], r["reference_loss"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        # endpoints
        for strat, marker, lbl in [("dkon", "s", "DKoN"), ("bkon", "^", "BKoN"), ("rkon", "x", "RKoN")]:
            row = sub[sub["strategy"] == strat]
            if not len(row):
                continue
            ax.scatter(row["multi_reward_mean"], row["reference_loss"], marker=marker, s=110,
                       facecolor="none" if strat != "rkon" else "red",
                       edgecolor="C3", linewidth=2, zorder=4, label=lbl)
        ax.set_xlabel(f"Reward (multi-k={k} mean) \u2192 better")
        ax.set_ylabel("Reference loss \u2190 better (lower is better)")
        ax.set_title(f"k={k}: \u03b1-sweep traces a Pareto frontier")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = OUT_DIR / "alpha_sweep_pareto.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[agg] wrote {out}")


if __name__ == "__main__":
    pareto_plot()
