#!/usr/bin/env python3
"""Variance decomposition analysis on existing publication runs.

Connects the theoretical variance decomposition:
    Var[L] = V_x / N  +  V_{y|x} / (Nk)

to empirical LLM fine-tuning results. Uses the 52 publication runs
(Gold full-data, Gold equal-opportunity, Nectar, UltraFeedback)
to estimate V_x, V_{y|x}, and the optimal k*.

Key outputs:
1. Within-strategy variance as a function of k (does Var decrease with k?)
2. Estimated V_x and V_{y|x} from the data
3. Optimal k* = sqrt((V_{y|x}/V_x) * (C_p/C_r)) for various cost ratios
4. Publication-quality figure
5. CSV with variance statistics
"""

import sys
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load data ──────────────────────────────────────────────────────────
csv_path = PROJECT_ROOT / "paper" / "results" / "publication_camera_ready_results_compact.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} runs from {csv_path.name}")
print(f"Stages: {df['stage_label'].unique()}")

# ── Key metrics to analyze ─────────────────────────────────────────────
METRICS = {
    "reference_loss": ("Reference Loss", "lower"),
    "single_reward_mean": ("Single Reward", "higher"),
    "multi_k16_reference_coverage_mean": ("Coverage@16", "higher"),
    "multi_k16_semantic_diversity_mean": ("Diversity@16", "higher"),
}


# ── 1. Cross-strategy variance at each k ──────────────────────────────
def cross_strategy_variance(df_stage, stage_name):
    """At each k, compute variance across strategies for a metric."""
    results = []
    for metric, (label, direction) in METRICS.items():
        k_values = sorted(df_stage["k"].unique())
        for k in k_values:
            df_k = df_stage[df_stage["k"] == k]
            vals = df_k[metric].dropna().values
            if len(vals) >= 2:
                results.append({
                    "stage": stage_name,
                    "metric": metric,
                    "metric_label": label,
                    "k": k,
                    "mean": np.mean(vals),
                    "std": np.std(vals, ddof=1),
                    "var": np.var(vals, ddof=1),
                    "n_strategies": len(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                    "range": np.max(vals) - np.min(vals),
                })
    return results


# ── 2. Reference loss vs 1/k fit ─────────────────────────────────────
def ref_loss_improvement(df_stage, stage_name):
    """Fit loss = a + b/k for each strategy. a ≈ V_x/N, b ≈ V_{y|x}/N."""
    results = []
    for strategy in sorted(df_stage["strategy"].unique()):
        df_s = df_stage[df_stage["strategy"] == strategy].sort_values("k")
        k_vals = df_s["k"].values
        loss_vals = df_s["reference_loss"].values

        if len(k_vals) >= 3:
            inv_k = 1.0 / k_vals.astype(float)
            A = np.column_stack([np.ones_like(inv_k), inv_k])
            coeffs, _, _, _ = np.linalg.lstsq(A, loss_vals, rcond=None)
            a_hat, b_hat = coeffs

            predicted = a_hat + b_hat * inv_k
            ss_res = np.sum((loss_vals - predicted) ** 2)
            ss_tot = np.sum((loss_vals - np.mean(loss_vals)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            results.append({
                "stage": stage_name,
                "strategy": strategy,
                "a_hat_Vx_over_N": a_hat,
                "b_hat_Vy_over_N": b_hat,
                "R_squared": r_squared,
                "k_values": k_vals.tolist(),
                "loss_values": loss_vals.tolist(),
                "predicted": predicted.tolist(),
                "Vx_Vy_ratio": a_hat / b_hat if b_hat != 0 else float("inf"),
            })
    return results


# ── 3. Optimal k* computation ─────────────────────────────────────────
def compute_optimal_k(ref_loss_results):
    """k* = sqrt((V_{y|x}/V_x) * (C_p/C_r)) for various cost ratios."""
    cost_ratios = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    results = []
    for r in ref_loss_results:
        a = r["a_hat_Vx_over_N"]
        b = r["b_hat_Vy_over_N"]
        if a > 0 and b > 0:
            vy_vx = b / a
            for cp_cr in cost_ratios:
                k_star = np.sqrt(vy_vx * cp_cr)
                results.append({
                    "stage": r["stage"],
                    "strategy": r["strategy"],
                    "Vy_Vx_ratio": vy_vx,
                    "Cp_Cr": cp_cr,
                    "k_star": k_star,
                    "k_star_rounded": max(1, round(k_star)),
                })
    return results


# ─── Run analysis ─────────────────────────────────────────────────────
all_cross_var = []
all_ref_loss = []
all_optimal_k = []

for stage_label in df["stage_label"].unique():
    df_stage = df[df["stage_label"] == stage_label]

    cross_var = cross_strategy_variance(df_stage, stage_label)
    all_cross_var.extend(cross_var)

    ref_loss = ref_loss_improvement(df_stage, stage_label)
    all_ref_loss.extend(ref_loss)

    optimal_k = compute_optimal_k(ref_loss)
    all_optimal_k.extend(optimal_k)

# ── Print summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VARIANCE DECOMPOSITION ANALYSIS")
print("=" * 70)

print("\n── Reference Loss vs 1/k Fit (loss = a + b/k) ──")
print(f"{'Stage':<30} {'Strategy':<8} {'a (V_x/N)':<12} {'b (V_{y|x}/N)':<14} {'R²':<8} {'V_y/V_x':<10}")
print("-" * 82)
for r in all_ref_loss:
    print(f"{r['stage']:<30} {r['strategy']:<8} {r['a_hat_Vx_over_N']:<12.4f} "
          f"{r['b_hat_Vy_over_N']:<14.4f} {r['R_squared']:<8.4f} "
          f"{r['Vx_Vy_ratio']:<10.4f}")

print("\n── Cross-Strategy Variance at Each k (Reference Loss) ──")
for stage_label in df["stage_label"].unique():
    stage_vars = [v for v in all_cross_var
                  if v["stage"] == stage_label and v["metric"] == "reference_loss"]
    if stage_vars:
        print(f"\n  {stage_label}:")
        for v in sorted(stage_vars, key=lambda x: x["k"]):
            print(f"    k={v['k']:>2d}: mean={v['mean']:.4f}  std={v['std']:.4f}  "
                  f"range=[{v['min']:.4f}, {v['max']:.4f}]  n={v['n_strategies']}")

print("\n── Optimal k* for Various Cost Ratios ──")
print(f"{'Stage':<30} {'Strategy':<8} {'V_y/V_x':<10} " +
      " ".join(f"C_p/C_r={cr:<4}" for cr in [1.0, 5.0, 10.0, 20.0]))
print("-" * 100)
for r in all_ref_loss:
    if r["b_hat_Vy_over_N"] > 0 and r["a_hat_Vx_over_N"] > 0:
        vr = r["b_hat_Vy_over_N"] / r["a_hat_Vx_over_N"]
        k_stars = [max(1, round(np.sqrt(vr * cr))) for cr in [1.0, 5.0, 10.0, 20.0]]
        print(f"{r['stage']:<30} {r['strategy']:<8} {vr:<10.4f} " +
              "        ".join(f"k*={k:>2d}" for k in k_stars))

# ── Save CSVs ─────────────────────────────────────────────────────────
output_dir = PROJECT_ROOT / "paper" / "results"
output_dir.mkdir(parents=True, exist_ok=True)

cv_df = pd.DataFrame(all_cross_var)
cv_path = output_dir / "variance_cross_strategy.csv"
cv_df.to_csv(cv_path, index=False)
print(f"\nSaved: {cv_path}")

rl_df = pd.DataFrame(all_ref_loss)
rl_path = output_dir / "variance_ref_loss_fit.csv"
rl_df.to_csv(rl_path, index=False)
print(f"Saved: {rl_path}")

if all_optimal_k:
    ok_df = pd.DataFrame(all_optimal_k)
    ok_path = output_dir / "variance_optimal_k.csv"
    ok_df.to_csv(ok_path, index=False)
    print(f"Saved: {ok_path}")

# ── Generate Publication Figure ───────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 9
    rcParams["axes.labelsize"] = 10
    rcParams["axes.titlesize"] = 10
    rcParams["legend.fontsize"] = 7.5
    rcParams["xtick.labelsize"] = 8
    rcParams["ytick.labelsize"] = 8

    STRATEGY_COLORS = {
        "rkon": "#4C72B0", "bkon": "#DD8452",
        "dkon": "#55A868", "dbkon": "#C44E52",
    }
    STRATEGY_LABELS = {
        "rkon": "RKoN", "bkon": "BKoN",
        "dkon": "DKoN", "dbkon": "DBKoN",
    }
    STRATEGY_MARKERS = {"rkon": "o", "bkon": "s", "dkon": "D", "dbkon": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.4))

    # ── Panel A: Reference loss vs k with 1/k fit ──
    ax = axes[0]
    gold_eq = [r for r in all_ref_loss if "equal" in r["stage"].lower()]
    for r in gold_eq:
        s = r["strategy"]
        k_vals = np.array(r["k_values"], dtype=float)
        loss_vals = np.array(r["loss_values"])
        ax.plot(k_vals, loss_vals,
                marker=STRATEGY_MARKERS.get(s, "o"),
                color=STRATEGY_COLORS.get(s, "gray"),
                label=f'{STRATEGY_LABELS.get(s, s)} (R²={r["R_squared"]:.2f})',
                markersize=5, linewidth=1.2)
        # Fitted curve
        k_smooth = np.linspace(1, 16, 50)
        pred = r["a_hat_Vx_over_N"] + r["b_hat_Vy_over_N"] / k_smooth
        ax.plot(k_smooth, pred, "--", color=STRATEGY_COLORS.get(s, "gray"),
                alpha=0.4, linewidth=0.8)

    ax.set_xlabel("k (responses per prompt)")
    ax.set_ylabel("Reference Loss")
    ax.set_title(r"(a) Loss $\approx a + b/k$")
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=6.5)
    ax.set_xticks([1, 2, 4, 8, 16])

    # ── Panel B: Cross-strategy variance shrinks with k ──
    ax = axes[1]
    for stage_label, ls in [("Gold equal-opportunity", "-"), ("Gold full-data", "--")]:
        stage_vars = [v for v in all_cross_var
                      if v["stage"] == stage_label and v["metric"] == "reference_loss"]
        if stage_vars:
            stage_vars_sorted = sorted(stage_vars, key=lambda x: x["k"])
            k_vals = [v["k"] for v in stage_vars_sorted]
            ranges = [v["range"] for v in stage_vars_sorted]
            short_label = "Equal-opp." if "equal" in stage_label.lower() else "Full-data"
            ax.plot(k_vals, ranges, marker="o", linestyle=ls, label=short_label,
                    markersize=5, linewidth=1.2)

    ax.set_xlabel("k (responses per prompt)")
    ax.set_ylabel("Strategy Spread (range)")
    ax.set_title("(b) Convergence across strategies")
    ax.legend(frameon=False)
    ax.set_xticks([1, 2, 4, 8, 16])

    # ── Panel C: Optimal k* vs cost ratio ──
    ax = axes[2]
    gold_eq_fits = [r for r in all_ref_loss if "equal" in r["stage"].lower()]
    for r in gold_eq_fits:
        s = r["strategy"]
        a, b = r["a_hat_Vx_over_N"], r["b_hat_Vy_over_N"]
        if a > 0 and b > 0:
            vy_vx = b / a
            cost_ratios = np.linspace(0.1, 25, 100)
            k_stars = np.sqrt(vy_vx * cost_ratios)
            ax.plot(cost_ratios, k_stars,
                    color=STRATEGY_COLORS.get(s, "gray"),
                    label=STRATEGY_LABELS.get(s, s),
                    linewidth=1.2)

    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
    ax.axhline(y=4, color="gray", linestyle=":", alpha=0.4, linewidth=0.5)
    ax.set_xlabel(r"Cost ratio $C_p / C_r$")
    ax.set_ylabel(r"Optimal $k^*$")
    ax.set_title(r"(c) $k^* = \sqrt{(V_{y|x}/V_x) \cdot C_p/C_r}$")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_ylim(bottom=0, top=8)

    plt.tight_layout()
    fig_path = PROJECT_ROOT / "paper" / "figures" / "variance_decomposition.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")
    print(f"Saved figure: {fig_path.with_suffix('.png')}")
    plt.close()

except ImportError as e:
    print(f"\nSkipping figure generation: {e}")

print("\nVariance decomposition analysis complete.")
