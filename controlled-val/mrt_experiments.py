"""
Multi-Response Training (MRT) — Unified Experiment Suite
=========================================================
Validates theoretical predictions for the repositioned paper:
"Multi-Response Training Improves Language Model Generalization: Theory and Practice"

Experiments:
  1. Variance of loss estimator vs k  (validates Var = V_x/N + V_{y|x}/(Nk))
  2. Budget allocation under fixed budget  (validates k* formula)
  3. Mode lottery: model-outcome consistency  (NEW — k>1 reduces training variance)
  4. Distributional learning NLL by selection strategy
  5. Gradient EMSE decomposition (Bias² + Variance)
  6. Similarity–gradient bridge validation
  7. Discrete-token Transformer simulation

All experiments run on CPU (M2 MacBook Air compatible).
Total runtime: ~15-25 minutes.
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

DEVICE = torch.device("cpu")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "random": "#1f77b4",
    "best":   "#ff7f0e",
    "diverse": "#2ca02c",
    "redundant": "#d62728",
    "theory": "#e41a1c",
}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION  (shared across experiments 1-6)
# ══════════════════════════════════════════════════════════════════════════════

def true_function(x: np.ndarray) -> np.ndarray:
    """f(x) = sin(2 x_1) + 0.5 x_2"""
    return np.sin(x[:, 0:1] * 2) + 0.5 * x[:, 1:2]


def generate_multimodal_responses(
    x: np.ndarray,
    n_responses: int,
    n_modes: int = 3,
    mode_spread: float = 1.5,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
    return_modes: bool = False,
):
    """Each prompt has n_modes valid responses spread around f(x)."""
    if seed is not None:
        np.random.seed(seed)
    n = x.shape[0]
    f_x = true_function(x)
    mode_offsets = np.linspace(-mode_spread, mode_spread, n_modes)
    responses = np.zeros((n, n_responses, 1), dtype=np.float32)
    modes_arr = np.zeros((n, n_responses), dtype=int)
    for i in range(n):
        modes = np.random.choice(n_modes, size=n_responses)
        modes_arr[i] = modes
        for j in range(n_responses):
            responses[i, j, 0] = (
                f_x[i, 0] + mode_offsets[modes[j]] + np.random.randn() * noise_std
            )
    if return_modes:
        return responses, modes_arr
    return responses


def compute_rewards(x: np.ndarray, responses: np.ndarray) -> np.ndarray:
    f_x = true_function(x)
    errors = (responses[:, :, 0] - f_x[:, 0:1]) ** 2
    return (-errors).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SELECTION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def select_random(n_prompts: int, n_responses: int, k: int, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(seed)
    sel = np.zeros((n_prompts, k), dtype=int)
    for i in range(n_prompts):
        sel[i] = np.random.choice(n_responses, k, replace=False)
    return sel


def select_best(rewards: np.ndarray, k: int):
    n = rewards.shape[0]
    sel = np.zeros((n, k), dtype=int)
    for i in range(n):
        sel[i] = np.argsort(rewards[i])[::-1][:k]
    return sel


def select_diverse(responses: np.ndarray, rewards: np.ndarray, k: int,
                   tau_quantile: float = 0.5):
    """Quality-filter then greedy max-min diversity in response space."""
    n = rewards.shape[0]
    sel = np.zeros((n, k), dtype=int)
    for i in range(n):
        threshold = np.quantile(rewards[i], tau_quantile)
        cands = np.where(rewards[i] >= threshold)[0]
        if len(cands) < k:
            cands = np.argsort(rewards[i])[::-1][:max(k, len(cands))]
        vals = responses[i, cands, 0]
        picked = [int(np.argmax(rewards[i, cands]))]
        for _ in range(k - 1):
            remaining = [c for c in range(len(cands)) if c not in picked]
            if not remaining:
                break
            picked_vals = vals[picked]
            best_idx, best_d = -1, -1.0
            for c in remaining:
                d = float(np.min(np.abs(vals[c] - picked_vals)))
                if d > best_d:
                    best_d, best_idx = d, c
            picked.append(best_idx)
        sel[i] = cands[picked[:k]]
    return sel


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class SimpleMLP(nn.Module):
    def __init__(self, d_in: int, d_hid: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_out),
        )
    def forward(self, x):
        return self.net(x)


class MixtureDensityNetwork(nn.Module):
    """Predicts parameters of a Gaussian mixture for P(y|x)."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_components: int = 5):
        super().__init__()
        self.n_components = n_components
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden_dim, n_components)
        self.mu_head = nn.Linear(hidden_dim, n_components)
        self.log_sigma_head = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.shared(x)
        pi = torch.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(-3, 3)
        return pi, mu, log_sigma

    def nll(self, x, y):
        pi, mu, log_sigma = self.forward(x)
        sigma = torch.exp(log_sigma)
        y_expand = y.expand_as(mu)
        log_probs = (
            -0.5 * ((y_expand - mu) / sigma) ** 2
            - log_sigma
            - 0.5 * np.log(2 * np.pi)
        )
        log_mix = torch.log(pi + 1e-10) + log_probs
        return -torch.logsumexp(log_mix, dim=-1).mean()

    def nll_single(self, x, y):
        pi, mu, log_sigma = self.forward(x)
        sigma = torch.exp(log_sigma)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        y_expand = y.expand_as(mu)
        log_probs = (
            -0.5 * ((y_expand - mu) / sigma) ** 2
            - log_sigma
            - 0.5 * np.log(2 * np.pi)
        )
        log_mix = torch.log(pi + 1e-10) + log_probs
        return -torch.logsumexp(log_mix, dim=-1).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def build_xy(prompts, responses, sel):
    N, k = sel.shape
    X = np.repeat(prompts, k, axis=0)
    Y = np.zeros((N * k, 1), dtype=np.float32)
    for i in range(N):
        for j in range(k):
            Y[i * k + j, 0] = responses[i, sel[i, j], 0]
    return torch.tensor(X, device=DEVICE), torch.tensor(Y, device=DEVICE)


def train_mdn(X_tr, Y_tr, dim, n_components=5, n_epochs=400, lr=0.005, seed=0):
    set_seed(seed)
    mdn = MixtureDensityNetwork(dim, hidden_dim=64, n_components=n_components)
    optimizer = optim.Adam(mdn.parameters(), lr=lr)
    mdn.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = mdn.nll(X_tr, Y_tr)
        if torch.isnan(loss) or torch.isinf(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdn.parameters(), 5.0)
        optimizer.step()
    return mdn


def eval_nll(mdn, prompts, test_responses, n_test=100):
    mdn.eval()
    X_t = torch.tensor(prompts, device=DEVICE)
    nlls = []
    with torch.no_grad():
        for j in range(min(n_test, test_responses.shape[1])):
            Y_j = torch.tensor(
                test_responses[:, j : j + 1, :].reshape(-1, 1), device=DEVICE
            )
            nlls.append(mdn.nll(X_t, Y_j).item())
    return np.mean(nlls)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Variance of Loss Estimator vs k
# ══════════════════════════════════════════════════════════════════════════════

def experiment_1_variance():
    """Verify Var[L̂] = V_x/N + V_{y|x}/(Nk) at a fixed θ."""
    print("\n" + "=" * 65)
    print("EXP 1: Variance of Empirical Loss Estimator vs k")
    print("=" * 65)

    dim, N, n_total = 5, 100, 50
    k_values = [1, 2, 4, 8, 16, 32]
    n_mc = 500

    set_seed(SEED)
    model = SimpleMLP(dim, 32, 1)
    results = {k: [] for k in k_values}

    for mc in range(n_mc):
        prompts = np.random.randn(N, dim).astype(np.float32)
        responses = generate_multimodal_responses(
            prompts, n_total, seed=SEED + mc * 7
        )
        X_t = torch.tensor(prompts, device=DEVICE)
        for k in k_values:
            idx = np.zeros((N, k), dtype=int)
            for ii in range(N):
                idx[ii] = np.random.choice(n_total, size=k, replace=False)
            losses = []
            with torch.no_grad():
                for j in range(k):
                    Y_j = torch.tensor(
                        responses[np.arange(N), idx[:, j]], device=DEVICE
                    )
                    pred = model(X_t)
                    per_sample = ((pred - Y_j) ** 2).squeeze()
                    losses.append(per_sample.numpy())
            mean_loss = np.mean(np.stack(losses, axis=1), axis=1)
            results[k].append(float(np.mean(mean_loss)))

    # Print table
    variances = {}
    var_k1 = np.var(results[1])
    print(f"{'k':>4s}  {'Var[L̂]':>14s}  {'Ratio to k=1':>14s}")
    for k in k_values:
        v = np.var(results[k])
        variances[k] = v
        ratio = v / var_k1 if var_k1 > 0 else 0
        print(f"{k:4d}  {v:14.6f}  {ratio:14.4f}")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    var_list = [variances[k] for k in k_values]
    ax.plot(k_values, var_list, "o-", lw=2, ms=7, color=COLORS["random"],
            label="Empirical", zorder=3)
    # O(1/k) reference anchored to k=1
    c = var_list[0] * k_values[0]
    theory = [c / k for k in k_values]
    ax.plot(k_values, theory, "s--", color=COLORS["theory"], alpha=0.7,
            label=r"$O(1/k)$ reference")
    v_floor = min(var_list)
    ax.axhline(v_floor, color="gray", ls=":", alpha=0.5,
               label=r"$V_{\mathbf{x}}/N$ floor")
    ax.set_xlabel("$k$ (responses per prompt)")
    ax.set_ylabel(r"$\mathrm{Var}[\hat{\mathcal{L}}_m(\theta)]$")
    ax.set_title("Variance of Loss Estimator vs. $k$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(RESULTS_DIR / "exp1_variance.pdf")
    fig.savefig(RESULTS_DIR / "exp1_variance.png")
    plt.close(fig)
    # Raw data dump for downstream paper-final figure rendering
    with open(RESULTS_DIR / "exp1_variance.json", "w") as fh:
        json.dump({
            "k_values": list(map(int, k_values)),
            "variance": [float(variances[k]) for k in k_values],
        }, fh, indent=2)
    print("  Saved: exp1_variance.pdf/.png/.json")
    return variances


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Budget Allocation
# ══════════════════════════════════════════════════════════════════════════════

def experiment_2_budget():
    """Validate k* under a fixed computational budget."""
    print("\n" + "=" * 65)
    print("EXP 2: Budget Allocation (fixed B, varying N and k)")
    print("=" * 65)

    dim = 5
    C_p, C_r = 4.0, 1.0
    B_total = 400.0
    n_cand = 32
    n_repeats = 8
    n_modes = 3

    k_values = [1, 2, 4, 6, 8, 12, 16]
    allocations = [(int(B_total / (C_p + k * C_r)), k) for k in k_values]
    for N, k in allocations:
        print(f"  k={k:2d}: N={N:3d}  (budget={N*(C_p+k*C_r):.0f})")

    # Estimate V_x and V_{y|x}
    set_seed(SEED + 7777)
    est_N, est_k_resp = 300, 200
    est_prompts = np.random.randn(est_N, dim).astype(np.float32)
    est_responses = generate_multimodal_responses(
        est_prompts, est_k_resp, n_modes=n_modes, seed=SEED + 7778
    )
    ref_model = MixtureDensityNetwork(dim)
    X_est = torch.tensor(est_prompts, device=DEVICE)
    with torch.no_grad():
        prompt_means, prompt_vars = [], []
        for i in range(est_N):
            losses_i = []
            for j in range(est_k_resp):
                Y_j = torch.tensor([[est_responses[i, j, 0]]], device=DEVICE)
                losses_i.append(ref_model.nll(X_est[i : i + 1], Y_j).item())
            prompt_means.append(np.mean(losses_i))
            prompt_vars.append(np.var(losses_i))
    V_x = float(np.var(prompt_means))
    V_yx = float(np.mean(prompt_vars))
    k_star = np.sqrt(V_yx / max(V_x, 1e-8) * C_p / C_r)
    print(f"\n  V_x={V_x:.4f}, V_{{y|x}}={V_yx:.4f}")
    print(f"  k* = {k_star:.2f}")

    results = {
        "Random": {k: [] for k in k_values},
        "GRADES": {k: [] for k in k_values},
    }

    for rep in range(n_repeats):
        set_seed(SEED + rep * 100)
        max_N = max(N for N, _ in allocations)
        prompts_pool = np.random.randn(max_N, dim).astype(np.float32)
        resp_pool = generate_multimodal_responses(
            prompts_pool, n_cand, n_modes=n_modes, seed=SEED + rep * 100 + 50
        )
        rew_pool = compute_rewards(prompts_pool, resp_pool)
        test_prompts = np.random.randn(100, dim).astype(np.float32)
        test_resp = generate_multimodal_responses(
            test_prompts, 100, n_modes=n_modes, seed=SEED + rep * 100 + 999
        )

        for N, k in allocations:
            prompts = prompts_pool[:N]
            responses = resp_pool[:N]
            rewards = rew_pool[:N]
            for strat_name in ["Random", "GRADES"]:
                if strat_name == "Random":
                    sel = select_random(N, n_cand, k, seed=SEED + rep + k)
                else:
                    sel = select_diverse(responses, rewards, k)
                X_tr, Y_tr = build_xy(prompts, responses, sel)
                mdn = train_mdn(X_tr, Y_tr, dim, seed=SEED + rep * 1000 + k)
                nll = eval_nll(mdn, test_prompts, test_resp)
                results[strat_name][k].append(nll)
        print(f"  Rep {rep+1}/{n_repeats}")

    # Print table
    print(f"\n  {'k':>3s} {'N':>4s} | {'Random':>14s} | {'GRADES':>14s}")
    print("  " + "-" * 44)
    for N, k in allocations:
        rm = np.mean(results["Random"][k])
        rs = np.std(results["Random"][k])
        dm = np.mean(results["GRADES"][k])
        ds = np.std(results["GRADES"][k])
        tag = " <-- k*" if abs(k - round(k_star)) <= 1 else ""
        print(f"  {k:3d} {N:4d} | {rm:6.1f} +/- {rs:5.1f} | {dm:6.1f} +/- {ds:5.1f}{tag}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rm = [np.mean(results["Random"][k]) for k in k_values]
    rs = [np.std(results["Random"][k]) for k in k_values]
    dm = [np.mean(results["GRADES"][k]) for k in k_values]
    ds = [np.std(results["GRADES"][k]) for k in k_values]

    axes[0].errorbar(k_values, rm, yerr=rs, marker="^", capsize=4, lw=2,
                     color=COLORS["random"], label="Random")
    axes[0].errorbar(k_values, dm, yerr=ds, marker="o", capsize=4, lw=2,
                     color=COLORS["diverse"], label="GRADES")
    axes[0].axvline(k_star, color=COLORS["theory"], ls="--", alpha=0.7,
                    label=f"$k^* = {k_star:.1f}$")
    axes[0].set_xlabel("$k$ (responses per prompt)")
    axes[0].set_ylabel("Test NLL")
    axes[0].set_title(f"Fixed Budget ($C_p$={C_p:.0f}, $C_r$={C_r:.0f}, $B$={B_total:.0f})")
    axes[0].legend()
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    ax2.set_xticks(k_values)
    ax2.set_xticklabels([f"N={N}" for N, _ in allocations], fontsize=8, rotation=30)

    k_cont = np.linspace(0.5, 20, 300)
    var_curve = [(V_x + V_yx / kc) * (C_p + kc * C_r) / B_total for kc in k_cont]
    axes[1].plot(k_cont, var_curve, "k-", lw=2, label=r"Var[$\hat{\mathcal{L}}$] (theory)")
    axes[1].axvline(k_star, color=COLORS["theory"], ls="--", alpha=0.7,
                    label=f"$k^* = {k_star:.1f}$")
    axes[1].set_xlabel("$k$")
    axes[1].set_ylabel("Estimator Variance")
    axes[1].set_title("Variance under Budget Constraint")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_budget.pdf")
    fig.savefig(RESULTS_DIR / "exp2_budget.png")
    plt.close(fig)
    with open(RESULTS_DIR / "exp2_budget.json", "w") as fh:
        json.dump({
            "k_values": list(map(int, k_values)),
            "N_values": [int(N) for N, _ in allocations],
            "random_mean":  rm, "random_std":  rs,
            "grades_mean": dm, "grades_std": ds,
            "k_star": float(k_star),
            "V_x": float(V_x), "V_yx": float(V_yx),
            "C_p": float(C_p), "C_r": float(C_r), "B_total": float(B_total),
        }, fh, indent=2)
    print("  Saved: exp2_budget.pdf/.png/.json")
    return results, k_star, V_x, V_yx


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Mode Lottery (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_3_mode_lottery():
    """
    Show that k=1 training is a 'mode lottery':
    - Train MDN many times with different random single responses (k=1)
    - Compare to k=4 and k=8
    - k>1 dramatically reduces variance in training outcomes.
    """
    print("\n" + "=" * 65)
    print("EXP 3: The Mode Lottery — Training Outcome Consistency")
    print("=" * 65)

    dim = 5
    N = 120
    n_cand = 32
    n_modes = 3
    k_values = [1, 2, 4, 8]
    n_repeats = 20  # many runs to show variance

    # Shared test data
    set_seed(SEED + 5000)
    test_prompts = np.random.randn(100, dim).astype(np.float32)
    test_resp = generate_multimodal_responses(
        test_prompts, 100, n_modes=n_modes, seed=SEED + 5001
    )

    results = {k: [] for k in k_values}

    for rep in range(n_repeats):
        set_seed(SEED + rep * 37)
        prompts = np.random.randn(N, dim).astype(np.float32)
        responses = generate_multimodal_responses(
            prompts, n_cand, n_modes=n_modes, seed=SEED + rep * 37 + 10
        )
        rewards = compute_rewards(prompts, responses)

        for k in k_values:
            # Random selection (the core MRT proposal)
            sel = select_random(N, n_cand, k, seed=SEED + rep * 100 + k)
            X_tr, Y_tr = build_xy(prompts, responses, sel)
            mdn = train_mdn(X_tr, Y_tr, dim, n_components=5,
                            seed=SEED + rep * 1000 + k)
            nll = eval_nll(mdn, test_prompts, test_resp)
            results[k].append(nll)

        if (rep + 1) % 5 == 0:
            print(f"  Rep {rep+1}/{n_repeats}")

    # Print summary
    print(f"\n  {'k':>4s}  {'Mean NLL':>10s}  {'Std NLL':>10s}  {'CV':>8s}")
    for k in k_values:
        m = np.mean(results[k])
        s = np.std(results[k])
        cv = s / abs(m) if abs(m) > 1e-6 else 0
        print(f"  {k:4d}  {m:10.2f}  {s:10.2f}  {cv:8.2%}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: box/violin plot of NLL across runs
    data_for_box = [results[k] for k in k_values]
    bp = axes[0].boxplot(
        data_for_box, positions=range(len(k_values)), widths=0.5,
        patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    blue_shades = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd"]
    for patch, color in zip(bp["boxes"], blue_shades):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
    axes[0].set_xticks(range(len(k_values)))
    axes[0].set_xticklabels([f"$k$={k}" for k in k_values])
    axes[0].set_ylabel("Test NLL")
    axes[0].set_title("The Mode Lottery: Training Outcome Variance")
    axes[0].set_xlabel("Responses per prompt")

    # Right: coefficient of variation vs k
    means = [np.mean(results[k]) for k in k_values]
    stds = [np.std(results[k]) for k in k_values]
    cvs = [s / abs(m) if abs(m) > 1e-6 else 0 for m, s in zip(means, stds)]

    ax_r = axes[1]
    ax_r.bar(range(len(k_values)), stds, color=blue_shades, edgecolor="black",
             alpha=0.9)
    ax_r.set_xticks(range(len(k_values)))
    ax_r.set_xticklabels([f"$k$={k}" for k in k_values])
    ax_r.set_ylabel("Std of Test NLL across runs")
    ax_r.set_title("Multi-Response Training Reduces Outcome Variance")
    ax_r.set_xlabel("Responses per prompt")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_mode_lottery.pdf")
    fig.savefig(RESULTS_DIR / "exp3_mode_lottery.png")
    plt.close(fig)
    print("  Saved: exp3_mode_lottery.pdf/.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Distributional Learning (NLL by Strategy)
# ══════════════════════════════════════════════════════════════════════════════

def experiment_4_distributional():
    """Train MDN to learn P(y|x); compare selection strategies."""
    print("\n" + "=" * 65)
    print("EXP 4: Distributional Learning — Selection Strategy Comparison")
    print("=" * 65)

    dim = 5
    N = 150
    n_cand = 32
    n_modes = 3
    k_values = [1, 2, 4, 8, 16]
    n_repeats = 10

    strategies = ["Top-$k$", "Random", "GRADES"]
    results = {s: {k: [] for k in k_values} for s in strategies}

    for rep in range(n_repeats):
        set_seed(SEED + rep)
        prompts = np.random.randn(N, dim).astype(np.float32)
        responses = generate_multimodal_responses(
            prompts, n_cand, n_modes=n_modes, seed=SEED + rep + 100
        )
        rewards = compute_rewards(prompts, responses)
        test_resp = generate_multimodal_responses(
            prompts, 100, n_modes=n_modes, seed=SEED + rep + 999
        )

        for k in k_values:
            for sname in strategies:
                if "Top" in sname:
                    sel = select_best(rewards, k)
                elif "Random" in sname:
                    sel = select_random(N, n_cand, k, seed=SEED + rep + k)
                else:
                    sel = select_diverse(responses, rewards, k)
                X_tr, Y_tr = build_xy(prompts, responses, sel)
                mdn = train_mdn(X_tr, Y_tr, dim, seed=SEED + rep * 1000 + k)
                nll = eval_nll(mdn, prompts, test_resp)
                results[sname][k].append(nll)
        print(f"  Rep {rep+1}/{n_repeats}")

    # Print table
    print(f"\n  {'Strategy':>22s} | ", end="")
    for k in k_values:
        print(f"{'k='+str(k):>10s}", end=" ")
    print()
    print("  " + "-" * 78)
    for s in strategies:
        print(f"  {s:>22s} | ", end="")
        for k in k_values:
            m = np.mean(results[s][k])
            print(f"{m:10.1f}", end=" ")
        print()

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {"Top-$k$": "s", "Random": "^",
               "GRADES": "o"}
    colors_s = {"Top-$k$": COLORS["best"],
                "Random": COLORS["random"],
                "GRADES": COLORS["diverse"]}
    for s in strategies:
        means = [np.mean(results[s][k]) for k in k_values]
        stds_ = [np.std(results[s][k]) for k in k_values]
        ax.errorbar(k_values, means, yerr=stds_, marker=markers[s], capsize=4,
                    lw=2, color=colors_s[s], label=s)
    ax.set_xlabel("$k$ (responses per prompt)")
    ax.set_ylabel("Test NLL")
    ax.set_title("Distributional Learning: NLL on Fresh Samples")
    ax.set_xscale("log", base=2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_distributional.pdf")
    fig.savefig(RESULTS_DIR / "exp4_distributional.png")
    plt.close(fig)
    print("  Saved: exp4_distributional.pdf/.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — Gradient EMSE Decomposition
# ══════════════════════════════════════════════════════════════════════════════

def experiment_5_gradient_emse():
    """Bias² + Variance of per-prompt gradient estimator."""
    print("\n" + "=" * 65)
    print("EXP 5: Gradient Estimator EMSE (Bias² + Variance)")
    print("=" * 65)

    dim = 5
    N = 30
    n_cand = 32
    k = 4
    n_trials = 200
    n_modes = 3

    set_seed(SEED)
    prompts = np.random.randn(N, dim).astype(np.float32)
    X_t = torch.tensor(prompts, device=DEVICE)
    mdn = MixtureDensityNetwork(dim, hidden_dim=64, n_components=5)

    # True per-prompt expected loss and gradient
    large_resp = generate_multimodal_responses(
        prompts, 500, n_modes=n_modes, seed=SEED + 9999
    )
    true_losses = np.zeros(N)
    true_grads = []
    with torch.no_grad():
        for i in range(N):
            ls = []
            for j in range(500):
                Y_j = torch.tensor([[large_resp[i, j, 0]]], device=DEVICE)
                ls.append(mdn.nll(X_t[i : i + 1], Y_j).item())
            true_losses[i] = np.mean(ls)

    for i in range(N):
        grad_acc = None
        for j in range(500):
            mdn.zero_grad()
            Y_j = torch.tensor([[large_resp[i, j, 0]]], device=DEVICE)
            nll = mdn.nll(X_t[i : i + 1], Y_j)
            nll.backward()
            g = torch.cat(
                [p.grad.flatten().clone() for p in mdn.parameters() if p.grad is not None]
            )
            grad_acc = g if grad_acc is None else grad_acc + g
        true_grads.append((grad_acc / 500).detach())

    strategies = {
        "Random": lambda resp, rew: select_random(N, n_cand, k, seed=None),
        "Top-$k$": lambda resp, rew: select_best(rew, k),
        "GRADES": lambda resp, rew: select_diverse(resp, rew, k),
    }

    all_loss_ests = {s: np.zeros((n_trials, N)) for s in strategies}
    all_grad_cos = {s: np.zeros((n_trials, N)) for s in strategies}

    for trial in range(n_trials):
        responses = generate_multimodal_responses(
            prompts, n_cand, n_modes=n_modes, seed=SEED + trial * 17 + 1
        )
        rewards = compute_rewards(prompts, responses)
        for sname, sfn in strategies.items():
            np.random.seed(SEED + trial * 31 + hash(sname) % 10000)
            sel = sfn(responses, rewards)
            for i in range(N):
                losses_i = []
                grad_acc = None
                for j in range(k):
                    mdn.zero_grad()
                    Y_j = torch.tensor(
                        [[responses[i, sel[i, j], 0]]], device=DEVICE
                    )
                    nll = mdn.nll(X_t[i : i + 1], Y_j)
                    nll.backward()
                    losses_i.append(nll.item())
                    g = torch.cat(
                        [p.grad.flatten().clone()
                         for p in mdn.parameters() if p.grad is not None]
                    )
                    grad_acc = g if grad_acc is None else grad_acc + g
                all_loss_ests[sname][trial, i] = np.mean(losses_i)
                g_hat = grad_acc / k
                cos_val = torch.dot(g_hat, true_grads[i]) / (
                    g_hat.norm() * true_grads[i].norm() + 1e-10
                )
                all_grad_cos[sname][trial, i] = cos_val.item()
        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{n_trials}")

    summary = {}
    print(f"\n  {'Strategy':>10s} | {'Bias²':>8s} {'Var':>8s} {'EMSE':>8s} | {'Grad cos':>8s}")
    print("  " + "-" * 55)
    for sname in strategies:
        est = all_loss_ests[sname]
        bias_per_prompt = est.mean(axis=0) - true_losses
        var_per_prompt = est.var(axis=0)
        bias_sq = float(np.mean(bias_per_prompt ** 2))
        variance = float(np.mean(var_per_prompt))
        emse = bias_sq + variance
        grad_cos = float(np.mean(all_grad_cos[sname]))
        summary[sname] = {
            "bias_sq": bias_sq, "var": variance, "emse": emse, "grad_cos": grad_cos
        }
        print(f"  {sname:>10s} | {bias_sq:8.4f} {variance:8.4f} {emse:8.4f} | {grad_cos:8.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(summary.keys())
    bias_sq_vals = [summary[n]["bias_sq"] for n in names]
    var_vals = [summary[n]["var"] for n in names]
    colors_bar = [COLORS["random"], COLORS["best"], COLORS["diverse"]]

    x_pos = np.arange(len(names))
    w = 0.35
    axes[0].bar(x_pos - w / 2, bias_sq_vals, w, label="Bias²",
                color=colors_bar, alpha=0.55, edgecolor="black")
    axes[0].bar(x_pos + w / 2, var_vals, w, label="Variance",
                color=colors_bar, alpha=0.9, edgecolor="black", hatch="//")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("EMSE component")
    axes[0].set_title(f"Bias² vs Variance ($k$={k})")
    axes[0].legend()

    hist_colors = {"Random": COLORS["random"], "Top-$k$": COLORS["best"],
                   "GRADES": COLORS["diverse"]}
    for sname in names:
        data = all_grad_cos[sname].mean(axis=1)
        axes[1].hist(data, bins=30, alpha=0.5, label=sname,
                     color=hist_colors[sname], edgecolor="black", lw=0.5)
    axes[1].set_xlabel("Gradient cosine similarity to true gradient")
    axes[1].set_ylabel("Count (trials)")
    axes[1].set_title(f"Gradient Direction Accuracy ($k$={k})")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_gradient_emse.pdf")
    fig.savefig(RESULTS_DIR / "exp5_gradient_emse.png")
    plt.close(fig)
    with open(RESULTS_DIR / "exp5_gradient_emse.json", "w") as fh:
        json.dump({
            "k": int(k),
            "strategies": list(summary.keys()),
            "summary": summary,
            "grad_cos_per_trial": {
                s: all_grad_cos[s].mean(axis=1).tolist()
                for s in summary.keys()
            },
        }, fh, indent=2)
    print("  Saved: exp5_gradient_emse.pdf/.png/.json")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — Similarity–Gradient Bridge
# ══════════════════════════════════════════════════════════════════════════════

def experiment_6_bridge():
    """Embedding similarity predicts gradient cosine similarity."""
    print("\n" + "=" * 65)
    print("EXP 6: Similarity–Gradient Bridge Validation")
    print("=" * 65)

    dim = 5
    N = 50
    n_cand = 16

    set_seed(SEED)
    prompts = np.random.randn(N, dim).astype(np.float32)
    responses, modes = generate_multimodal_responses(
        prompts, n_cand, seed=SEED + 100, return_modes=True
    )
    mdn = MixtureDensityNetwork(dim, hidden_dim=64, n_components=5)
    X_t = torch.tensor(prompts, device=DEVICE)

    all_resp_sims, all_grad_sims, all_same_mode = [], [], []

    for i in range(N):
        vals = responses[i, :, 0]
        dists = np.abs(vals[:, None] - vals[None, :])
        max_d = dists.max() + 1e-8
        resp_sim = 1.0 - dists / max_d

        gradients = []
        for j in range(n_cand):
            mdn.zero_grad()
            x_i = torch.tensor(prompts[i], device=DEVICE).unsqueeze(0)
            y_j = torch.tensor([[responses[i, j, 0]]], device=DEVICE)
            nll = mdn.nll_single(x_i, y_j)
            nll.backward()
            g = torch.cat(
                [p.grad.flatten().clone() for p in mdn.parameters() if p.grad is not None]
            )
            gradients.append(g.detach())
        grad_matrix = torch.stack(gradients)
        norms = grad_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
        grad_normed = grad_matrix / norms
        grad_cos_sim = (grad_normed @ grad_normed.T).numpy()

        for j1 in range(n_cand):
            for j2 in range(j1 + 1, n_cand):
                all_resp_sims.append(resp_sim[j1, j2])
                all_grad_sims.append(grad_cos_sim[j1, j2])
                all_same_mode.append(float(modes[i, j1] == modes[i, j2]))

    all_resp_sims = np.array(all_resp_sims)
    all_grad_sims = np.array(all_grad_sims)
    all_same_mode = np.array(all_same_mode)

    r_pearson, p_pearson = stats.pearsonr(all_resp_sims, all_grad_sims)
    same_grad = all_grad_sims[all_same_mode == 1]
    diff_grad = all_grad_sims[all_same_mode == 0]

    print(f"  Pearson r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"  Same-mode grad sim: {same_grad.mean():.4f} +/- {same_grad.std():.4f}")
    print(f"  Cross-mode grad sim: {diff_grad.mean():.4f} +/- {diff_grad.std():.4f}")

    # k_eff
    rewards = compute_rewards(prompts, responses)
    k_list = [2, 4, 8]
    strat_fns = {
        "Random": lambda k_: select_random(N, n_cand, k_, seed=SEED + 777),
        "Top-$k$": lambda k_: select_best(rewards, k_),
        "GRADES": lambda k_: select_diverse(responses, rewards, k_),
    }
    keff_results = {s: [] for s in strat_fns}
    for k_ in k_list:
        for name, fn in strat_fns.items():
            sel = fn(k_)
            avg_sim_list = []
            for i in range(N):
                sel_vals = responses[i, sel[i], 0]
                if len(sel_vals) < 2:
                    avg_sim_list.append(1.0)
                    continue
                d = np.abs(sel_vals[:, None] - sel_vals[None, :])
                max_d = (
                    np.max(np.abs(responses[i, :, 0][:, None] - responses[i, :, 0][None, :]))
                    + 1e-8
                )
                sims = 1.0 - d / max_d
                n_pairs = k_ * (k_ - 1)
                avg_sim_list.append((sims.sum() - k_) / max(n_pairs, 1))
            avg_sim_overall = np.mean(avg_sim_list)
            keff_results[name].append(k_ / (1 + (k_ - 1) * max(avg_sim_overall, 0)))

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: scatter + binned means
    n_bins = 15
    bin_edges = np.linspace(all_resp_sims.min(), all_resp_sims.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means, bin_stds = [], []
    for b in range(n_bins):
        mask = (all_resp_sims >= bin_edges[b]) & (all_resp_sims < bin_edges[b + 1])
        if mask.sum() > 5:
            bin_means.append(np.mean(all_grad_sims[mask]))
            bin_stds.append(np.std(all_grad_sims[mask]) / np.sqrt(mask.sum()))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    sub = np.random.choice(len(all_resp_sims), min(3000, len(all_resp_sims)), replace=False)
    axes[0].scatter(all_resp_sims[sub], all_grad_sims[sub], alpha=0.12, s=6,
                    color="steelblue", rasterized=True)
    valid = ~np.isnan(bin_means)
    axes[0].errorbar(
        np.array(bin_centers)[valid], np.array(bin_means)[valid],
        yerr=np.array(bin_stds)[valid], color="red", lw=2.5, capsize=3,
        marker="o", ms=7, label=f"Binned mean ($r$={r_pearson:.2f})",
    )
    axes[0].set_xlabel("Response similarity")
    axes[0].set_ylabel("Gradient cosine similarity")
    axes[0].set_title("Similarity $\\rightarrow$ Gradient Alignment")
    axes[0].legend()

    # Right: k_eff
    markers_keff = {"Random": "^", "Top-$k$": "s", "GRADES": "o"}
    colors_keff = {"Random": COLORS["random"], "Top-$k$": COLORS["best"],
                   "GRADES": COLORS["diverse"]}
    for name in strat_fns:
        axes[1].plot(k_list, keff_results[name], marker=markers_keff[name],
                     color=colors_keff[name], lw=2.5, ms=9, label=name)
    axes[1].plot(k_list, k_list, "k--", alpha=0.3, lw=1.5,
                 label="$k_{\\mathrm{eff}}=k$ (ideal)")
    axes[1].set_xlabel("$k$ (selected responses)")
    axes[1].set_ylabel("$k_{\\mathrm{eff}}$")
    axes[1].set_title("Effective Sample Size by Strategy")
    axes[1].legend()
    axes[1].set_xticks(k_list)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp6_bridge.pdf")
    fig.savefig(RESULTS_DIR / "exp6_bridge.png")
    plt.close(fig)
    print("  Saved: exp6_bridge.pdf/.png")
    return r_pearson, keff_results


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7 — Discrete-Token Transformer
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenSimConfig:
    seed: int = 0
    n_train_prompts: int = 200
    n_test_prompts: int = 200
    prompt_len: int = 4
    resp_len: int = 6
    n_candidates: int = 32
    k_values: Tuple[int, ...] = (1, 2, 4, 8)
    n_repeats: int = 5
    n_test_samples: int = 20
    modes: Tuple[str, ...] = ("A", "B", "C")
    true_mode_probs: Tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    gen_mode_probs: Tuple[float, ...] = (0.95, 0.04, 0.01)
    reward_means: Tuple[float, ...] = (1.00, 0.97, 0.96)
    reward_noise: float = 0.02
    tau_quantile: float = 0.30
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    lr: float = 2e-3
    n_steps: int = 600
    batch_size: int = 64


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.tr(h, mask=mask)
        return self.lm_head(h)


def _token_build_vocab(cfg):
    vocab = {"<pad>": 0, "<bos>": 1, "<sep>": 2, "<eos>": 3}
    for i in range(64):
        vocab[f"px{i}"] = len(vocab)
    for m in cfg.modes:
        vocab[f"m{m}"] = len(vocab)
    for i in range(128):
        vocab[f"c{i}"] = len(vocab)
    return vocab


def _token_sample_prompts(n, cfg, vocab):
    pts = [vocab[f"px{i}"] for i in range(64)]
    return [random.sample(pts, cfg.prompt_len) for _ in range(n)]


def _token_response(prompt, mode, cfg, vocab):
    mode_tok = vocab[f"m{mode}"]
    cbase = sum(prompt) + (ord(mode) * 13)
    content = [vocab[f"c{(cbase + 17*t) % 128}"] for t in range(cfg.resp_len - 2)]
    return [mode_tok] + content + [vocab["<eos>"]]


def _token_sample_mode(modes, probs):
    r = random.random()
    c = 0.0
    for m, p in zip(modes, probs):
        c += p
        if r <= c:
            return m
    return modes[-1]


def _token_gen_candidates(prompt, cfg, vocab):
    cands = []
    for _ in range(cfg.n_candidates):
        mode = _token_sample_mode(cfg.modes, cfg.gen_mode_probs)
        resp = _token_response(prompt, mode, cfg, vocab)
        mean_r = cfg.reward_means[cfg.modes.index(mode)]
        reward = mean_r + random.gauss(0, cfg.reward_noise)
        cands.append((resp, mode, reward))
    return cands


def _token_select_random(cands, k):
    return random.sample(cands, k)

def _token_select_best(cands, k):
    return sorted(cands, key=lambda t: t[2], reverse=True)[:k]

def _token_select_diverse(cands, k, tau):
    rewards = np.array([t[2] for t in cands])
    thr = float(np.quantile(rewards, tau))
    filt = [t for t in cands if t[2] >= thr]
    if len(filt) < k:
        filt = sorted(cands, key=lambda t: t[2], reverse=True)[:k]
    picked = [sorted(filt, key=lambda t: t[2], reverse=True)[0]]
    used_modes = {picked[0][1]}
    while len(picked) < k:
        remaining = [t for t in filt if t not in picked]
        if not remaining:
            break
        best, best_score = None, -1e9
        for t in remaining:
            bonus = 1.0 if t[1] not in used_modes else 0.0
            score = 10.0 * bonus + t[2]
            if score > best_score:
                best_score, best = score, t
        picked.append(best)
        used_modes.add(best[1])
    return picked


def _token_build_seqs(prompts, responses, vocab, cfg):
    bos, sep, pad = vocab["<bos>"], vocab["<sep>"], vocab["<pad>"]
    max_len = 1 + cfg.prompt_len + 1 + cfg.resp_len
    seqs = []
    for p, r in zip(prompts, responses):
        s = [bos] + p + [sep] + r
        s = s + [pad] * (max_len - len(s))
        seqs.append(s)
    return torch.tensor(seqs, dtype=torch.long, device=DEVICE)


def _token_nll(model, seqs, pad_id):
    model.eval()
    with torch.no_grad():
        logits = model(seqs[:, :-1])
        targets = seqs[:, 1:]
        return float(
            nn.CrossEntropyLoss(ignore_index=pad_id, reduction="mean")(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            ).item()
        )


def _token_train(model, seqs, pad_id, cfg):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    n = seqs.size(0)
    for _ in range(cfg.n_steps):
        idx = torch.randint(0, n, (min(cfg.batch_size, n),))
        batch = seqs[idx]
        opt.zero_grad()
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def experiment_7_token_sim():
    """Discrete-token Transformer simulation with biased generator."""
    print("\n" + "=" * 65)
    print("EXP 7: Discrete-Token Transformer Simulation")
    print("=" * 65)

    cfg = TokenSimConfig()
    all_results = []

    for rep in range(cfg.n_repeats):
        seed = cfg.seed + 1000 * rep
        set_seed(seed)
        vocab = _token_build_vocab(cfg)
        pad_id = vocab["<pad>"]
        max_len = 1 + cfg.prompt_len + 1 + cfg.resp_len
        vocab_size = len(vocab)

        train_prompts = _token_sample_prompts(cfg.n_train_prompts, cfg, vocab)
        test_prompts = _token_sample_prompts(cfg.n_test_prompts, cfg, vocab)
        train_cands = [_token_gen_candidates(p, cfg, vocab) for p in train_prompts]

        strategies = {
            "Top-$k$": lambda c, k: _token_select_best(c, k),
            "Random": lambda c, k: _token_select_random(c, k),
            "GRADES": lambda c, k: _token_select_diverse(c, k, cfg.tau_quantile),
        }

        run_results = {s: {} for s in strategies}
        for k in cfg.k_values:
            for sname, sfn in strategies.items():
                tr_p, tr_r = [], []
                for p, cands in zip(train_prompts, train_cands):
                    for resp, _m, _rew in sfn(cands, k):
                        tr_p.append(p)
                        tr_r.append(resp)
                seqs = _token_build_seqs(tr_p, tr_r, vocab, cfg)
                model = TinyCausalTransformer(
                    vocab_size, cfg.d_model, cfg.n_heads, cfg.n_layers,
                    cfg.d_ff, cfg.dropout, max_len,
                ).to(DEVICE)
                _token_train(model, seqs, pad_id, cfg)

                ev_p, ev_r = [], []
                for p in test_prompts:
                    for _ in range(cfg.n_test_samples):
                        mode = _token_sample_mode(cfg.modes, cfg.true_mode_probs)
                        ev_p.append(p)
                        ev_r.append(_token_response(p, mode, cfg, vocab))
                ev_seqs = _token_build_seqs(ev_p, ev_r, vocab, cfg)
                run_results[sname][k] = _token_nll(model, ev_seqs, pad_id)
        all_results.append(run_results)
        print(f"  Rep {rep+1}/{cfg.n_repeats}")

    # Aggregate
    strat_names = list(all_results[0].keys())
    agg = {s: {} for s in strat_names}
    for s in strat_names:
        for k in cfg.k_values:
            vals = [r[s][k] for r in all_results]
            agg[s][k] = (float(np.mean(vals)), float(np.std(vals)))

    # Print table
    print(f"\n  {'Strategy':>22s} | ", end="")
    for k in cfg.k_values:
        print(f"{'k='+str(k):>12s}", end=" ")
    print()
    print("  " + "-" * 75)
    for s in strat_names:
        print(f"  {s:>22s} | ", end="")
        for k in cfg.k_values:
            m, st = agg[s][k]
            print(f"{m:5.3f}+/-{st:.3f}", end=" ")
        print()

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = list(cfg.k_values)
    colors_t = {"Top-$k$": COLORS["best"],
                "Random": COLORS["random"],
                "GRADES": COLORS["diverse"]}
    markers_t = {"Top-$k$": "s", "Random": "^",
                 "GRADES": "o"}
    for s in strat_names:
        means = [agg[s][k][0] for k in ks]
        stds_ = [agg[s][k][1] for k in ks]
        ax.errorbar(ks, means, yerr=stds_, marker=markers_t[s],
                    color=colors_t[s], lw=2, capsize=4, label=s)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$k$ (responses per prompt)")
    ax.set_ylabel("Test NLL (true uniform $P(\\mathbf{y}|\\mathbf{x})$)")
    ax.set_title("Discrete-Token Simulation: Biased Generator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp7_token_sim.pdf")
    fig.savefig(RESULTS_DIR / "exp7_token_sim.png")
    plt.close(fig)
    print("  Saved: exp7_token_sim.pdf/.png")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated subset of experiment ids "
                             "to run (e.g. '1,2,5'); empty = run all.")
    args = parser.parse_args()
    runners = {
        "1": experiment_1_variance,
        "2": experiment_2_budget,
        "3": experiment_3_mode_lottery,
        "4": experiment_4_distributional,
        "5": experiment_5_gradient_emse,
        "6": experiment_6_bridge,
        "7": experiment_7_token_sim,
    }
    if args.only:
        ids = [s.strip() for s in args.only.split(",") if s.strip()]
    else:
        ids = list(runners.keys())

    print("Multi-Response Training — Unified Experiment Suite")
    print("=" * 65)
    for i in ids:
        runners[i]()

    print("\n" + "=" * 65)
    print("DONE.")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
