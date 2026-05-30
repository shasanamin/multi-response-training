#!/usr/bin/env python3
"""
Extract comprehensive metrics from all publication run summary.json files.

Produces a consolidated CSV and prints summary tables to stdout.
"""

import json
import os
import re
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUTPUT_CSV = OUTPUT_DIR / "comprehensive_reward_metrics.csv"

# ── Publication run patterns ──────────────────────────────────────────────────
# We match directories that are clearly publication runs:
#   gold_gens_*_gold_full, gold_gens_*_paper, gold_gens_*_publication,
#   gold_gens_*_equal_examples_v2,
#   nectar_*_nectar_full, ultrafeedback_*_ultrafeedback_full,
#   code_contests_*, gold_gens_gemma3_*_paper, gold_gens_llama32_3b_base_*_paper
PUB_PATTERN = re.compile(
    r"^("
    r"gold_gens_.*_(gold_full|paper|publication|equal_examples_v2)"
    r"|nectar_.*_nectar_full"
    r"|ultrafeedback_.*_ultrafeedback_full"
    r"|code_contests_\w+_k\d+"
    r")$"
)

# ── Strategy / k extraction ──────────────────────────────────────────────────
STRATEGY_K_RE = re.compile(r"(rkon|bkon|dkon|dbkon)_k(\d+)")


def classify_dataset(run_name: str) -> str:
    """Determine dataset label from run directory name."""
    if run_name.startswith("code_contests_"):
        return "code_contests"
    if run_name.startswith("nectar_"):
        return "nectar"
    if run_name.startswith("ultrafeedback_"):
        return "ultrafeedback"
    # Gold variants
    if "gemma3_4b_it" in run_name:
        return "gold_gemma3_4b_it"
    if "gemma3_4b_pt" in run_name:
        return "gold_gemma3_4b_pt"
    if "llama32_3b_base" in run_name:
        return "gold_llama32_3b_base"
    if "llama32_3b_instruct" in run_name:
        # sub-classify by setting
        if "compute_matched50" in run_name:
            return "gold_llama32_3b_compute_matched"
        return "gold_llama32_3b"
    if "llama31_8b_instruct" in run_name:
        if "gold_full" in run_name:
            return "gold_llama31_8b_full"
        if "compute_matched50" in run_name or "publication" in run_name:
            return "gold_llama31_8b_compute_matched"
        if "equal_examples" in run_name:
            return "gold_llama31_8b_equal_opp"
        return "gold_llama31_8b"
    return "unknown"


def extract_metrics(summary: dict, run_name: str) -> dict:
    """Pull all requested metrics out of a summary.json dict."""
    em = summary.get("evaluation_metrics", {})

    # Strategy and k from run name
    m = STRATEGY_K_RE.search(run_name)
    strategy = m.group(1).upper() if m else "UNKNOWN"
    k = int(m.group(2)) if m else -1

    row = {
        "run_name": run_name,
        "dataset": classify_dataset(run_name),
        "strategy": strategy,
        "k": k,
        # Core metrics
        "reference_loss": em.get("reference_loss"),
        "reference_perplexity": em.get("reference_perplexity"),
        "single_reward_mean": em.get("single_reward_mean"),
    }

    # Multi-K reward mean and best at each evaluation K
    for eval_k in [2, 4, 8, 16]:
        prefix = f"multi_k{eval_k}"
        row[f"{prefix}_reward_mean"] = em.get(f"{prefix}_reward_mean")
        row[f"{prefix}_reward_best_mean"] = em.get(f"{prefix}_reward_best_mean")
        row[f"{prefix}_reward_prompt_mean"] = em.get(f"{prefix}_reward_prompt_mean")
        row[f"{prefix}_semantic_diversity_mean"] = em.get(f"{prefix}_semantic_diversity_mean")
        row[f"{prefix}_reference_coverage_mean"] = em.get(f"{prefix}_reference_coverage_mean")
        row[f"{prefix}_reference_alignment_mean"] = em.get(f"{prefix}_reference_alignment_mean")
        row[f"{prefix}_response_chars_mean"] = em.get(f"{prefix}_response_chars_mean")
        row[f"{prefix}_response_words_mean"] = em.get(f"{prefix}_response_words_mean")

        # Check for reward_min (worst-of-K)
        row[f"{prefix}_reward_min"] = em.get(f"{prefix}_reward_min")

    # Also grab the un-prefixed multi_ fields (= k16 duplicates usually)
    row["multi_reward_mean"] = em.get("multi_reward_mean")
    row["multi_reward_best_mean"] = em.get("multi_reward_best_mean")
    row["multi_semantic_diversity_mean"] = em.get("multi_semantic_diversity_mean")
    row["multi_reference_coverage_mean"] = em.get("multi_reference_coverage_mean")

    # Single response length
    row["single_response_chars_mean"] = em.get("single_response_chars_mean")
    row["single_response_words_mean"] = em.get("single_response_words_mean")

    # Computed: reward spread = best-of-16 reward minus mean reward at k=16
    best16 = em.get("multi_k16_reward_best_mean")
    mean16 = em.get("multi_k16_reward_mean")
    if best16 is not None and mean16 is not None:
        row["reward_spread_k16"] = best16 - mean16
    else:
        row["reward_spread_k16"] = None

    # Counts
    counts = summary.get("counts", {})
    row["train_prompts"] = counts.get("train_prompts")
    row["train_pairs"] = counts.get("train_pairs")

    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    has_reward_min = False

    for entry in sorted(os.listdir(RUNS_DIR)):
        if not PUB_PATTERN.match(entry):
            continue
        summary_path = RUNS_DIR / entry / "summary.json"
        if not summary_path.exists():
            missing.append(entry)
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        row = extract_metrics(summary, entry)
        rows.append(row)

        # Check if any run has reward_min
        if any(v is not None for k_name, v in row.items() if "reward_min" in k_name):
            has_reward_min = True

    if not rows:
        print("ERROR: No publication runs found!")
        return

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    if missing:
        print(f"WARNING: {len(missing)} dirs had no summary.json: {missing}")

    # ── Reward min availability ───────────────────────────────────────────────
    print()
    if has_reward_min:
        print("NOTE: multi_k*_reward_min (worst-of-K) IS available in some runs.")
    else:
        print("NOTE: multi_k*_reward_min (worst-of-K) is NOT available in any summary.json.")
        print("      The evaluation pipeline does not currently compute per-prompt min reward.")

    # ── Summary tables ────────────────────────────────────────────────────────
    # Group by dataset
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    # Determine column widths
    hdr_fmt = "{strat:<8s} {k:>3s}  {ref_loss:>10s}  {rwd_mean:>10s}  {rwd_best:>10s}  {spread:>10s}  {coverage:>10s}  {diversity:>10s}"
    row_fmt = "{strat:<8s} {k:>3d}  {ref_loss:>10s}  {rwd_mean:>10s}  {rwd_best:>10s}  {spread:>10s}  {coverage:>10s}  {diversity:>10s}"

    def fmt(val, decimals=4):
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}"

    print()
    print("=" * 100)
    print("COMPREHENSIVE REWARD METRICS SUMMARY")
    print("=" * 100)
    print()
    print("Columns: ref_loss = reference loss on held-out test set")
    print("         rwd_mean = multi_k16_reward_mean (avg reward across 16 generations)")
    print("         rwd_best = multi_k16_reward_best_mean (best-of-16 reward)")
    print("         spread   = rwd_best - rwd_mean (reward spread; wider = more diverse)")
    print("         coverage = multi_k16_reference_coverage_mean")
    print("         diversity = multi_k16_semantic_diversity_mean")
    print()

    for dataset in sorted(by_dataset.keys()):
        ds_rows = by_dataset[dataset]
        # Sort by strategy then k
        strat_order = {"RKON": 0, "BKON": 1, "DKON": 2, "DBKON": 3}
        ds_rows.sort(key=lambda r: (strat_order.get(r["strategy"], 9), r["k"]))

        print(f"──── {dataset} ({len(ds_rows)} runs) " + "─" * (80 - len(dataset)))
        print(hdr_fmt.format(
            strat="Strategy", k="k",
            ref_loss="ref_loss", rwd_mean="rwd_mean",
            rwd_best="rwd_best", spread="spread",
            coverage="coverage", diversity="diversity"
        ))
        print("-" * 100)
        for r in ds_rows:
            print(row_fmt.format(
                strat=r["strategy"],
                k=r["k"],
                ref_loss=fmt(r["reference_loss"]),
                rwd_mean=fmt(r.get("multi_k16_reward_mean")),
                rwd_best=fmt(r.get("multi_k16_reward_best_mean")),
                spread=fmt(r.get("reward_spread_k16")),
                coverage=fmt(r.get("multi_k16_reference_coverage_mean")),
                diversity=fmt(r.get("multi_k16_semantic_diversity_mean")),
            ))
        print()

    # ── Detailed reward-at-each-K table for one key dataset ───────────────────
    print("=" * 120)
    print("REWARD PROGRESSION ACROSS EVALUATION K (for each run)")
    print("  Shows how mean reward and best-of-K reward scale with # generations at inference time")
    print("=" * 120)
    print()

    rk_hdr = "{run:<65s} {sr:>8s}  {m2:>8s}  {m4:>8s}  {m8:>8s}  {m16:>8s}  {b2:>8s}  {b4:>8s}  {b8:>8s}  {b16:>8s}"
    rk_row = "{run:<65s} {sr:>8s}  {m2:>8s}  {m4:>8s}  {m8:>8s}  {m16:>8s}  {b2:>8s}  {b4:>8s}  {b8:>8s}  {b16:>8s}"

    for dataset in sorted(by_dataset.keys()):
        ds_rows = by_dataset[dataset]
        ds_rows.sort(key=lambda r: (strat_order.get(r["strategy"], 9), r["k"]))

        print(f"──── {dataset} " + "─" * (100 - len(dataset)))
        print(rk_hdr.format(
            run="Run (strategy/k)", sr="single",
            m2="mean@2", m4="mean@4", m8="mean@8", m16="mean@16",
            b2="best@2", b4="best@4", b8="best@8", b16="best@16"
        ))
        print("-" * 160)
        for r in ds_rows:
            label = f"{r['strategy']}/k={r['k']}"
            print(rk_row.format(
                run=label,
                sr=fmt(r.get("single_reward_mean"), 2),
                m2=fmt(r.get("multi_k2_reward_mean"), 2),
                m4=fmt(r.get("multi_k4_reward_mean"), 2),
                m8=fmt(r.get("multi_k8_reward_mean"), 2),
                m16=fmt(r.get("multi_k16_reward_mean"), 2),
                b2=fmt(r.get("multi_k2_reward_best_mean"), 2),
                b4=fmt(r.get("multi_k4_reward_best_mean"), 2),
                b8=fmt(r.get("multi_k8_reward_best_mean"), 2),
                b16=fmt(r.get("multi_k16_reward_best_mean"), 2),
            ))
        print()

    # ── Aggregate: mean across strategies within each dataset ─────────────────
    print("=" * 100)
    print("AGGREGATE: Mean reference_loss by dataset / strategy / k")
    print("=" * 100)
    print()

    agg_hdr = "{dataset:<35s} {strat:<8s} {k:>3s}  {ref_loss:>10s}  {n:>3s}"
    agg_row = "{dataset:<35s} {strat:<8s} {k:>3d}  {ref_loss:>10s}  {n:>3d}"

    # Group by (dataset, strategy, k)
    agg = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["strategy"], r["k"])
        if r["reference_loss"] is not None:
            agg[key].append(r["reference_loss"])

    print(agg_hdr.format(dataset="Dataset", strat="Strategy", k="k", ref_loss="ref_loss", n="n"))
    print("-" * 70)
    for (dataset, strategy, k) in sorted(agg.keys()):
        vals = agg[(dataset, strategy, k)]
        mean_val = sum(vals) / len(vals)
        print(agg_row.format(
            dataset=dataset, strat=strategy, k=k,
            ref_loss=fmt(mean_val), n=len(vals)
        ))

    print()
    print(f"\nTotal publication runs processed: {len(rows)}")
    print(f"CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
