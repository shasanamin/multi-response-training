from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT  # noqa: F401


DEFAULT_MANIFEST = "configs/experiments/mosaic_suite/mosaic_llama31_8b_full_v1_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default="results/analysis/mosaic")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in payload.get("experiments", [])]


def metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def load_summary(experiment: dict[str, Any]) -> dict[str, Any] | None:
    path = PROJECT_ROOT / "results" / "runs" / experiment["run_name"] / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    summary = load_summary(experiment)
    row = dict(experiment)
    row["status"] = "missing" if summary is None else "complete"
    if summary is None:
        return row

    eval_metrics = summary.get("evaluation_metrics", {})
    counts = summary.get("counts", {})
    row.update(
        {
            "train_prompts": counts.get("train_prompts"),
            "train_pairs": counts.get("train_pairs"),
            "validation_pairs": counts.get("validation_pairs"),
            "test_pairs": counts.get("test_pairs"),
            "reference_loss": metric(eval_metrics, "reference_loss"),
            "reference_perplexity": metric(eval_metrics, "reference_perplexity"),
            "coverage_at8": metric(
                eval_metrics,
                "multi_k8_reference_coverage_mean",
                "multi_reference_coverage_mean",
            ),
            "diversity_at8": metric(
                eval_metrics,
                "multi_k8_semantic_diversity_mean",
                "multi_semantic_diversity_mean",
            ),
            "response_words_at8": metric(
                eval_metrics,
                "multi_k8_response_words_mean",
                "multi_response_words_mean",
            ),
        }
    )
    return row


def fit_loss_law(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fit_df = df[
        (df["status"] == "complete")
        & (df["experiment_tag"] == "selector")
        & (df["prompt_variants_per_cluster"] == 1)
        & df["reference_loss"].notna()
    ]
    for strategy, group in fit_df.groupby("strategy"):
        group = group.sort_values("k")
        if len(group) < 3:
            continue
        k = group["k"].to_numpy(dtype=float)
        y = group["reference_loss"].to_numpy(dtype=float)
        x = 1.0 / k
        coeffs, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)
        a_hat, b_hat = coeffs
        pred = a_hat + b_hat * x
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {
                "strategy": strategy,
                "a_hat": float(a_hat),
                "b_hat": float(b_hat),
                "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                "vy_over_vx_proxy": float(b_hat / a_hat) if a_hat > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def print_tables(df: pd.DataFrame, fit_df: pd.DataFrame) -> None:
    complete = df[df["status"] == "complete"].copy()
    print(f"Completed runs: {len(complete)}/{len(df)}")
    if complete.empty:
        return

    columns = [
        "run_name",
        "experiment_tag",
        "strategy",
        "prompt_variants_per_cluster",
        "response_mode",
        "k",
        "train_pairs",
        "reference_loss",
        "coverage_at8",
        "diversity_at8",
    ]
    print("\nMOSAIC summary:")
    print(complete[columns].sort_values(["experiment_tag", "strategy", "k"]).to_string(index=False))

    triad = complete[
        complete["experiment_tag"].isin(["selector", "hybrid", "implicit", "same"])
        & (
            ((complete["prompt_variants_per_cluster"] == 1) & (complete["k"] == 4))
            | ((complete["prompt_variants_per_cluster"] == 2) & (complete["k"] == 2))
            | ((complete["prompt_variants_per_cluster"] == 4) & (complete["k"] == 1))
        )
    ]
    if not triad.empty:
        print("\nMatched p*k=4 redundancy comparison:")
        print(
            triad[
                [
                    "run_name",
                    "experiment_tag",
                    "response_mode",
                    "train_pairs",
                    "reference_loss",
                    "coverage_at8",
                    "diversity_at8",
                ]
            ].sort_values("reference_loss").to_string(index=False)
        )

    if not fit_df.empty:
        print("\nExact-prompt loss law fits, loss ~= a + b/k:")
        print(fit_df.to_string(index=False))


def main() -> None:
    args = parse_args()
    experiments = load_manifest(PROJECT_ROOT / args.manifest)
    rows = [row_from_experiment(experiment) for experiment in experiments]
    df = pd.DataFrame(rows)
    fit_df = fit_loss_law(df)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "mosaic_suite_summary.csv", index=False)
    fit_df.to_csv(out_dir / "mosaic_loss_law_fits.csv", index=False)
    print_tables(df, fit_df)
    print(f"\nWrote {out_dir / 'mosaic_suite_summary.csv'}")
    print(f"Wrote {out_dir / 'mosaic_loss_law_fits.csv'}")


if __name__ == "__main__":
    main()
