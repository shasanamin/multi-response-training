#!/usr/bin/env python
"""Generate alpha-sweep DBKoN configs at fixed k for the publication setup.

For each (k, alpha) in the chosen grid, copy the publication-suite gold_full
template and override `selection.quality_weight_alpha` and run name.

Outputs go to configs/experiments/alpha_sweep/.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "configs" / "experiments" / "publication_suite"
OUT_DIR = ROOT / "configs" / "experiments" / "alpha_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
K_VALUES = [2, 4, 8]


def fmt_alpha(a: float) -> str:
    return f"{a:.2f}".replace(".", "p").rstrip("0").rstrip("p") if "." in f"{a:.2f}" else f"{a:.0f}"


def fmt_alpha_label(a: float) -> str:
    s = f"{a:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def main() -> None:
    written = []
    for k in K_VALUES:
        tpl = TEMPLATE_DIR / f"gold_gens_llama31_8b_instruct_dbkon_k{k}_gold_full.yaml"
        with tpl.open() as fh:
            base = yaml.safe_load(fh)
        for a in ALPHAS:
            cfg = yaml.safe_load(yaml.safe_dump(base))  # deep copy
            cfg["selection"]["quality_weight_alpha"] = float(a)
            tag = fmt_alpha_label(a)
            run_name = f"gold_gens_llama31_8b_instruct_dbkon_k{k}_alpha{tag}_alpha_sweep"
            cfg["run"]["name"] = run_name
            out = OUT_DIR / f"{run_name}.yaml"
            with out.open("w") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False)
            written.append((k, a, out))
    for k, a, p in written:
        print(f"k={k} alpha={a} -> {p.relative_to(ROOT)}")
    print(f"[gen] wrote {len(written)} configs to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
