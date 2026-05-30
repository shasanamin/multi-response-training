# Multi-Response Training (MRT)

Code and data for the paper "**Escaping the Mode Lottery: Multi-Response Training Improves Language Model Generalization**".

## Repository Layout

```
├── README.md                       # this file
├── pyproject.toml                  # installable package definition
├── requirements.txt                # pinned dependencies
├── data/
│   └── mosaic-1k.jsonl             # the MOSAIC-1K benchmark
├── src/                            
│   ├── config.py                   # YAML schema and resolution
│   ├── data.py                     # native multi-response prompt-pool loaders
│   ├── selection.py                # RKoN, BKoN, DKoN, DBKoN selectors
│   ├── train.py                    # LoRA / full SFT trainer
│   ├── reward.py                   # reward-model scoring with caching
│   ├── embeddings.py               # sentence-transformer embedding cache
│   ├── evaluation.py               # held-out NLL + multi-sample generation metrics
│   ├── formatting.py               # chat-template handling per model family
│   ├── jsonl_io.py / paths.py / runtime.py
│   ├── reporting.py                
│   └── utils.py
├── configs/
│   ├── runtime/local.yaml          # cache / scratch path defaults
│   ├── datasets/                   # per-dataset preparation configs
│   └── experiments/                # experiment configs used in the paper
│       ├── publication_suite/      # Gold full-data selector grid (main table)
│       ├── gold_cross_family_suite/# RKoN cross-family scaling (Llama / Qwen / Gemma)
│       ├── gold_base_pretrained_suite/ # base vs. instruct comparison
│       ├── mosaic_suite/           # implicit multi-objective (Mosaic-1k)
│       ├── alpha_sweep/            # DBKoN diversity–quality trade-off (alpha)
│       ├── code_contests/          # Code Contests pass@k validation
│       └── dataset_smoke/          # tiny configs for a quick pipeline check
├── controlled-val/
│   ├── mrt_experiments.py          # controlled CPU-scale validation
│   └── results/                    # simulation outputs (JSON consumed by figures)
├── results/
│   └── analysis/                   # aggregated metric tables consumed by figures
└── scripts/
    ├── run_experiment.py           # single-config training + eval entry point
    ├── make_paper_figures.py       # regenerate all paper figures
    ├── generate_*_configs.py       # deterministic config materializers per suite
    ├── analyze_*.py                # aggregation behind each table
    ├── aggregate_alpha_sweep.py    # alpha-sweep aggregation
    ├── evaluate_*.py               # post-hoc eval utilities (checkpoints, pass@k)
    ├── activate_env.sh / create_env.sh # environment wiring
    └── _bootstrap.py               # puts src/ on sys.path for scripts
```

## Quickstart

### 1. Reproduce the figures (no GPU required)

The five figures in the paper are regenerated from artifacts bundled in this archive (`controlled-val/results/*.json` and `results/analysis/*.csv`):

```bash
pip install -r requirements.txt
python scripts/make_paper_figures.py
```

### 2. Run the controlled validation (CPU, minutes)

```bash
python controlled-val/mrt_experiments.py
```

### 3. Run a full experiment (GPU)

```bash
bash scripts/create_env.sh                  # creates a venv with pinned deps
source scripts/activate_env.sh              # exports cache paths, activates venv
export HF_TOKEN=<your_huggingface_token>    # gated Llama / Skywork checkpoints

python scripts/run_experiment.py \
  --config configs/experiments/publication_suite/<one_config>.yaml \
  --runtime-config configs/runtime/local.yaml
```

To relocate caches and run outputs on a cluster, set `SCRATCH_ROOT` before sourcing `activate_env.sh`. A tiny end-to-end check is available via the configs in `configs/experiments/dataset_smoke/`.

## Determinism and Caching Notes

- Every run materializes a per-seed `prepared_path` so multi-GPU launches do not race on a shared cache.
- Reward-model scores and sentence embeddings are cached on disk per `(dataset, base_model, candidate_subset)`, so sweeps over `k` and selector reuse them.
- Each `summary.json` records the full merged config, dataset preparation metadata (caps, splits, seed), reward / embedding model identifiers, and all metrics.
- `scripts/extract_comprehensive_metrics.py` walks `results/runs/*/summary.json` to produce the consolidated CSVs used for the tables.
