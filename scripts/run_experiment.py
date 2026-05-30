from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from _bootstrap import PROJECT_ROOT  # noqa: F401

from config import dump_yaml, load_config
from data import (
    flatten_all_pairs,
    flatten_selected_pairs,
    load_candidate_pools,
    pools_by_split,
    prepare_code_contests_dataset,
    prepare_gold_labelled_gens_dataset,
    prepare_helpsteer2_dataset,
    prepare_mosaic_dataset,
    prepare_nectar_dataset,
    prepare_ultrafeedback_dataset,
    reference_texts_by_prompt,
    sample_pool_responses,
)
from embeddings import SentenceEmbedder
from evaluation import evaluate_checkpoint
from jsonl_io import write_jsonl
from models import resolve_model_config
from reward import RewardModelScorer
from reporting import rebuild_summary_index, write_project_run_artifacts
from runtime import apply_runtime_environment, ensure_runtime_directories, run_root
from selection import select_response_indices
from train import train_model
from utils import seed_everything, slugify, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-config", default="configs/runtime/cluster.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    return parser.parse_args()


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _prepared_pool_path(config: dict) -> Path:
    data_cfg = config["data"]
    if data_cfg.get("prepared_path"):
        return Path(data_cfg["prepared_path"])
    if data_cfg.get("source") == "local_mosaic":
        return Path(config["paths"]["data_root"]) / "mosaic" / "prompt_pools.jsonl"
    return Path(config["paths"]["data_root"]) / "gold_labelled_gens" / "prompt_pools.jsonl"


def _prepared_meta_path(prepared_path: Path) -> Path:
    if prepared_path.suffix:
        return prepared_path.with_suffix(f"{prepared_path.suffix}.meta.json")
    return Path(f"{prepared_path}.meta.json")


def _expected_prepare_metadata(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    return {
        "source": data_cfg.get("source"),
        "hf_dataset_id": data_cfg.get("hf_dataset_id"),
        "hf_subset": data_cfg.get("hf_subset"),
        "hf_split": data_cfg.get("hf_split", "validation"),
        "seed": int(config["run"].get("seed", 7)),
        "min_responses": int(data_cfg.get("min_responses", 2)),
        "train_fraction": float(data_cfg.get("train_fraction", 0.8)),
        "validation_fraction": float(data_cfg.get("validation_fraction", 0.1)),
        "max_train_prompts": data_cfg.get("max_train_prompts"),
        "max_validation_prompts": data_cfg.get("max_validation_prompts"),
        "max_test_prompts": data_cfg.get("max_test_prompts"),
        "raw_path": data_cfg.get("raw_path"),
        "prompt_variants_per_cluster": data_cfg.get("prompt_variants_per_cluster"),
        "eval_prompt_variants_per_cluster": data_cfg.get("eval_prompt_variants_per_cluster"),
        "response_mode": data_cfg.get("response_mode"),
        "eval_response_mode": data_cfg.get("eval_response_mode"),
        "response_index": data_cfg.get("response_index"),
        "max_train_clusters": data_cfg.get("max_train_clusters"),
        "max_validation_clusters": data_cfg.get("max_validation_clusters"),
        "max_test_clusters": data_cfg.get("max_test_clusters"),
    }


def _prepared_metadata_matches(config: dict[str, Any], prepared_path: Path) -> bool:
    meta_path = _prepared_meta_path(prepared_path)
    if not prepared_path.exists() or not meta_path.exists():
        return False
    try:
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = _expected_prepare_metadata(config)
    comparable_actual = {key: actual.get(key) for key in expected}
    return comparable_actual == expected


def _write_prepare_metadata(config: dict[str, Any], prepared_path: Path) -> None:
    metadata = _expected_prepare_metadata(config)
    metadata["prepared_at"] = utc_now_iso()
    meta_path = _prepared_meta_path(prepared_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _prepare_candidate_pools(config: dict, force_prepare: bool) -> Path:
    data_cfg = config["data"]
    prepared_path = _prepared_pool_path(config)
    if prepared_path.exists() and not force_prepare and _prepared_metadata_matches(config, prepared_path):
        return prepared_path
    if prepared_path.exists() and not force_prepare:
        print(
            f"Re-preparing candidate pools at {prepared_path} because the existing file "
            "does not match the current data-preparation configuration."
        )

    source = data_cfg.get("source")
    prepare_kwargs = {
        "hf_dataset_id": data_cfg.get("hf_dataset_id"),
        "hf_split": data_cfg.get("hf_split", "validation"),
        "hf_subset": data_cfg.get("hf_subset"),
        "output_path": prepared_path,
        "seed": int(config["run"].get("seed", 7)),
        "min_responses": int(data_cfg.get("min_responses", 2)),
        "train_fraction": float(data_cfg.get("train_fraction", 0.8)),
        "validation_fraction": float(data_cfg.get("validation_fraction", 0.1)),
        "max_train_prompts": data_cfg.get("max_train_prompts"),
        "max_validation_prompts": data_cfg.get("max_validation_prompts"),
        "max_test_prompts": data_cfg.get("max_test_prompts"),
    }

    if source == "hf_gold_labelled_gens":
        prepared = prepare_gold_labelled_gens_dataset(**prepare_kwargs)
    elif source == "hf_ultrafeedback":
        prepared = prepare_ultrafeedback_dataset(**prepare_kwargs)
    elif source == "hf_nectar":
        prepared = prepare_nectar_dataset(**prepare_kwargs)
    elif source == "hf_helpsteer2":
        prepared = prepare_helpsteer2_dataset(**prepare_kwargs)
    elif source == "hf_code_contests":
        prepared = prepare_code_contests_dataset(**prepare_kwargs)
    elif source == "local_mosaic":
        prepared = prepare_mosaic_dataset(
            raw_path=data_cfg.get("raw_path", "data/mosaic-1k.jsonl"),
            output_path=prepared_path,
            seed=int(config["run"].get("seed", 7)),
            train_fraction=float(data_cfg.get("train_fraction", 0.86)),
            validation_fraction=float(data_cfg.get("validation_fraction", 0.04)),
            prompt_variants_per_cluster=int(data_cfg.get("prompt_variants_per_cluster", 1)),
            eval_prompt_variants_per_cluster=int(data_cfg.get("eval_prompt_variants_per_cluster", 4)),
            response_mode=str(data_cfg.get("response_mode", "all")),
            eval_response_mode=str(data_cfg.get("eval_response_mode", "all")),
            response_index=int(data_cfg.get("response_index", 0)),
            max_train_clusters=data_cfg.get("max_train_clusters"),
            max_validation_clusters=data_cfg.get("max_validation_clusters"),
            max_test_clusters=data_cfg.get("max_test_clusters"),
        )
    else:
        raise ValueError(f"Unsupported data source: {source}")

    _write_prepare_metadata(config, prepared)
    return prepared


def _candidate_subset_path(config: dict[str, Any], prepared_path: Path) -> Path | None:
    selection_cfg = config["selection"]
    candidate_limit = selection_cfg.get("n") or selection_cfg.get("n_candidates")
    if not candidate_limit:
        return None
    seed = int(config["run"].get("seed", 7))
    subset_dir = prepared_path.parent / "candidate_subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    stem = prepared_path.name.replace(".", "_")
    return subset_dir / f"{stem}_train_n{int(candidate_limit)}_seed{seed}.jsonl"


def _expected_candidate_subset_metadata(config: dict[str, Any], prepared_path: Path) -> dict[str, Any]:
    selection_cfg = config["selection"]
    return {
        "prepared_pool_path": str(prepared_path),
        "seed": int(config["run"].get("seed", 7)),
        "candidate_limit": int(selection_cfg.get("n") or selection_cfg.get("n_candidates")),
        "source": config["data"].get("source"),
    }


def _candidate_subset_matches(config: dict[str, Any], prepared_path: Path, subset_path: Path) -> bool:
    meta_path = _prepared_meta_path(subset_path)
    if not subset_path.exists() or not meta_path.exists():
        return False
    try:
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = _expected_candidate_subset_metadata(config, prepared_path)
    return {key: actual.get(key) for key in expected} == expected


def _write_candidate_subset_metadata(config: dict[str, Any], prepared_path: Path, subset_path: Path) -> None:
    metadata = _expected_candidate_subset_metadata(config, prepared_path)
    metadata["prepared_at"] = utc_now_iso()
    _prepared_meta_path(subset_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _materialize_train_candidate_pools(
    *,
    config: dict[str, Any],
    prepared_path: Path,
    train_pools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    subset_path = _candidate_subset_path(config, prepared_path)
    if subset_path is None:
        return train_pools
    if _candidate_subset_matches(config, prepared_path, subset_path):
        return load_candidate_pools(subset_path)

    selection_cfg = config["selection"]
    candidate_limit = int(selection_cfg.get("n") or selection_cfg.get("n_candidates"))
    seed = int(config["run"].get("seed", 7))
    rng = np.random.default_rng(seed)

    subset_pools: list[dict[str, Any]] = []
    for pool in train_pools:
        responses = pool["responses"]
        if len(responses) <= candidate_limit:
            subset_pools.append(pool)
            continue
        chosen = sorted(rng.choice(len(responses), size=candidate_limit, replace=False).tolist())
        subset_pool = dict(pool)
        subset_pool["responses"] = [responses[index] for index in chosen]
        subset_pools.append(subset_pool)

    write_jsonl(subset_pools, subset_path)
    _write_candidate_subset_metadata(config, prepared_path, subset_path)
    return subset_pools


def _selection_score_source(config: dict[str, Any]) -> str:
    return str(config.get("selection", {}).get("score_source", "dataset")).lower()


def _strategy_requires_scores(config: dict[str, Any]) -> bool:
    strategy = str(config.get("selection", {}).get("strategy", "rkon")).lower()
    return strategy in {"top_k", "bkon", "best_k", "best_k_of_n", "grades", "dbkon", "diverse_best_k", "diverse_best_k_of_n"}


def _selection_reward_cfg(config: dict[str, Any]) -> dict[str, Any] | None:
    if not _strategy_requires_scores(config):
        return None
    selection_source = _selection_score_source(config)
    reward_cfg = dict(config.get("selection", {}).get("reward_model") or config.get("evaluation", {}).get("reward_model") or {})
    if selection_source == "reward_model":
        if not reward_cfg.get("model_name"):
            raise ValueError("selection.score_source=reward_model requires a reward model config.")
        return reward_cfg
    if selection_source == "auto" and reward_cfg.get("model_name"):
        return reward_cfg
    return None


def _reward_scored_subset_path(subset_path: Path, reward_model_name: str) -> Path:
    return subset_path.with_name(f"{subset_path.stem}_{slugify(reward_model_name)}.jsonl")


def _expected_reward_subset_metadata(
    *,
    config: dict[str, Any],
    subset_path: Path,
    reward_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_subset_path": str(subset_path),
        "reward_model_name": reward_cfg.get("model_name"),
        "reward_batch_size": int(reward_cfg.get("batch_size", 4)),
        "score_source": _selection_score_source(config),
    }


def _reward_subset_matches(config: dict[str, Any], subset_path: Path, scored_path: Path, reward_cfg: dict[str, Any]) -> bool:
    meta_path = _prepared_meta_path(scored_path)
    if not scored_path.exists() or not meta_path.exists():
        return False
    try:
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    expected = _expected_reward_subset_metadata(config=config, subset_path=subset_path, reward_cfg=reward_cfg)
    return {key: actual.get(key) for key in expected} == expected


def _write_reward_subset_metadata(config: dict[str, Any], subset_path: Path, scored_path: Path, reward_cfg: dict[str, Any]) -> None:
    metadata = _expected_reward_subset_metadata(config=config, subset_path=subset_path, reward_cfg=reward_cfg)
    metadata["prepared_at"] = utc_now_iso()
    _prepared_meta_path(scored_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _all_scores_missing(pools: list[dict[str, Any]]) -> bool:
    score_count = 0
    for pool in pools:
        for response in pool["responses"]:
            if response.get("score") is not None:
                score_count += 1
    return score_count == 0


def _materialize_selection_scores(
    *,
    config: dict[str, Any],
    prepared_path: Path,
    train_pools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selection_source = _selection_score_source(config)
    reward_cfg = _selection_reward_cfg(config)
    if reward_cfg is None:
        return train_pools
    if selection_source == "auto" and not _all_scores_missing(train_pools):
        return train_pools

    subset_path = _candidate_subset_path(config, prepared_path)
    cache_anchor = subset_path or prepared_path
    scored_path = _reward_scored_subset_path(cache_anchor, reward_cfg["model_name"])
    if _reward_subset_matches(config, cache_anchor, scored_path, reward_cfg):
        return load_candidate_pools(scored_path)

    scorer = RewardModelScorer(
        reward_cfg["model_name"],
        cache_dir=config["huggingface"].get("cache_dir"),
        token=os.environ.get("HF_TOKEN") or None,
        trust_remote_code=reward_cfg.get("trust_remote_code", True),
    )

    prompts: list[str] = []
    responses: list[str] = []
    for pool in train_pools:
        for response in pool["responses"]:
            prompts.append(pool["prompt"])
            responses.append(response["text"])

    scores = scorer.score(prompts, responses, batch_size=int(reward_cfg.get("batch_size", 4)))
    score_iter = iter(scores)
    scored_pools: list[dict[str, Any]] = []
    for pool in train_pools:
        scored_pool = dict(pool)
        scored_responses = []
        for response in pool["responses"]:
            scored_response = dict(response)
            scored_response["dataset_score"] = response.get("score")
            scored_response["reward_score"] = float(next(score_iter))
            scored_responses.append(scored_response)
        scored_pool["responses"] = scored_responses
        scored_pools.append(scored_pool)

    write_jsonl(scored_pools, scored_path)
    _write_reward_subset_metadata(config, cache_anchor, scored_path, reward_cfg)
    return scored_pools


def _scores_for_selection(config: dict[str, Any], responses: list[dict[str, Any]]) -> tuple[list[float] | None, str]:
    selection_source = _selection_score_source(config)
    if selection_source == "reward_model":
        values = [response.get("reward_score") for response in responses]
        if any(value is None for value in values):
            return None, "reward_model"
        return [float(value) for value in values], "reward_model"
    if selection_source == "auto":
        reward_values = [response.get("reward_score") for response in responses]
        if all(value is not None for value in reward_values):
            return [float(value) for value in reward_values], "reward_model"
    dataset_values = [response.get("score") for response in responses]
    if any(value is None for value in dataset_values):
        return None, "dataset"
    return [float(value) for value in dataset_values], "dataset"


def _response_embedding_diversity(response_texts: list[str], embedder: SentenceEmbedder) -> float:
    if len(response_texts) < 2:
        return 0.0
    embeddings = embedder.encode(response_texts)
    similarity = cosine_similarity(embeddings)
    triu = similarity[np.triu_indices_from(similarity, k=1)]
    return float(max(0.0, 1.0 - np.mean(triu)))


def _response_length_variance(response_texts: list[str]) -> float:
    if len(response_texts) < 2:
        return 0.0
    lengths = np.asarray([len(text.split()) for text in response_texts], dtype=np.float32)
    return float(lengths.var(ddof=1))


def _load_external_adaptive_scores(adaptive_cfg: dict[str, Any]) -> dict[str, float]:
    raw_path = adaptive_cfg.get("score_path") or adaptive_cfg.get("external_score_path")
    if not raw_path:
        raise ValueError("adaptive_k.score=external requires score_path.")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"External adaptive score file not found: {path}")

    prompt_column = str(adaptive_cfg.get("prompt_id_column", "prompt_id"))
    score_column = str(adaptive_cfg.get("score_column", "score"))
    scores: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"External adaptive score file is empty: {path}")
        missing = {prompt_column, score_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"External adaptive score file {path} missing columns: {sorted(missing)}")
        for row in reader:
            prompt_id = str(row[prompt_column])
            raw_value = row.get(score_column)
            if raw_value in (None, ""):
                continue
            scores[prompt_id] = float(raw_value)
    if not scores:
        raise ValueError(f"External adaptive score file has no usable rows: {path}")
    return scores


def _adaptive_scores_for_pools(
    *,
    train_pools: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
    embedder: SentenceEmbedder | None,
) -> dict[str, float]:
    adaptive_cfg = selection_cfg.get("adaptive_k") or {}
    score_name = str(adaptive_cfg.get("score", "response_embedding_diversity")).lower()
    score_power = float(adaptive_cfg.get("score_power", 1.0))
    score_floor = float(adaptive_cfg.get("score_floor", 1e-8))
    external_scores = (
        _load_external_adaptive_scores(adaptive_cfg)
        if score_name in {"external", "external_csv", "prompt_score", "prompt_scores"}
        else None
    )
    missing_external = str(adaptive_cfg.get("missing_score", "error")).lower()

    scores: dict[str, float] = {}
    for pool in train_pools:
        prompt_id = str(pool["prompt_id"])
        response_texts = [response["text"] for response in pool["responses"]]
        if external_scores is not None:
            if prompt_id in external_scores:
                raw_score = external_scores[prompt_id]
            elif missing_external in {"floor", "min"}:
                raw_score = score_floor
            else:
                raise KeyError(f"Missing external adaptive score for prompt_id={prompt_id}")
        elif score_name in {"response_embedding_diversity", "embedding_diversity", "semantic_diversity"}:
            if embedder is None:
                raise ValueError("Adaptive response_embedding_diversity scoring requires an embedder.")
            raw_score = _response_embedding_diversity(response_texts, embedder)
        elif score_name in {"response_length_variance", "length_variance"}:
            raw_score = _response_length_variance(response_texts)
        elif score_name in {"uniform", "constant"}:
            raw_score = 1.0
        else:
            raise ValueError(f"Unsupported adaptive_k.score: {score_name}")
        scores[prompt_id] = float(max(score_floor, raw_score) ** score_power)
    return scores


def _adaptive_k_allocation(
    *,
    train_pools: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
    embedder: SentenceEmbedder | None,
) -> dict[str, int | float]:
    adaptive_cfg = selection_cfg.get("adaptive_k") or {}
    min_k = int(adaptive_cfg.get("min_k", 1))
    max_k = int(adaptive_cfg.get("max_k", selection_cfg.get("k", 1)))
    if min_k < 1 or max_k < min_k:
        raise ValueError("adaptive_k requires 1 <= min_k <= max_k.")

    prompt_ids = [str(pool["prompt_id"]) for pool in train_pools]
    n_prompts = len(prompt_ids)
    if n_prompts == 0:
        return {}

    if adaptive_cfg.get("target_total") is not None:
        target_total = int(adaptive_cfg["target_total"])
    else:
        target_mean = float(adaptive_cfg.get("target_mean_k", selection_cfg.get("k", 1)))
        target_total = int(round(target_mean * n_prompts))
    target_total = max(n_prompts * min_k, min(n_prompts * max_k, target_total))

    scores = _adaptive_scores_for_pools(
        train_pools=train_pools,
        selection_cfg=selection_cfg,
        embedder=embedder,
    )
    allocation = {prompt_id: min_k for prompt_id in prompt_ids}
    remaining = target_total - n_prompts * min_k
    for _ in range(remaining):
        best_prompt = None
        best_gain = -np.inf
        for prompt_id in prompt_ids:
            current_k = allocation[prompt_id]
            if current_k >= max_k:
                continue
            gain = scores[prompt_id] * (1.0 / current_k - 1.0 / (current_k + 1.0))
            if gain > best_gain:
                best_gain = gain
                best_prompt = prompt_id
        if best_prompt is None:
            break
        allocation[best_prompt] += 1

    ordered_by_score = {
        prompt_id: rank
        for rank, prompt_id in enumerate(
            sorted(prompt_ids, key=lambda item: scores[item], reverse=True),
            start=1,
        )
    }
    result: dict[str, int | float] = {prompt_id: int(allocation[prompt_id]) for prompt_id in prompt_ids}
    for prompt_id in prompt_ids:
        result[f"{prompt_id}::score"] = float(scores[prompt_id])
        result[f"{prompt_id}::rank"] = int(ordered_by_score[prompt_id])
    result["__target_total__"] = int(sum(allocation.values()))

    counts = {k_value: list(allocation.values()).count(k_value) for k_value in range(min_k, max_k + 1)}
    print(
        "Adaptive k allocation: "
        f"score={adaptive_cfg.get('score', 'response_embedding_diversity')}, "
        f"target_total={sum(allocation.values())}, "
        f"mean_k={sum(allocation.values()) / n_prompts:.3f}, "
        f"counts={counts}",
        flush=True,
    )
    return result


def _select_train_pairs(
    *,
    config: dict,
    train_pools: list[dict],
    run_dir: Path,
) -> tuple[Path, Path]:
    selection_cfg = config["selection"]
    strategy = selection_cfg["strategy"]
    k = int(selection_cfg["k"])
    seed = int(config["run"].get("seed", 7))
    rng = np.random.default_rng(seed)

    embedder = None
    adaptive_cfg = selection_cfg.get("adaptive_k") or {}
    adaptive_enabled = bool(adaptive_cfg.get("enabled", adaptive_cfg))
    adaptive_score = str(adaptive_cfg.get("score", "")).lower()
    adaptive_uses_embeddings = adaptive_enabled and adaptive_score in {
        "response_embedding_diversity",
        "embedding_diversity",
        "semantic_diversity",
    }
    if strategy.lower() in {"grades", "dbkon", "dkon"} or adaptive_uses_embeddings:
        embedder = SentenceEmbedder(selection_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))

    adaptive_k_by_prompt = (
        _adaptive_k_allocation(
            train_pools=train_pools,
            selection_cfg=selection_cfg,
            embedder=embedder,
        )
        if adaptive_enabled
        else {}
    )

    selected_indices: dict[str, list[int]] = {}
    selection_manifest = []
    for pool in train_pools:
        candidate_responses = pool["responses"]
        response_texts = [response["text"] for response in candidate_responses]
        scores, score_source_used = _scores_for_selection(config, candidate_responses)
        prompt_id = str(pool["prompt_id"])
        prompt_k = int(adaptive_k_by_prompt.get(prompt_id, k))
        local_indices = select_response_indices(
            strategy=strategy,
            response_texts=response_texts,
            k=prompt_k,
            rng=rng,
            scores=scores,
            embedder=embedder,
            alpha=float(selection_cfg.get("quality_weight_alpha", 1.0)),
        )
        indices = local_indices
        selected_indices[prompt_id] = indices
        manifest_row = {
            "prompt_id": pool["prompt_id"],
            "strategy": strategy,
            "n": len(pool["responses"]),
            "k": prompt_k,
            "score_source": score_source_used,
            "candidate_indices": list(range(len(pool["responses"]))),
            "selected_indices": indices,
        }
        if adaptive_enabled:
            manifest_row["adaptive_k"] = prompt_k
            manifest_row["adaptive_score"] = adaptive_k_by_prompt.get(f"{prompt_id}::score")
            manifest_row["adaptive_rank"] = adaptive_k_by_prompt.get(f"{prompt_id}::rank")
            manifest_row["adaptive_target_total"] = adaptive_k_by_prompt.get("__target_total__")
        selection_manifest.append(manifest_row)

    selected_pairs = flatten_selected_pairs(train_pools, selected_indices)
    selected_dir = run_dir / "selected"
    train_path = selected_dir / "train.jsonl"
    manifest_path = selected_dir / "selection_manifest.jsonl"
    write_jsonl(selected_pairs, train_path)
    write_jsonl(selection_manifest, manifest_path)
    return train_path, manifest_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.runtime_config)
    apply_runtime_environment(config)
    ensure_runtime_directories(config)

    run_cfg = config["run"]
    run_name = run_cfg["name"]
    seed = int(run_cfg.get("seed", 7))
    seed_everything(seed)

    run_dir = run_root(config, run_name)
    dump_yaml(run_dir / "merged_config.yaml", config)

    prepared_path = _prepare_candidate_pools(config, args.force_prepare)
    pools = load_candidate_pools(prepared_path)
    grouped = pools_by_split(pools)
    train_pools = grouped["train"]
    validation_pools = grouped["validation"]
    test_pools = grouped["test"]
    train_pools = _materialize_train_candidate_pools(
        config=config,
        prepared_path=prepared_path,
        train_pools=train_pools,
    )
    train_pools = _materialize_selection_scores(
        config=config,
        prepared_path=prepared_path,
        train_pools=train_pools,
    )

    max_reference_responses = config["data"].get("max_reference_responses_per_prompt")
    validation_reference_pools = sample_pool_responses(
        validation_pools,
        max_responses_per_prompt=max_reference_responses,
        seed=seed + 101,
    )
    test_reference_pools = sample_pool_responses(
        test_pools,
        max_responses_per_prompt=max_reference_responses,
        seed=seed + 202,
    )

    train_path, manifest_path = _select_train_pairs(
        config=config,
        train_pools=train_pools,
        run_dir=run_dir,
    )

    validation_pairs = flatten_all_pairs(validation_reference_pools)
    test_pairs = flatten_all_pairs(test_reference_pools)
    validation_path = run_dir / "selected" / "validation_all.jsonl"
    test_path = run_dir / "selected" / "test_all.jsonl"
    write_jsonl(validation_pairs, validation_path)
    write_jsonl(test_pairs, test_path)

    spec, model_cfg = resolve_model_config(config)
    model_cfg.setdefault("cache_dir", config["huggingface"]["cache_dir"])

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    train_metrics = {}
    if not args.skip_train:
        checkpoint_dir, train_metrics = train_model(
            train_path=train_path,
            eval_path=validation_path,
            output_dir=checkpoint_dir,
            spec=spec,
            model_cfg=model_cfg,
            training_cfg=config["training"],
            seed=seed,
        )

    eval_metrics = {}
    if not args.skip_eval:
        print(
            "Starting evaluation with "
            f"{len(test_pairs)} reference pairs and {len(test_reference_pools)} test prompts."
        )
        eval_metrics = evaluate_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_cfg=model_cfg,
            spec=spec,
            test_pairs=test_pairs,
            test_pools=test_reference_pools,
            references_by_prompt=reference_texts_by_prompt(test_reference_pools),
            generation_cfg=config["generation"],
            evaluation_cfg=config["evaluation"],
            output_dir=run_dir / "evaluation",
        )
        print(f"Finished evaluation for {run_name}.")

    summary = {
        "created_at": utc_now_iso(),
        "run_name": run_name,
        "config_path": args.config,
        "runtime_config_path": args.runtime_config,
        "prepared_pool_path": str(prepared_path),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "selection_manifest_path": str(manifest_path),
        "checkpoint_dir": str(checkpoint_dir),
        "train_metrics": train_metrics,
        "evaluation_metrics": eval_metrics,
        "counts": {
            "train_prompts": len(train_pools),
            "validation_prompts": len(validation_pools),
            "test_prompts": len(test_pools),
            "train_pairs": _count_lines(train_path),
            "validation_pairs": len(validation_pairs),
            "test_pairs": len(test_pairs),
        },
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_project_run_artifacts(summary)
    rebuild_summary_index()
    print(json.dumps(summary, indent=2))
    print(summary_path)


if __name__ == "__main__":
    main()
