from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from jsonl_io import iter_jsonl, write_jsonl


def _format_prompt(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}".strip()
    return instruction


def _coerce_response_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("text", "answer", "response", "output", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(item).strip()


def _coerce_score(item: Any) -> float | None:
    if item is None:
        return None
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for key in ("score", "gold_score", "value"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _mean_numeric_values(item: Any) -> float | None:
    if isinstance(item, dict):
        values = [float(value) for value in item.values() if isinstance(value, (int, float))]
        if values:
            return float(np.mean(values))
    if isinstance(item, list):
        values = [float(value) for value in item if isinstance(value, (int, float))]
        if values:
            return float(np.mean(values))
    return None


def _load_hf_dataset(
    *,
    hf_dataset_id: str,
    hf_split: str,
    hf_subset: str | None = None,
) -> Any:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": hf_split}
    if hf_subset:
        kwargs["name"] = hf_subset
    return load_dataset(hf_dataset_id, **kwargs)


def _build_prepared_records(
    *,
    prompt_pools: list[dict[str, Any]],
    output_path: str | Path,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    max_train_prompts: int | None,
    max_validation_prompts: int | None,
    max_test_prompts: int | None,
) -> Path:
    rng = np.random.default_rng(seed)
    rng.shuffle(prompt_pools)

    total = len(prompt_pools)
    train_count = int(total * train_fraction)
    validation_count = int(total * validation_fraction)
    test_count = total - train_count - validation_count

    train_pools = prompt_pools[:train_count]
    validation_pools = prompt_pools[train_count : train_count + validation_count]
    test_pools = prompt_pools[train_count + validation_count : train_count + validation_count + test_count]

    if max_train_prompts is not None:
        train_pools = train_pools[:max_train_prompts]
    if max_validation_prompts is not None:
        validation_pools = validation_pools[:max_validation_prompts]
    if max_test_prompts is not None:
        test_pools = test_pools[:max_test_prompts]

    prepared_records = []
    for split_name, pools in (
        ("train", train_pools),
        ("validation", validation_pools),
        ("test", test_pools),
    ):
        for pool in pools:
            record = dict(pool)
            record["split"] = split_name
            prepared_records.append(record)

    return write_jsonl(prepared_records, output_path)


def _cap_records_stratified(
    records: list[dict[str, Any]],
    *,
    cap: int | None,
) -> list[dict[str, Any]]:
    if cap is None or len(records) <= cap:
        return records

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_domain[str(record.get("domain", "unknown"))].append(record)

    domains = sorted(by_domain)
    base = cap // max(len(domains), 1)
    remainder = cap % max(len(domains), 1)
    capped: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(domains):
        take = base + (1 if domain_index < remainder else 0)
        capped.extend(by_domain[domain][:take])
    return capped


def _split_mosaic_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    max_train_clusters: int | None,
    max_validation_clusters: int | None,
    max_test_clusters: int | None,
) -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_domain[str(record["domain"])].append(record)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for domain in sorted(by_domain):
        domain_records = list(by_domain[domain])
        rng.shuffle(domain_records)
        total = len(domain_records)
        train_count = int(total * train_fraction)
        validation_count = int(total * validation_fraction)

        splits["train"].extend(domain_records[:train_count])
        splits["validation"].extend(domain_records[train_count : train_count + validation_count])
        splits["test"].extend(domain_records[train_count + validation_count :])

    return {
        "train": _cap_records_stratified(splits["train"], cap=max_train_clusters),
        "validation": _cap_records_stratified(splits["validation"], cap=max_validation_clusters),
        "test": _cap_records_stratified(splits["test"], cap=max_test_clusters),
    }


def _variant_indices(count: int | None) -> list[int]:
    if count is None:
        count = 4
    if count < 1 or count > 4:
        raise ValueError("MOSAIC prompt variant counts must be in [1, 4].")
    return list(range(count))


def _mosaic_response_indices(
    *,
    mode: str,
    variant_index: int,
    response_count: int,
    response_index: int,
) -> list[int]:
    normalized = mode.lower()
    if normalized == "all":
        return list(range(response_count))
    if normalized in {"cyclic_one", "diagonal_one", "balanced_one"}:
        return [variant_index % response_count]
    if normalized in {"same_one", "first_one"}:
        return [response_index % response_count]
    raise ValueError(f"Unsupported MOSAIC response_mode: {mode}")


def _mosaic_pool_records(
    *,
    raw_records: list[dict[str, Any]],
    split_name: str,
    prompt_variant_count: int,
    response_mode: str,
    response_index: int,
) -> list[dict[str, Any]]:
    pools: list[dict[str, Any]] = []
    for row in raw_records:
        prompt_variants = row.get("prompt_variants") or []
        response_grid = row.get("responses") or []
        metadata = row.get("metadata") or {}
        for variant_index in _variant_indices(prompt_variant_count):
            if variant_index >= len(prompt_variants) or variant_index >= len(response_grid):
                continue
            variant_responses = response_grid[variant_index]
            chosen_indices = _mosaic_response_indices(
                mode=response_mode,
                variant_index=variant_index,
                response_count=len(variant_responses),
                response_index=response_index,
            )
            response_records = []
            for local_rank, response_idx in enumerate(chosen_indices):
                text = _coerce_response_text(variant_responses[response_idx])
                if not text:
                    continue
                response_records.append(
                    {
                        "response_id": f"{row['id']}_v{variant_index}_r{response_idx}",
                        "text": text,
                        "score": None,
                        "rank": local_rank,
                        "mosaic_response_index": response_idx,
                    }
                )
            if not response_records:
                continue

            pools.append(
                {
                    "dataset": "mosaic-1k",
                    "prompt_id": f"{row['id']}_v{variant_index}",
                    "cluster_id": row["id"],
                    "source_index": row["id"],
                    "prompt": str(prompt_variants[variant_index]).strip(),
                    "responses": response_records,
                    "split": split_name,
                    "domain": row.get("domain"),
                    "subdomain": row.get("subdomain"),
                    "base_theme": row.get("base_theme"),
                    "prompt_variant_index": variant_index,
                    "response_mode": response_mode,
                    "response_type": metadata.get("response_type"),
                    "expected_diversity": metadata.get("expected_diversity"),
                    "difficulty": metadata.get("difficulty"),
                }
            )
    return pools


def prepare_mosaic_dataset(
    *,
    raw_path: str | Path,
    output_path: str | Path,
    seed: int,
    train_fraction: float = 0.86,
    validation_fraction: float = 0.04,
    prompt_variants_per_cluster: int = 1,
    eval_prompt_variants_per_cluster: int = 4,
    response_mode: str = "all",
    eval_response_mode: str = "all",
    response_index: int = 0,
    max_train_clusters: int | None = None,
    max_validation_clusters: int | None = None,
    max_test_clusters: int | None = None,
) -> Path:
    """Prepare MOSAIC-1K into the prompt-pool format used by the MRT harness.

    Training pools can intentionally vary the number of near-duplicate prompt
    variants and the number of responses retained per variant. Validation and
    test pools default to the full 4x4 lattice, keeping downstream evaluation
    fixed across MOSAIC ablations.
    """
    records = list(iter_jsonl(raw_path))
    splits = _split_mosaic_records(
        records,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        max_train_clusters=max_train_clusters,
        max_validation_clusters=max_validation_clusters,
        max_test_clusters=max_test_clusters,
    )

    prepared_records: list[dict[str, Any]] = []
    prepared_records.extend(
        _mosaic_pool_records(
            raw_records=splits["train"],
            split_name="train",
            prompt_variant_count=prompt_variants_per_cluster,
            response_mode=response_mode,
            response_index=response_index,
        )
    )
    for split_name in ("validation", "test"):
        prepared_records.extend(
            _mosaic_pool_records(
                raw_records=splits[split_name],
                split_name=split_name,
                prompt_variant_count=eval_prompt_variants_per_cluster,
                response_mode=eval_response_mode,
                response_index=response_index,
            )
        )

    return write_jsonl(prepared_records, output_path)


def prepare_gold_labelled_gens_dataset(
    *,
    hf_dataset_id: str,
    hf_split: str,
    hf_subset: str | None,
    output_path: str | Path,
    seed: int,
    min_responses: int = 2,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    max_train_prompts: int | None = None,
    max_validation_prompts: int | None = None,
    max_test_prompts: int | None = None,
) -> Path:
    dataset = _load_hf_dataset(
        hf_dataset_id=hf_dataset_id,
        hf_split=hf_split,
        hf_subset=hf_subset,
    )
    prompt_pools: list[dict[str, Any]] = []

    for row_index, row in enumerate(dataset):
        prompt = _format_prompt(row.get("instruction", ""), row.get("input", ""))
        answers = row.get("answers") or row.get("responses") or []
        scores = row.get("gold_scores") or row.get("scores") or []

        response_records = []
        for answer_index, answer in enumerate(answers):
            text = _coerce_response_text(answer)
            if not text:
                continue
            score = _coerce_score(scores[answer_index] if answer_index < len(scores) else None)
            response_records.append(
                {
                    "response_id": f"{row_index}_{answer_index}",
                    "text": text,
                    "score": score,
                    "rank": answer_index,
                }
            )

        if len(response_records) < min_responses:
            continue

        prompt_pools.append(
            {
                "dataset": hf_dataset_id,
                "prompt_id": row.get("id", row_index),
                "source_index": row_index,
                "prompt": prompt,
                "responses": response_records,
            }
        )

    return _build_prepared_records(
        prompt_pools=prompt_pools,
        output_path=output_path,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        max_train_prompts=max_train_prompts,
        max_validation_prompts=max_validation_prompts,
        max_test_prompts=max_test_prompts,
    )


def prepare_ultrafeedback_dataset(
    *,
    hf_dataset_id: str,
    hf_split: str,
    hf_subset: str | None,
    output_path: str | Path,
    seed: int,
    min_responses: int = 2,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    max_train_prompts: int | None = None,
    max_validation_prompts: int | None = None,
    max_test_prompts: int | None = None,
) -> Path:
    dataset = _load_hf_dataset(
        hf_dataset_id=hf_dataset_id,
        hf_split=hf_split,
        hf_subset=hf_subset,
    )
    prompt_pools: list[dict[str, Any]] = []

    for row_index, row in enumerate(dataset):
        prompt = _format_prompt(row.get("instruction", ""), row.get("input", ""))
        completions = row.get("completions") or []
        models = row.get("models") or []

        response_records = []
        for answer_index, completion in enumerate(completions):
            if not isinstance(completion, dict):
                completion = {"response": completion}
            text = _coerce_response_text(completion.get("response", completion))
            if not text:
                continue
            score = _coerce_score(completion.get("overall_score"))
            if score is None:
                score = _mean_numeric_values(completion.get("fine-grained_score"))
            response_records.append(
                {
                    "response_id": f"{row_index}_{answer_index}",
                    "text": text,
                    "score": score,
                    "rank": answer_index,
                    "model": completion.get("model") or (models[answer_index] if answer_index < len(models) else None),
                }
            )

        if len(response_records) < min_responses:
            continue

        prompt_pools.append(
            {
                "dataset": hf_dataset_id,
                "prompt_id": row_index,
                "source_index": row_index,
                "prompt": prompt,
                "responses": response_records,
            }
        )

    return _build_prepared_records(
        prompt_pools=prompt_pools,
        output_path=output_path,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        max_train_prompts=max_train_prompts,
        max_validation_prompts=max_validation_prompts,
        max_test_prompts=max_test_prompts,
    )


def prepare_nectar_dataset(
    *,
    hf_dataset_id: str,
    hf_split: str,
    hf_subset: str | None,
    output_path: str | Path,
    seed: int,
    min_responses: int = 2,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    max_train_prompts: int | None = None,
    max_validation_prompts: int | None = None,
    max_test_prompts: int | None = None,
) -> Path:
    dataset = _load_hf_dataset(
        hf_dataset_id=hf_dataset_id,
        hf_split=hf_split,
        hf_subset=hf_subset,
    )
    prompt_pools: list[dict[str, Any]] = []

    for row_index, row in enumerate(dataset):
        prompt = (row.get("prompt") or "").strip()
        answers = row.get("answers") or []

        response_records = []
        ranks: list[int] = []
        for answer_index, answer in enumerate(answers):
            if not isinstance(answer, dict):
                answer = {"answer": answer}
            text = _coerce_response_text(answer.get("answer", answer))
            if not text:
                continue
            raw_rank = answer.get("rank")
            rank = int(raw_rank) if isinstance(raw_rank, (int, float)) else answer_index
            ranks.append(rank)
            response_records.append(
                {
                    "response_id": f"{row_index}_{answer_index}",
                    "text": text,
                    "score": None,
                    "rank": rank,
                    "model": answer.get("model"),
                }
            )

        if len(response_records) < min_responses:
            continue

        max_rank = max(ranks) if ranks else len(response_records) - 1
        for response in response_records:
            response["score"] = float(max_rank - int(response["rank"]))

        prompt_pools.append(
            {
                "dataset": hf_dataset_id,
                "prompt_id": row_index,
                "source_index": row_index,
                "prompt": prompt,
                "responses": response_records,
            }
        )

    return _build_prepared_records(
        prompt_pools=prompt_pools,
        output_path=output_path,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        max_train_prompts=max_train_prompts,
        max_validation_prompts=max_validation_prompts,
        max_test_prompts=max_test_prompts,
    )


def prepare_helpsteer2_dataset(
    *,
    hf_dataset_id: str,
    hf_split: str,
    hf_subset: str | None,
    output_path: str | Path,
    seed: int,
    min_responses: int = 2,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    max_train_prompts: int | None = None,
    max_validation_prompts: int | None = None,
    max_test_prompts: int | None = None,
) -> Path:
    dataset = _load_hf_dataset(
        hf_dataset_id=hf_dataset_id,
        hf_split=hf_split,
        hf_subset=hf_subset,
    )
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        prompt = (row.get("prompt") or "").strip()
        if prompt:
            grouped_rows[prompt].append(row)

    prompt_pools: list[dict[str, Any]] = []
    for prompt_index, (prompt, rows) in enumerate(grouped_rows.items()):
        response_records = []
        for response_index, row in enumerate(rows):
            text = _coerce_response_text(row.get("response", ""))
            if not text:
                continue
            attribute_scores = [
                row.get("helpfulness"),
                row.get("correctness"),
                row.get("coherence"),
            ]
            numeric_scores = [float(value) for value in attribute_scores if isinstance(value, (int, float))]
            response_records.append(
                {
                    "response_id": f"{prompt_index}_{response_index}",
                    "text": text,
                    "score": float(np.mean(numeric_scores)) if numeric_scores else None,
                    "rank": response_index,
                    "helpfulness": row.get("helpfulness"),
                    "correctness": row.get("correctness"),
                    "coherence": row.get("coherence"),
                    "complexity": row.get("complexity"),
                    "verbosity": row.get("verbosity"),
                }
            )

        if len(response_records) < min_responses:
            continue

        prompt_pools.append(
            {
                "dataset": hf_dataset_id,
                "prompt_id": prompt_index,
                "source_index": prompt_index,
                "prompt": prompt,
                "responses": response_records,
            }
        )

    return _build_prepared_records(
        prompt_pools=prompt_pools,
        output_path=output_path,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        max_train_prompts=max_train_prompts,
        max_validation_prompts=max_validation_prompts,
        max_test_prompts=max_test_prompts,
    )


def prepare_code_contests_dataset(
    *,
    hf_dataset_id: str,
    hf_split: str,
    hf_subset: str | None,
    output_path: str | Path,
    seed: int,
    min_responses: int = 4,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    max_train_prompts: int | None = None,
    max_validation_prompts: int | None = None,
    max_test_prompts: int | None = None,
) -> Path:
    """Prepare deepmind/code_contests for MRT.

    Filters to Python3 solutions only (language == 3).  Each problem's
    ``description`` becomes the prompt and every Python3 solution becomes
    a response record with ``score=None``.
    """
    dataset = _load_hf_dataset(
        hf_dataset_id=hf_dataset_id,
        hf_split=hf_split,
        hf_subset=hf_subset,
    )
    prompt_pools: list[dict[str, Any]] = []

    _PYTHON3_LANGUAGE = 3

    for row_index, row in enumerate(dataset):
        prompt = (row.get("description") or "").strip()
        if not prompt:
            continue

        solutions = row.get("solutions") or {}
        languages = solutions.get("language") or []
        solution_texts = solutions.get("solution") or []

        response_records = []
        for sol_index, (lang, text) in enumerate(zip(languages, solution_texts)):
            if lang != _PYTHON3_LANGUAGE:
                continue
            text = (text or "").strip()
            if not text:
                continue
            response_records.append(
                {
                    "response_id": f"{row_index}_{sol_index}",
                    "text": text,
                    "score": None,
                    "rank": len(response_records),
                }
            )

        if len(response_records) < min_responses:
            continue

        prompt_pools.append(
            {
                "dataset": hf_dataset_id,
                "prompt_id": row_index,
                "source_index": row_index,
                "prompt": prompt,
                "responses": response_records,
                "difficulty": row.get("difficulty"),
                "cf_rating": row.get("cf_rating"),
            }
        )

    return _build_prepared_records(
        prompt_pools=prompt_pools,
        output_path=output_path,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        max_train_prompts=max_train_prompts,
        max_validation_prompts=max_validation_prompts,
        max_test_prompts=max_test_prompts,
    )


def load_candidate_pools(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def pools_by_split(pools: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for pool in pools:
        grouped.setdefault(pool["split"], []).append(pool)
    return grouped


def sample_pool_responses(
    pools: list[dict[str, Any]],
    *,
    max_responses_per_prompt: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if not max_responses_per_prompt:
        return pools

    rng = np.random.default_rng(seed)
    sampled_pools: list[dict[str, Any]] = []
    for pool in pools:
        responses = pool["responses"]
        if len(responses) <= max_responses_per_prompt:
            sampled_pools.append(pool)
            continue
        chosen = sorted(rng.choice(len(responses), size=max_responses_per_prompt, replace=False).tolist())
        sampled_pool = dict(pool)
        sampled_pool["responses"] = [responses[index] for index in chosen]
        sampled_pools.append(sampled_pool)
    return sampled_pools


def flatten_selected_pairs(
    pools: list[dict[str, Any]],
    selected_indices: dict[str, list[int]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pool in pools:
        chosen = selected_indices[str(pool["prompt_id"])]
        for index in chosen:
            response = pool["responses"][index]
            records.append(
                {
                    "prompt_id": pool["prompt_id"],
                    "prompt": pool["prompt"],
                    "response": response["text"],
                    "score": response.get("score"),
                }
            )
    return records


def flatten_all_pairs(pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pool in pools:
        for response in pool["responses"]:
            records.append(
                {
                    "prompt_id": pool["prompt_id"],
                    "prompt": pool["prompt"],
                    "response": response["text"],
                    "score": response.get("score"),
                }
            )
    return records


def reference_texts_by_prompt(pools: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        str(pool["prompt_id"]): [response["text"] for response in pool["responses"]]
        for pool in pools
    }
