from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

from embeddings import SentenceEmbedder
from formatting import build_generation_prompt
from jsonl_io import write_jsonl
from models import ModelSpec, load_pretrained_model, load_text_preprocessor
from reward import RewardModelScorer
from train import SupervisedCollator, SupervisedFineTuningDataset


def _load_tokenizer(model_cfg: dict[str, Any]) -> Any:
    tokenizer = load_text_preprocessor(model_cfg)
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_inference(
    *,
    checkpoint_dir: str | Path,
    model_cfg: dict[str, Any],
) -> tuple[Any, Any]:
    from peft import PeftModel

    checkpoint_path = Path(checkpoint_dir)
    tokenizer = _load_tokenizer(model_cfg)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    auto_model_class = str(model_cfg.get("auto_model_class", "causal_lm"))
    model = load_pretrained_model(
        model_cfg=model_cfg,
        auto_model_class=auto_model_class,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if (checkpoint_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, checkpoint_path)
    elif (checkpoint_path / "config.json").exists() and checkpoint_path != Path(model_cfg["hf_id"]):
        checkpoint_model_cfg = dict(model_cfg)
        checkpoint_model_cfg["hf_id"] = str(checkpoint_path)
        model = load_pretrained_model(
            model_cfg=checkpoint_model_cfg,
            auto_model_class=auto_model_class,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    model.eval()
    return model, tokenizer


def compute_reference_nll(
    *,
    model: Any,
    tokenizer: Any,
    spec: ModelSpec,
    records: list[dict[str, Any]],
    max_length: int,
    batch_size: int = 2,
) -> dict[str, float]:
    dataset = SupervisedFineTuningDataset(records, tokenizer, spec, max_length=max_length)
    if len(dataset) == 0:
        return {"reference_loss": 0.0, "reference_perplexity": 1.0}

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=SupervisedCollator(tokenizer))
    losses: list[float] = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().float().cpu()))

    loss = float(np.mean(losses))
    perplexity = float(math.exp(min(loss, 20.0)))
    return {"reference_loss": loss, "reference_perplexity": perplexity}


def generate_samples(
    *,
    model: Any,
    tokenizer: Any,
    spec: ModelSpec,
    pools: list[dict[str, Any]],
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    generated_records: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    for pool in pools:
        prompt = pool["prompt"]
        prompt_text = build_generation_prompt(tokenizer, prompt, spec)
        encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_length = encoded["input_ids"].shape[-1]
        for generation_index, output_ids in enumerate(outputs):
            response = tokenizer.decode(
                output_ids[input_length:],
                skip_special_tokens=True,
            ).strip()
            generated_records.append(
                {
                    "prompt_id": pool["prompt_id"],
                    "prompt": prompt,
                    "generation_index": generation_index,
                    "response": response,
                }
            )
    return generated_records


def _group_generations_by_prompt(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[str(record["prompt_id"])].append(record["response"])
    return grouped


def _group_records_by_prompt(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["prompt_id"])].append(record)
    for prompt_id in grouped:
        grouped[prompt_id] = sorted(grouped[prompt_id], key=lambda item: int(item.get("generation_index", 0)))
    return grouped


def _prefix_records(records: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = []
    for prompt_records in _group_records_by_prompt(records).values():
        prefix.extend(prompt_records[:k])
    return prefix


def _resolve_multi_k_values(generation_cfg: dict[str, Any]) -> list[int]:
    configured = generation_cfg.get("multi_k_values")
    if configured:
        values = sorted({int(value) for value in configured if int(value) > 0})
        if values:
            return values
    return [int(generation_cfg.get("num_return_sequences_multi", 4))]


def compute_length_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    char_lengths = [len(record["response"]) for record in records]
    word_lengths = [len(record["response"].split()) for record in records]
    return {
        "response_chars_mean": float(np.mean(char_lengths)) if char_lengths else 0.0,
        "response_words_mean": float(np.mean(word_lengths)) if word_lengths else 0.0,
    }


def compute_reward_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    scores = [record["reward_score"] for record in records if record.get("reward_score") is not None]
    if not scores:
        return {}

    grouped = _group_records_by_prompt(records)
    prompt_best_scores: list[float] = []
    prompt_mean_scores: list[float] = []
    for prompt_records in grouped.values():
        prompt_scores = [record["reward_score"] for record in prompt_records if record.get("reward_score") is not None]
        if not prompt_scores:
            continue
        prompt_best_scores.append(float(np.max(prompt_scores)))
        prompt_mean_scores.append(float(np.mean(prompt_scores)))

    return {
        "reward_mean": float(np.mean(scores)),
        "reward_prompt_mean": float(np.mean(prompt_mean_scores)) if prompt_mean_scores else 0.0,
        "reward_best_mean": float(np.mean(prompt_best_scores)) if prompt_best_scores else 0.0,
    }


def compute_diversity_metrics(
    grouped_generations: dict[str, list[str]],
    *,
    embedder: SentenceEmbedder,
) -> dict[str, float]:
    prompt_diversities: list[float] = []
    for responses in grouped_generations.values():
        if len(responses) < 2:
            continue
        embeddings = embedder.encode(responses)
        similarity = cosine_similarity(embeddings)
        triu = similarity[np.triu_indices_from(similarity, k=1)]
        prompt_diversities.append(float(1.0 - np.mean(triu)))
    return {
        "semantic_diversity_mean": float(np.mean(prompt_diversities)) if prompt_diversities else 0.0
    }


def compute_reference_coverage(
    grouped_generations: dict[str, list[str]],
    references_by_prompt: dict[str, list[str]],
    *,
    embedder: SentenceEmbedder,
) -> dict[str, float]:
    coverage_scores: list[float] = []
    alignment_scores: list[float] = []
    for prompt_id, generations in grouped_generations.items():
        references = references_by_prompt.get(prompt_id, [])
        if not generations or not references:
            continue
        gen_embeddings = embedder.encode(generations)
        ref_embeddings = embedder.encode(references)
        similarity = cosine_similarity(ref_embeddings, gen_embeddings)
        coverage_scores.append(float(np.mean(np.max(similarity, axis=1))))
        alignment_scores.append(float(np.mean(np.max(similarity, axis=0))))
    return {
        "reference_coverage_mean": float(np.mean(coverage_scores)) if coverage_scores else 0.0,
        "reference_alignment_mean": float(np.mean(alignment_scores)) if alignment_scores else 0.0,
    }


def maybe_score_rewards(
    records: list[dict[str, Any]],
    reward_cfg: dict[str, Any] | None,
    *,
    cache_dir: str | None,
    scorer: RewardModelScorer | None = None,
) -> list[dict[str, Any]]:
    if not reward_cfg or not reward_cfg.get("model_name"):
        return records

    active_scorer = scorer or RewardModelScorer(
        reward_cfg["model_name"],
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN") or None,
        trust_remote_code=reward_cfg.get("trust_remote_code", True),
    )
    prompts = [record["prompt"] for record in records]
    responses = [record["response"] for record in records]
    scores = active_scorer.score(prompts, responses, batch_size=int(reward_cfg.get("batch_size", 4)))
    scored_records: list[dict[str, Any]] = []
    for record, score in zip(records, scores):
        enriched = dict(record)
        enriched["reward_score"] = float(score)
        scored_records.append(enriched)
    return scored_records


def evaluate_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model_cfg: dict[str, Any],
    spec: ModelSpec,
    test_pairs: list[dict[str, Any]],
    test_pools: list[dict[str, Any]],
    references_by_prompt: dict[str, list[str]],
    generation_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
    output_dir: str | Path,
    write_generation_artifacts: bool = True,
    reward_scorer: RewardModelScorer | None = None,
    embedder: SentenceEmbedder | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    multi_k_values = _resolve_multi_k_values(generation_cfg)
    max_multi_generations = max(multi_k_values)
    generation_seed = evaluation_cfg.get("seed")

    print(f"Loading inference model from {checkpoint_dir}...")
    model, tokenizer = load_model_for_inference(
        checkpoint_dir=checkpoint_dir,
        model_cfg=model_cfg,
    )

    print(f"Computing reference NLL on {len(test_pairs)} held-out prompt-response pairs...")
    metrics = compute_reference_nll(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        records=test_pairs,
        max_length=int(generation_cfg.get("max_length_for_eval", 1024)),
        batch_size=int(evaluation_cfg.get("eval_batch_size", 2)),
    )

    print(f"Generating single-sample outputs for {len(test_pools)} prompts...")
    single_records = generate_samples(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        pools=test_pools,
        num_return_sequences=int(generation_cfg.get("num_return_sequences_single", 1)),
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 256)),
        temperature=float(generation_cfg.get("temperature", 0.8)),
        top_p=float(generation_cfg.get("top_p", 0.95)),
        seed=int(generation_seed) + 11 if generation_seed is not None else None,
    )
    print(f"Generating up to {max_multi_generations} samples per prompt for {len(test_pools)} prompts...")
    multi_records = generate_samples(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        pools=test_pools,
        num_return_sequences=max_multi_generations,
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 256)),
        temperature=float(generation_cfg.get("temperature", 0.8)),
        top_p=float(generation_cfg.get("top_p", 0.95)),
        seed=int(generation_seed) + 97 if generation_seed is not None else None,
    )

    active_reward_scorer = reward_scorer
    reward_cfg = evaluation_cfg.get("reward_model")
    if active_reward_scorer is None and reward_cfg and reward_cfg.get("model_name"):
        print(f"Loading reward model {reward_cfg['model_name']}...")
        active_reward_scorer = RewardModelScorer(
            reward_cfg["model_name"],
            cache_dir=model_cfg.get("cache_dir"),
            token=os.environ.get("HF_TOKEN") or None,
            trust_remote_code=reward_cfg.get("trust_remote_code", True),
        )

    single_records = maybe_score_rewards(
        single_records,
        reward_cfg,
        cache_dir=model_cfg.get("cache_dir"),
        scorer=active_reward_scorer,
    )
    multi_records = maybe_score_rewards(
        multi_records,
        reward_cfg,
        cache_dir=model_cfg.get("cache_dir"),
        scorer=active_reward_scorer,
    )

    active_embedder = embedder
    if active_embedder is None:
        print("Loading embedding model for diversity and coverage metrics...")
        active_embedder = SentenceEmbedder(
            evaluation_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )

    metrics.update({f"single_{k}": v for k, v in compute_length_metrics(single_records).items()})
    metrics.update({f"single_{k}": v for k, v in compute_reward_metrics(single_records).items()})

    for k_value in multi_k_values:
        k_records = _prefix_records(multi_records, k_value)
        grouped_multi = _group_generations_by_prompt(k_records)
        k_metrics: dict[str, float] = {}
        k_metrics.update(compute_length_metrics(k_records))
        k_metrics.update(compute_diversity_metrics(grouped_multi, embedder=active_embedder))
        k_metrics.update(
            compute_reference_coverage(
                grouped_multi,
                references_by_prompt,
                embedder=active_embedder,
            )
        )
        k_metrics.update(compute_reward_metrics(k_records))
        metrics.update({f"multi_k{k_value}_{name}": value for name, value in k_metrics.items()})
        if k_value == max_multi_generations:
            metrics.update({f"multi_{name}": value for name, value in k_metrics.items()})
        if write_generation_artifacts:
            write_jsonl(k_records, output_path / f"multi_k{k_value}_generations.jsonl")

    if write_generation_artifacts:
        write_jsonl(single_records, output_path / "single_generations.jsonl")
        write_jsonl(multi_records, output_path / "multi_generations.jsonl")
        print(f"Wrote evaluation artifacts to {output_path}.")
    else:
        print(f"Computed evaluation metrics without writing generation artifacts to {output_path}.")
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics
