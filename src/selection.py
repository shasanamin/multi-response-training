from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def normalize_scores(scores: list[float]) -> np.ndarray:
    raw = np.asarray(scores, dtype=np.float32)
    if raw.size == 0:
        return raw
    lo = float(raw.min())
    hi = float(raw.max())
    if hi - lo < 1e-8:
        return np.ones_like(raw)
    return (raw - lo) / (hi - lo + 1e-8)


def _grades_select(
    embeddings: np.ndarray,
    scores: list[float] | None,
    k: int,
    alpha: float,
) -> list[int]:
    n = embeddings.shape[0]
    if n <= k:
        return list(range(n))

    sim = cosine_similarity(embeddings)
    sim = (sim + 1.0) / 2.0

    base_weights = np.ones(n, dtype=np.float32)
    candidate_weights = np.ones(n, dtype=np.float32)
    if scores is not None:
        normalized = normalize_scores(scores)
        candidate_weights = np.power(np.clip(normalized, 1e-4, 1.0), alpha)
        base_weights = np.maximum(candidate_weights, 1e-4)

    selected: list[int] = []
    current_coverage = np.zeros(n, dtype=np.float32)

    for _ in range(k):
        best_index = -1
        best_gain = -1.0
        for candidate in range(n):
            if candidate in selected:
                continue
            candidate_cover = candidate_weights[candidate] * sim[:, candidate]
            gain = float(np.sum(base_weights * np.maximum(current_coverage, candidate_cover)))
            if gain > best_gain:
                best_gain = gain
                best_index = candidate
        if best_index < 0:
            break
        selected.append(best_index)
        current_coverage = np.maximum(
            current_coverage,
            candidate_weights[best_index] * sim[:, best_index],
        )
    return selected


def select_response_indices(
    *,
    strategy: str,
    response_texts: list[str],
    k: int,
    rng: np.random.Generator,
    scores: list[float] | None = None,
    embedder: Any | None = None,
    alpha: float = 1.0,
) -> list[int]:
    normalized = strategy.lower()
    n = len(response_texts)
    if n == 0:
        return []
    if n <= k:
        return list(range(n))

    if normalized in {"random", "rkon"}:
        return sorted(rng.choice(n, size=k, replace=False).tolist())

    if normalized in {"top_k", "bkon", "best_k", "best_k_of_n"}:
        if scores is None:
            raise ValueError("top_k selection requires scores.")
        order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1]
        return sorted(order[:k].tolist())

    if normalized in {"dkon", "diverse_k", "diverse_k_of_n"}:
        if embedder is None:
            raise ValueError("dkon selection requires an embedder.")
        embeddings = embedder.encode(response_texts)
        return sorted(_grades_select(embeddings, None, k, alpha=0.0))

    if normalized in {"grades", "dbkon", "diverse_best_k", "diverse_best_k_of_n"}:
        if embedder is None:
            raise ValueError("grades selection requires an embedder.")
        embeddings = embedder.encode(response_texts)
        return sorted(_grades_select(embeddings, scores, k, alpha))

    raise ValueError(f"Unknown selection strategy: {strategy}")
