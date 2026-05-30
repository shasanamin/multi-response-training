from __future__ import annotations

from typing import Iterable

import numpy as np


class SentenceEmbedder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        kwargs = {}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)
