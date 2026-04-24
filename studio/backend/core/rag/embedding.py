# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from threading import RLock
from typing import Protocol


class EmbeddingProviderProtocol(Protocol):
    def embed_texts(self, model_id: str, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddingProvider:
    def __init__(self) -> None:
        self._lock = RLock()
        self._models: dict[str, object] = {}

    def embed_texts(self, model_id: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model(model_id)
        vectors = model.encode(  # type: ignore[call-arg]
            texts,
            normalize_embeddings = True,
            show_progress_bar = False,
        )
        return [list(map(float, row)) for row in vectors]

    def _get_model(self, model_id: str):
        with self._lock:
            existing = self._models.get(model_id)
            if existing is not None:
                return existing
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on runtime
                raise RuntimeError(
                    "Embedding provider `sentence-transformers` is not available."
                ) from exc
            model = SentenceTransformer(model_id)
            self._models[model_id] = model
            return model
