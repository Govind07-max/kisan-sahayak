"""Local multilingual embedding model."""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Sequence

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or used."""


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s on %s", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
                dimension_method = getattr(
                    self._model,
                    "get_embedding_dimension",
                    getattr(self._model, "get_sentence_embedding_dimension", None),
                )
                logger.info(
                    "Embedding model loaded; dimension=%s",
                    dimension_method() if callable(dimension_method) else "unknown",
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}'. "
                    "Check your internet connection for the first download and the "
                    "EMBEDDING_MODEL setting. Original error: " + str(exc)
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        model = self._load()
        dimension_method = getattr(
            model,
            "get_embedding_dimension",
            getattr(model, "get_sentence_embedding_dimension", None),
        )
        dimension = dimension_method() if callable(dimension_method) else None
        if not dimension:
            raise EmbeddingError("Embedding model did not report a vector dimension.")
        return int(dimension)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [text if text and text.strip() else " " for text in texts]
        model = self._load()
        try:
            vectors = model.encode(
                cleaned,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Document embedding failed: {exc}") from exc

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise EmbeddingError(f"Unexpected embedding shape: {array.shape}")
        return array.tolist()

    def encode_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty question.")
        model = self._load()
        try:
            vector = model.encode(
                [text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]
        except Exception as exc:
            raise EmbeddingError(f"Question embedding failed: {exc}") from exc
        return np.asarray(vector, dtype=np.float32).tolist()


@lru_cache(maxsize=4)
def get_embedding_service(
    model_name: str,
    batch_size: int = 32,
    device: str = "cpu",
) -> EmbeddingService:
    return EmbeddingService(model_name=model_name, batch_size=batch_size, device=device)
