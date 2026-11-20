"""Persistent ChromaDB collection wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStoreError(RuntimeError):
    """A ChromaDB operation failed."""


class ChromaStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = Path(settings.chroma_path)
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            self.collection = self.client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB ready at %s (collection=%s, count=%d)",
                self.path,
                settings.collection_name,
                self.collection.count(),
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not open ChromaDB: {exc}") from exc

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:
            raise VectorStoreError(f"Could not count collection: {exc}") from exc

    def upsert_chunks(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise VectorStoreError("Mismatched lengths for ids/documents/embeddings/metadatas.")
        try:
            self.collection.upsert(
                ids=list(ids),
                documents=list(documents),
                embeddings=[list(vector) for vector in embeddings],
                metadatas=list(metadatas),
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB upsert failed: {exc}") from exc

    def delete_ids(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        try:
            self.collection.delete(ids=list(ids))
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB delete by ID failed: {exc}") from exc

    def delete_document(self, document_id: str) -> None:
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB delete by document failed: {exc}") from exc

    def ids_for_document(self, document_id: str) -> list[str]:
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                include=[],
            )
            return list(result.get("ids", []))
        except Exception as exc:
            raise VectorStoreError(f"Could not list chunks for {document_id}: {exc}") from exc

    def query(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
        category: str | None = None,
    ) -> dict[str, Any]:
        if n_results <= 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        where = {"category": category} if category else None
        try:
            return self.collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

    def reset(self) -> None:
        """Delete all chunks.  Intended for tests and explicit maintenance only."""
        try:
            self.client.delete_collection(self.settings.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not reset collection: {exc}") from exc

    def statistics(self) -> dict[str, Any]:
        return {
            "collection": self.settings.collection_name,
            "path": str(self.path),
            "count": self.count(),
        }
