"""Semantic retrieval with configurable relevance filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings
from src.embeddings.embedder import EmbeddingError, EmbeddingService
from src.retrieval.query_expansion import build_retrieval_queries
from src.utils.logging import get_logger
from src.vectorstore.chroma_store import ChromaStore, VectorStoreError

logger = get_logger(__name__)


class RetrievalError(RuntimeError):
    """Retrieval could not be completed."""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 1.0
    score: float = 0.0


class Retriever:
    def __init__(
        self,
        settings: Settings,
        store: ChromaStore,
        embedder: EmbeddingService,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        threshold: float | None = None,
        category: str | None = None,
    ) -> list[RetrievedChunk]:
        if not question or not question.strip():
            return []
        k = top_k or self.settings.top_k
        if k <= 0:
            raise RetrievalError("top_k must be positive.")
        if self.store.count() == 0:
            logger.info("Retrieval skipped: collection is empty.")
            return []

        used_threshold = (
            self.settings.relevance_threshold if threshold is None else threshold
        )
        search_queries = (
            build_retrieval_queries(question.strip())
            if self.settings.query_expansion_enabled
            else [question.strip()]
        )
        per_document_cap = max(0, self.settings.max_chunks_per_document)
        # Fetch extra candidates so per-document diversification does not
        # starve the final result list.
        fetch_k = min(
            max(k * (per_document_cap + 2), k + 8) if per_document_cap else k,
            100,
        )

        candidate_lists: list[list[RetrievedChunk]] = []
        try:
            for search_query in search_queries:
                query_embedding = self.embedder.encode_query(search_query)
                result = self.store.query(
                    query_embedding=query_embedding,
                    n_results=fetch_k,
                    category=category,
                )
                ids = (result.get("ids") or [[]])[0]
                documents = (result.get("documents") or [[]])[0]
                metadatas = (result.get("metadatas") or [[]])[0]
                distances = (result.get("distances") or [[]])[0]

                query_candidates: list[RetrievedChunk] = []
                for index, chunk_id in enumerate(ids):
                    text = documents[index] if index < len(documents) else ""
                    metadata = metadatas[index] if index < len(metadatas) else {}
                    distance = (
                        float(distances[index]) if index < len(distances) else 1.0
                    )
                    score = 1.0 - distance  # Chroma cosine distance
                    if used_threshold is not None and score < used_threshold:
                        continue
                    query_candidates.append(
                        RetrievedChunk(
                            chunk_id=str(chunk_id),
                            text=text or "",
                            metadata=metadata or {},
                            distance=distance,
                            score=score,
                        )
                    )
                candidate_lists.append(query_candidates)
        except (EmbeddingError, VectorStoreError) as exc:
            raise RetrievalError(str(exc)) from exc
        except Exception as exc:
            raise RetrievalError(f"Unexpected retrieval failure: {exc}") from exc

        if not candidate_lists or all(not items for items in candidate_lists):
            candidates: list[RetrievedChunk] = []
        elif len(candidate_lists) == 1:
            candidates = candidate_lists[0]
        else:
            # A single conflicting query can drown the section-level intent in
            # generic definitions.  Keep the best few results from each query,
            # merge by chunk ID, and rank the merged candidates by similarity.
            per_query_limit = max(1, min(k, 3))
            fused: dict[str, RetrievedChunk] = {}
            for query_candidates in candidate_lists:
                for candidate in query_candidates[:per_query_limit]:
                    existing = fused.get(candidate.chunk_id)
                    if existing is None or candidate.score > existing.score:
                        fused[candidate.chunk_id] = candidate

            if len(fused) < k:
                for query_candidates in candidate_lists:
                    for candidate in query_candidates:
                        if candidate.chunk_id not in fused:
                            fused[candidate.chunk_id] = candidate
                        if len(fused) >= k:
                            break
                    if len(fused) >= k:
                        break

            candidates = sorted(
                fused.values(), key=lambda candidate: candidate.score, reverse=True
            )

        chunks: list[RetrievedChunk] = []
        document_counts: dict[str, int] = {}
        selected_ids: set[str] = set()
        for candidate in candidates:
            document_id = str(candidate.metadata.get("document_id", ""))
            if (
                per_document_cap
                and document_counts.get(document_id, 0) >= per_document_cap
            ):
                continue
            document_counts[document_id] = document_counts.get(document_id, 0) + 1
            chunks.append(candidate)
            selected_ids.add(candidate.chunk_id)
            if len(chunks) >= k:
                break

        # If diversification leaves fewer than k results (for example, when one
        # manual dominates the query), fill the remaining slots with the next
        # best chunks.  This keeps the context complete while still preferring
        # document diversity when it is available.
        if len(chunks) < k:
            for candidate in candidates:
                if candidate.chunk_id in selected_ids:
                    continue
                chunks.append(candidate)
                if len(chunks) >= k:
                    break

        logger.info(
            "Retrieved %d chunk(s) for query (top_k=%d, fetched=%d, queries=%d, "
            "threshold=%s, per_document_cap=%d)",
            len(chunks),
            k,
            fetch_k,
            len(search_queries),
            used_threshold,
            per_document_cap,
        )
        return chunks
