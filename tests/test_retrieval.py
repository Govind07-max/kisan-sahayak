"""Retrieval tests using a temporary ChromaDB collection."""

from __future__ import annotations

from config.settings import Settings
from src.retrieval.retriever import Retriever
from src.vectorstore.chroma_store import ChromaStore


class FixedEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def encode_query(self, text):
        return self.vector

    def encode_documents(self, texts):
        return [self.vector for _ in texts]


def _add(store: ChromaStore, chunks):
    store.upsert_chunks(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def test_retrieves_relevant_document_with_metadata(settings: Settings):
    store = ChromaStore(settings)
    _add(
        store,
        [
            {
                "id": "pm-kisan-1",
                "text": "PM-KISAN provides income support.",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {
                    "document_id": "GRD-001",
                    "document_title": "PM-KISAN FAQ",
                    "source_filename": "pm-kisan.pdf",
                    "page_number": 2,
                    "page_end": 2,
                    "category": "central_schemes",
                    "language": "en",
                    "source_url": "",
                    "source_path": "data/raw/central_schemes/pm-kisan.pdf",
                    "scheme_or_crop": "PM-KISAN",
                    "publication_date": "",
                    "validity_status": "undated",
                    "chunk_index": 0,
                },
            },
            {
                "id": "soil-1",
                "text": "Soil Health Card provides soil testing.",
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {
                    "document_id": "GRD-037",
                    "document_title": "Soil Health Card",
                    "source_filename": "soil.pdf",
                    "page_number": 3,
                    "page_end": 3,
                    "category": "soil_and_fertilizer",
                    "language": "en",
                    "source_url": "",
                    "source_path": "data/raw/soil_and_fertilizer/soil.pdf",
                    "scheme_or_crop": "Soil Health Card",
                    "publication_date": "2025-08-16",
                    "validity_status": "published 2025-08-16",
                    "chunk_index": 0,
                },
            },
        ],
    )
    retriever = Retriever(settings, store, FixedEmbedder([1.0, 0.0, 0.0]))
    chunks = retriever.retrieve("PM-KISAN", top_k=1)
    assert len(chunks) == 1
    assert chunks[0].metadata["document_id"] == "GRD-001"
    assert chunks[0].metadata["page_number"] == 2
    assert chunks[0].score > 0.9


def test_top_k_limits_results(settings: Settings):
    store = ChromaStore(settings)
    _add(
        store,
        [
            {
                "id": f"doc-{index}",
                "text": f"document {index}",
                "embedding": [1.0, float(index) / 10.0, 0.0],
                "metadata": {
                    "document_id": f"DOC-{index}",
                    "document_title": f"Document {index}",
                    "source_filename": f"doc-{index}.pdf",
                    "page_number": 1,
                    "page_end": 1,
                    "category": "central_schemes",
                    "language": "en",
                    "source_url": "",
                    "source_path": "",
                    "scheme_or_crop": "",
                    "publication_date": "",
                    "validity_status": "",
                    "chunk_index": 0,
                },
            }
            for index in range(4)
        ],
    )
    retriever = Retriever(settings, store, FixedEmbedder([1.0, 0.0, 0.0]))
    assert len(retriever.retrieve("anything", top_k=2)) == 2
    assert len(retriever.retrieve("anything", top_k=4)) == 4


def test_empty_collection(settings: Settings):
    store = ChromaStore(settings)
    retriever = Retriever(settings, store, FixedEmbedder([1.0, 0.0, 0.0]))
    assert retriever.retrieve("PM-KISAN") == []


def test_relevance_threshold_filters(settings: Settings):
    store = ChromaStore(settings)
    _add(
        store,
        [
            {
                "id": "unrelated",
                "text": "unrelated content",
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {
                    "document_id": "DOC-X",
                    "document_title": "Unrelated",
                    "source_filename": "x.pdf",
                    "page_number": 1,
                    "page_end": 1,
                    "category": "central_schemes",
                    "language": "en",
                    "source_url": "",
                    "source_path": "",
                    "scheme_or_crop": "",
                    "publication_date": "",
                    "validity_status": "",
                    "chunk_index": 0,
                },
            }
        ],
    )
    retriever = Retriever(settings, store, FixedEmbedder([1.0, 0.0, 0.0]))
    assert retriever.retrieve("PM-KISAN", threshold=0.5) == []


def test_per_document_diversification(settings: Settings):
    store = ChromaStore(settings)
    chunks = []
    for document_id in ("DOC-A", "DOC-B"):
        for index in range(3):
            chunks.append(
                {
                    "id": f"{document_id}-{index}",
                    "text": f"{document_id} relevant scheme text {index}",
                    "embedding": [1.0, 0.0, 0.0],
                    "metadata": {
                        "document_id": document_id,
                        "document_title": f"Document {document_id}",
                        "source_filename": f"{document_id}.pdf",
                        "page_number": index + 1,
                        "page_end": index + 2,
                        "category": "central_schemes",
                        "language": "en",
                        "source_url": "",
                        "source_path": "",
                        "scheme_or_crop": "PM-KISAN",
                        "publication_date": "",
                        "validity_status": "",
                        "chunk_index": index,
                    },
                }
            )
    _add(store, chunks)
    retriever = Retriever(settings, store, FixedEmbedder([1.0, 0.0, 0.0]))
    results = retriever.retrieve("PM-KISAN", top_k=4)
    counts = {}
    for chunk in results:
        document_id = chunk.metadata["document_id"]
        counts[document_id] = counts.get(document_id, 0) + 1
    assert len(results) == 4
    assert all(count <= settings.max_chunks_per_document for count in counts.values())
    assert set(counts) == {"DOC-A", "DOC-B"}

    # With only six candidates, the diversified first pass cannot fill five
    # slots; the retriever must top up rather than silently return four.
    filled = retriever.retrieve("PM-KISAN", top_k=5)
    assert len(filled) == 5


def test_category_filter(settings: Settings):
    store = ChromaStore(settings)
    _add(
        store,
        [
            {
                "id": "central",
                "text": "central scheme",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {
                    "document_id": "C-1",
                    "document_title": "Central",
                    "source_filename": "c.pdf",
                    "page_number": 1,
                    "page_end": 1,
                    "category": "central_schemes",
                    "language": "en",
                    "source_url": "",
                    "source_path": "",
                    "scheme_or_crop": "",
                    "publication_date": "",
                    "validity_status": "",
                    "chunk_index": 0,
                },
            },
            {
                "id": "state",
                "text": "state scheme",
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {
                    "document_id": "S-1",
                    "document_title": "State",
                    "source_filename": "s.pdf",
                    "page_number": 1,
                    "page_end": 1,
                    "category": "maharashtra_schemes",
                    "language": "mr",
                    "source_url": "",
                    "source_path": "",
                    "scheme_or_crop": "",
                    "publication_date": "",
                    "validity_status": "",
                    "chunk_index": 0,
                },
            },
        ],
    )
    retriever = Retriever(settings, store, FixedEmbedder([1.0, 0.0, 0.0]))
    chunks = retriever.retrieve("scheme", category="maharashtra_schemes")
    assert len(chunks) == 1
    assert chunks[0].metadata["document_id"] == "S-1"
