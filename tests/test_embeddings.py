"""Real embedding model tests (multilingual, normalized, batched)."""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import Settings
from src.embeddings.embedder import get_embedding_service


@pytest.fixture(scope="session")
def embedding_service():
    # Load once for the whole test session.
    service = get_embedding_service(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        4,
        "cpu",
    )
    return service


def test_embedding_dimension_and_normalization(embedding_service):
    vectors = embedding_service.encode_documents(
        ["PM-KISAN income support for farmers.", "पीएम किसान योजना।"]
    )
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    norms = [float(np.linalg.norm(vector)) for vector in vectors]
    assert all(abs(norm - 1.0) < 1e-4 for norm in norms)


def test_batch_processing(embedding_service):
    texts = ["one", "two", "three", "four", "five"]
    vectors = embedding_service.encode_documents(texts)
    assert len(vectors) == len(texts)
    assert all(len(vector) == 384 for vector in vectors)


def test_multilingual_similarity(embedding_service):
    vectors = embedding_service.encode_documents(
        [
            "The scheme provides income support to small farmers.",
            "योजना छोटे किसानों को आय सहायता देती है।",
            "योजना लहान शेतकऱ्यांना उत्पन्न सहाय्य देते.",
        ]
    )
    english_hindi = float(np.dot(vectors[0], vectors[1]))
    english_marathi = float(np.dot(vectors[0], vectors[2]))
    assert english_hindi > 0.4
    assert english_marathi > 0.4


def test_empty_document_batch(embedding_service):
    assert embedding_service.encode_documents([]) == []


def test_query_embedding(embedding_service):
    vector = embedding_service.encode_query("पीएम किसान योजनेची माहिती द्या.")
    assert len(vector) == 384
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-4
