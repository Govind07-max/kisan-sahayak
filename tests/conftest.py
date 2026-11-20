"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings  # noqa: E402
from src.llm.base import LLMError, LLMResponse  # noqa: E402
from src.retrieval.retriever import RetrievedChunk  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        document_dir=tmp_path / "docs",
        inventory_path=tmp_path / "inventory.csv",
        chroma_path=tmp_path / "chroma",
        manifest_path=tmp_path / "manifest.json",
        processed_dir=tmp_path / "processed",
        failed_dir=tmp_path / "failed",
        log_file=tmp_path / "logs" / "test.log",
        collection_name="test_farmer_knowledge",
        embedding_batch_size=2,
        chunk_size=300,
        chunk_overlap=50,
        top_k=3,
        relevance_threshold=None,
        ocr_enabled=False,
        gemini_api_key=None,
        gemini_model="",
    )


@pytest.fixture
def make_pdf(tmp_path: Path):
    import pymupdf

    def _make(name: str, pages: list[str]) -> Path:
        path = tmp_path / name
        document = pymupdf.open()
        for text in pages:
            page = document.new_page()
            page.insert_textbox(
                pymupdf.Rect(50, 50, 550, 750),
                text,
                fontsize=11,
                fontname="helv",
            )
        document.save(path)
        document.close()
        return path

    return _make


class FakeEmbedder:
    """Deterministic small vectors: no model download needed for unit tests."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def encode_documents(self, texts):
        return [self._vector(text) for text in texts]

    def encode_query(self, text):
        return self._vector(text)

    def _vector(self, text: str):
        vector = [0.0] * self.dimension
        for index, char in enumerate(text.lower()):
            vector[index % self.dimension] += (ord(char) % 17) / 100.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


class FakeLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, answer: str = "Answer from documents [S1].") -> None:
        self.answer = answer
        self.calls = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        if isinstance(self.answer, Exception):
            raise self.answer
        return LLMResponse(text=self.answer, provider=self.name, model=self.model)

    def list_models(self) -> list[str]:
        return [self.model]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


def make_chunks(count: int = 2) -> list[RetrievedChunk]:
    chunks = []
    for index in range(count):
        page = index + 2
        chunks.append(
            RetrievedChunk(
                chunk_id=f"chunk-{index}",
                text=(
                    "PM-KISAN provides income support of Rs 6000 per year to "
                    f"eligible small and marginal farmer families. Excerpt {index}."
                ),
                metadata={
                    "document_id": "GRD-001",
                    "document_title": "PM-KISAN FAQ",
                    "source_filename": "pm-kisan.pdf",
                    "source_path": "data/raw/central_schemes/pm-kisan.pdf",
                    "source_url": "",
                    "page_number": page,
                    "page_end": page,
                    "category": "central_schemes",
                    "scheme_or_crop": "PM-KISAN",
                    "language": "en",
                    "publication_date": "",
                    "validity_status": "historical/undated",
                    "chunk_index": index,
                },
                distance=0.2,
                score=0.8,
            )
        )
    return chunks


class FakeStore:
    def __init__(self, count: int = 1) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class FakeRetriever:
    def __init__(self, chunks=None, count: int = 1) -> None:
        self.chunks = chunks if chunks is not None else make_chunks(2)
        self.store = FakeStore(count)

    def retrieve(self, question, top_k=None, threshold=None, category=None):
        return self.chunks
