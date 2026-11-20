"""Chunking tests."""

from __future__ import annotations

from src.ingestion.chunker import TextChunker
from src.ingestion.cleaner import CleanedPage
from src.utils.document_utils import stable_chunk_id


def _page(number: int, words: int, token: str) -> CleanedPage:
    text = " ".join([f"{token}{index}" for index in range(words)])
    return CleanedPage(page_number=number, text=text, original_char_count=len(text))


def test_chunk_creation_and_metadata():
    pages = [_page(1, 220, "alpha"), _page(2, 220, "beta"), _page(3, 220, "gamma")]
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) >= 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.text.strip()
        assert 1 <= chunk.page_start <= 3
        assert chunk.page_start <= chunk.page_end <= 3
        assert chunk.word_count == len(chunk.text.split())


def test_chunk_overlap_exists():
    pages = [_page(1, 400, "word")]
    chunks = TextChunker(chunk_size=250, chunk_overlap=60).chunk_pages(pages)
    assert len(chunks) >= 2
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words


def test_sliding_chunks_have_exact_word_overlap():
    pages = [_page(1, 400, "word")]
    chunks = TextChunker(chunk_size=200, chunk_overlap=100).chunk_pages(pages)
    assert len(chunks) >= 3
    for index in range(len(chunks) - 1):
        current = chunks[index].text.split()
        following = chunks[index + 1].text.split()
        assert current[-100:] == following[:100]
        assert chunks[index].word_count <= 200


def test_empty_text_returns_no_chunks():
    chunks = TextChunker().chunk_pages([CleanedPage(1, "", 0)])
    assert chunks == []


def test_short_document_is_one_chunk():
    pages = [_page(1, 20, "short")]
    chunks = TextChunker(chunk_size=600, chunk_overlap=100).chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].word_count == 20


def test_stable_chunk_id_is_deterministic():
    first = stable_chunk_id("GRD-001", 1, 0, "same text")
    second = stable_chunk_id("GRD-001", 1, 0, "same text")
    other_page = stable_chunk_id("GRD-001", 2, 0, "same text")
    assert first == second
    assert first != other_page
    assert len(first) == 32


def test_chunks_do_not_start_with_empty_text():
    pages = [
        CleanedPage(1, "1. Eligibility\n\nSmall and marginal farmers are eligible.", 50),
        CleanedPage(2, "2. Benefits\n\nThe scheme provides income support.", 50),
    ]
    chunks = TextChunker(chunk_size=100, chunk_overlap=20).chunk_pages(pages)
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
