"""Configurable, page-aware word chunking.

The embedding model used by this project has a maximum sequence length of 128
tokens.  Chunks therefore use a true sliding word window (200 words with 100
words of overlap by default), so every part of a long document is represented
by at least one embedded window.  A chunk records its first and last page so
citations remain truthful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingestion.cleaner import CleanedPage
from src.utils.document_utils import normalize_whitespace, word_count
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")


@dataclass
class TextUnit:
    unit_id: int
    text: str
    page_number: int
    words: int
    break_before: bool = True


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    word_count: int
    unit_ids: list[int]


@dataclass
class _Word:
    text: str
    page_number: int
    unit_id: int


def _hard_split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [
        " ".join(words[index : index + max_words])
        for index in range(0, len(words), max_words)
    ]


def _split_long_text(text: str, max_unit_words: int) -> list[str]:
    if word_count(text) <= max_unit_words:
        return [text]

    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if len(sentences) <= 1:
        return _hard_split_words(text, max_unit_words)

    pieces: list[str] = []
    for sentence in sentences:
        if word_count(sentence) <= max_unit_words:
            pieces.append(sentence)
        else:
            pieces.extend(_hard_split_words(sentence, max_unit_words))
    return pieces


def _is_heading(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 12:
        return False
    if text.rstrip().endswith((".", "!", "?", "।", ":", ";")):
        return False
    # Numbered headings such as "3. Eligibility" are useful to keep with the
    # paragraph that follows.
    return True


def _units_from_pages(pages: list[CleanedPage], max_unit_words: int = 220) -> list[TextUnit]:
    units: list[TextUnit] = []
    unit_id = 0

    for page in pages:
        if not page.text.strip():
            continue

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page.text) if part.strip()]
        if not paragraphs:
            paragraphs = [page.text.strip()]

        for paragraph in paragraphs:
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            # Preserve table-like content row by row.  This keeps numbers and
            # column labels readable instead of flattening them into one blob.
            is_table_like = (
                len(lines) > 6
                and sum(len(line.split()) for line in lines) / max(len(lines), 1) <= 12
            )
            if is_table_like:
                pieces = lines
            else:
                pieces = _split_long_text(paragraph, max_unit_words)

            for piece_index, piece in enumerate(pieces):
                piece = normalize_whitespace(piece)
                if not piece:
                    continue
                units.append(
                    TextUnit(
                        unit_id=unit_id,
                        text=piece,
                        page_number=page.page_number,
                        words=word_count(piece),
                        break_before=(piece_index == 0),
                    )
                )
                unit_id += 1

    # Merge short headings into the following block so a chunk does not begin
    # with a dangling heading.
    merged: list[TextUnit] = []
    index = 0
    while index < len(units):
        current = units[index]
        if (
            index + 1 < len(units)
            and _is_heading(current.text)
            and current.words <= 12
            and current.page_number == units[index + 1].page_number
        ):
            following = units[index + 1]
            merged.append(
                TextUnit(
                    unit_id=current.unit_id,
                    text=f"{current.text}\n{following.text}",
                    page_number=current.page_number,
                    words=current.words + following.words,
                    break_before=current.break_before,
                )
            )
            index += 2
        else:
            merged.append(current)
            index += 1

    return merged


def _flatten_words(units: list[TextUnit]) -> list[_Word]:
    """Flatten units while retaining page and unit boundary information."""
    words: list[_Word] = []
    for unit in units:
        for token in unit.text.split():
            words.append(
                _Word(
                    text=token,
                    page_number=unit.page_number,
                    unit_id=unit.unit_id,
                )
            )
    return words


def _window_text(window: list[_Word]) -> str:
    parts: list[str] = []
    previous_unit_id: int | None = None
    for word in window:
        if previous_unit_id is not None:
            parts.append("\n" if word.unit_id != previous_unit_id else " ")
        parts.append(word.text)
        previous_unit_id = word.unit_id
    return normalize_whitespace("".join(parts))


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 100,
        min_chunk_words: int = 30,
        max_chunk_words: int = 800,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max(max_chunk_words, chunk_size)

    def chunk_pages(self, pages: list[CleanedPage]) -> list[Chunk]:
        # Long paragraphs are split into sentence/word units no larger than the
        # chunk size.  The final window is still built from a flat word stream,
        # which guarantees exact word-level overlap between adjacent chunks.
        units = _units_from_pages(
            pages, max_unit_words=max(20, min(220, self.chunk_size))
        )
        words = _flatten_words(units)
        if not words:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[Chunk] = []
        previous_text = ""
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            window = words[start:end]
            text = _window_text(window)
            if text and text != previous_text:
                chunks.append(
                    Chunk(
                        text=text,
                        page_start=window[0].page_number,
                        page_end=window[-1].page_number,
                        chunk_index=len(chunks),
                        word_count=word_count(text),
                        unit_ids=list(dict.fromkeys(word.unit_id for word in window)),
                    )
                )
                previous_text = text

            if end >= len(words):
                break
            start += step

        if chunks:
            logger.info(
                "Created %d chunk(s); words min/avg/max = %d/%d/%d",
                len(chunks),
                min(chunk.word_count for chunk in chunks),
                round(sum(chunk.word_count for chunk in chunks) / len(chunks)),
                max(chunk.word_count for chunk in chunks),
            )
        return chunks

    # Alias used by tests and simple callers.
    def chunk(self, pages: list[CleanedPage]) -> list[Chunk]:
        return self.chunk_pages(pages)
