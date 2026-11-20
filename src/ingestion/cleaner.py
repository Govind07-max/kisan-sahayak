"""Conservative text cleaning for English, Marathi, and Hindi documents.

The cleaner deliberately avoids aggressive transformations: eligibility clauses,
numbers, tables, and Devanagari terminology must survive intact.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.ingestion.extractor import ExtractedPage
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*page\s*\d{1,4}\s*(of|/)\s*\d{1,4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*(of|/)\s*\d{1,4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*page\s*\d{1,4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*p\.?\s*\d{1,4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*(पृष्ठ|पेज|पान)\s*(क्र\.?|number|no\.?)?\s*[:.-]?\s*\d{1,4}\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,4}\s*$"),
]


@dataclass
class CleanedPage:
    page_number: int
    text: str
    original_char_count: int
    removed_lines: int = 0


def _normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    line = line.replace("\u00ad", "")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def _looks_like_page_number(line: str) -> bool:
    return any(pattern.match(line) for pattern in _PAGE_NUMBER_PATTERNS)


def _edge_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return lines[:2] + (lines[-2:] if len(lines) > 2 else [])


def _detect_repeated_edge_lines(pages: list[ExtractedPage]) -> set[str]:
    """Find headers/footers that repeat at page edges.

    Only edge lines are candidates, which avoids deleting repeated table headers
    from the middle of a page.  A line must occur on at least 30% of pages (and
    at least three pages) to be considered boilerplate.
    """
    if len(pages) < 3:
        return set()

    occurrences: Counter[str] = Counter()
    page_sets: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        for line in set(_edge_lines(page.text)):
            normalized = _normalize_line(line)
            if not (3 <= len(normalized) <= 150):
                continue
            if _looks_like_page_number(normalized):
                continue
            occurrences[normalized] += 1
            page_sets[normalized].add(page.page_number)

    threshold = max(3, math.ceil(len(pages) * 0.30))
    repeated = {
        line for line, count in occurrences.items() if len(page_sets[line]) >= threshold
    }
    if repeated:
        logger.info("Detected %d repeated header/footer line(s)", len(repeated))
    return repeated


def _clean_page_text(
    text: str,
    repeated_lines: set[str],
    page_count: int,
) -> tuple[str, int]:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        char
        for char in text
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )

    lines = text.split("\n")
    cleaned_lines: list[str] = []
    removed = 0
    nonempty_positions = [i for i, line in enumerate(lines) if line.strip()]

    for index, line in enumerate(lines):
        normalized = _normalize_line(line)
        if not normalized:
            cleaned_lines.append("")
            continue

        is_edge = False
        if nonempty_positions:
            first = nonempty_positions[:2]
            last = nonempty_positions[-2:]
            is_edge = index in first or index in last

        if is_edge and page_count >= 3 and normalized in repeated_lines:
            removed += 1
            continue
        if is_edge and _looks_like_page_number(normalized):
            removed += 1
            continue

        cleaned_lines.append(normalized)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip(), removed


def clean_pages(pages: list[ExtractedPage]) -> list[CleanedPage]:
    repeated = _detect_repeated_edge_lines(pages)
    cleaned: list[CleanedPage] = []
    for page in pages:
        text, removed = _clean_page_text(page.text, repeated, len(pages))
        cleaned.append(
            CleanedPage(
                page_number=page.page_number,
                text=text,
                original_char_count=page.char_count,
                removed_lines=removed,
            )
        )
    retained = sum(len(page.text) for page in cleaned)
    original = sum(page.char_count for page in pages)
    if original:
        logger.info(
            "Cleaning retained %d of %d characters (%.1f%%)",
            retained,
            original,
            100.0 * retained / original,
        )
    return cleaned
