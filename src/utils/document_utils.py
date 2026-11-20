"""Small, reusable helpers for files, text, language detection, and security."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

# Devanagari block plus common punctuation used by Marathi and Hindi.
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_MARATHI_CUES = (
    "आहे",
    "आहेत",
    "काय",
    "कसे",
    "कशी",
    "कसा",
    "मला",
    "माझ्या",
    "योजनेची",
    "योजनेचा",
    "योजनेबद्दल",
    "माहिती",
    "साठी",
    "करून",
    "शेतकरी",
    "अनुदान",
    "अर्ज",
    "पीक",
    "सिंचन",
    "मिळू",
    "मिळते",
)

_HINDI_CUES = (
    "है",
    "हैं",
    "क्या",
    "कैसे",
    "किस",
    "मुझे",
    "मेरे",
    "योजना",
    "योजनाओं",
    "जानकारी",
    "किसान",
    "अनुदान",
    "आवेदन",
    "फसल",
    "सिंचाई",
    "मिलता",
    "मिलती",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_chunk_id(
    document_id: str, page_number: int, chunk_index: int, text: str
) -> str:
    """Deterministic ID so re-ingestion updates instead of duplicating."""
    raw = f"{document_id}|{page_number}|{chunk_index}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def word_count(text: str) -> int:
    return len(text.split())


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str, ui_language: str | None = None) -> str:
    """Return ``mr``, ``hi``, or ``en``.

    The UI can pass an explicit language.  For typed questions we use script
    plus common function words.  This is intentionally simple: the embedding
    model is multilingual regardless of the detected label.
    """
    if ui_language in {"mr", "hi", "en"}:
        return ui_language

    devanagari = len(_DEVANAGARI_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if devanagari == 0:
        return "en"
    if devanagari >= latin:
        marathi_score = sum(text.count(cue) for cue in _MARATHI_CUES)
        hindi_score = sum(text.count(cue) for cue in _HINDI_CUES)
        if marathi_score > hindi_score:
            return "mr"
        if hindi_score > marathi_score:
            return "hi"
        # Ambiguous Devanagari: Marathi is the primary target language of this
        # project, so prefer it.  The UI selector overrides this default.
        return "mr"
    return "en"


def sanitize_question(question: str, min_chars: int = 2, max_chars: int = 1500) -> str:
    """Reject empty/oversized inputs and remove control characters."""
    if not isinstance(question, str):
        raise ValueError("The question must be text.")
    cleaned = "".join(
        char for char in question if char in "\n\t" or not unicodedata_control(char)
    )
    cleaned = normalize_whitespace(cleaned)
    if len(cleaned) < min_chars:
        raise ValueError("The question is too short.")
    if len(cleaned) > max_chars:
        raise ValueError(f"The question must be at most {max_chars} characters.")
    return cleaned


def unicodedata_control(char: str) -> bool:
    import unicodedata

    return unicodedata.category(char).startswith("C")


def redact_sensitive_text(text: str) -> str:
    """Best-effort redaction before text is sent to a remote LLM.

    The knowledge base contains public documents, not farmer records.  Portal
    manuals can nevertheless contain screenshot examples.  We remove Aadhaar-like
    12-digit groups and long account-like numbers.
    """
    text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "[REDACTED-ID]", text)
    text = re.sub(r"\b\d{9,18}\b", "[REDACTED-NUMBER]", text)
    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[REDACTED-EMAIL]",
        text,
    )
    return text


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())
