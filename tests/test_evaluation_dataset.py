"""Evaluation dataset shape tests."""

from __future__ import annotations

import json
from pathlib import Path

QUESTIONS = Path(__file__).resolve().parent.parent / "evaluation" / "questions.jsonl"


def _load():
    return [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_thirty_questions_ten_per_language():
    questions = _load()
    assert len(questions) == 30
    counts = {"mr": 0, "hi": 0, "en": 0}
    for item in questions:
        counts[item["language"]] += 1
    assert counts == {"mr": 10, "hi": 10, "en": 10}


def test_questions_have_required_fields_and_unique_ids():
    questions = _load()
    ids = [item["id"] for item in questions]
    assert len(ids) == len(set(ids))
    for item in questions:
        assert item["question"].strip()
        assert isinstance(item["answerable"], bool)
        assert isinstance(item["expected_doc_ids"], list)
        assert isinstance(item["expected_keywords"], list)
        if item["answerable"]:
            assert item["expected_doc_ids"]
            assert item["expected_keywords"]
        else:
            assert item["expected_doc_ids"] == []


def test_dataset_includes_unsupported_questions():
    questions = _load()
    unsupported = [item for item in questions if not item["answerable"]]
    assert len(unsupported) == 6
