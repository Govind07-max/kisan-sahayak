"""Multilingual scheme-name expansion tests."""

from __future__ import annotations

from src.retrieval.query_expansion import build_retrieval_queries, expand_query


def test_marathi_scheme_alias_adds_english_name():
    expanded = expand_query("पीएम किसान योजनेचा उद्देश काय आहे?")
    assert "PM-KISAN" in expanded
    assert "Pradhan Mantri Kisan Samman Nidhi" in expanded


def test_english_scheme_acronym_is_expanded():
    expanded = expand_query("What is PM-KISAN?")
    assert "Pradhan Mantri Kisan Samman Nidhi" in expanded


def test_hindi_crop_alias_adds_english_name():
    expanded = expand_query("कपास में कीट प्रबंधन कैसे करें?")
    assert "cotton" in expanded
    assert "integrated pest management" in expanded


def test_cotton_method_question_adds_practice_hint():
    queries = build_retrieval_queries(
        "कपास में एकीकृत कीट प्रबंधन के लिए कौन सी विधियाँ बताई गई हैं?"
    )
    assert len(queries) == 2
    assert "cultural practices" in queries[1]
    assert "recommended methods" in queries[1]


def test_crop_insurance_claim_question_adds_procedure_hint():
    queries = build_retrieval_queries("फसल बीमा में दावा कैसे करें?")
    assert len(queries) == 2
    assert "claim intimation" in queries[1]


def test_agriculture_infrastructure_interest_question_adds_hint():
    queries = build_retrieval_queries(
        "कृषि अवसंरचना कोष में ब्याज सब्सिडी कितनी है?"
    )
    assert len(queries) == 2
    assert "interest subvention norms" in queries[1]


def test_mahaagri_objective_question_adds_policy_hint():
    queries = build_retrieval_queries(
        "महाॲग्री-एआय धोरण 2025-2029 चा कालावधी आणि उद्दिष्ट काय आहे?"
    )
    assert len(queries) == 2
    assert "vision objectives" in queries[1]


def test_unsupported_current_weather_question_is_not_expanded():
    question = "आज पुण्यात पाऊस पडेल का?"
    assert expand_query(question) == question
    assert build_retrieval_queries(question) == [question]


def test_short_abbreviation_does_not_match_inside_word():
    question = "What is rainfall in Pune?"
    assert expand_query(question) == question
