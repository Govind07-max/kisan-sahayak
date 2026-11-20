"""Multilingual scheme-name expansion tests."""

from __future__ import annotations

from src.retrieval.query_expansion import expand_query


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


def test_unsupported_current_weather_question_is_not_expanded():
    question = "आज पुण्यात पाऊस पडेल का?"
    assert expand_query(question) == question


def test_short_abbreviation_does_not_match_inside_word():
    question = "What is rainfall in Pune?"
    assert expand_query(question) == question
