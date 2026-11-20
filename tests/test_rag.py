"""RAG chain tests with a fake LLM (no Gemini quota is consumed)."""

from __future__ import annotations

import pytest

from config.prompts import NOT_FOUND_MESSAGES
from config.settings import Settings
from src.llm.base import LLMError
from src.rag.chain import RAGChain
from tests.conftest import FakeLLM, FakeRetriever, make_chunks


def _chain(settings: Settings, retriever=None, llm=None) -> RAGChain:
    return RAGChain(
        settings,
        retriever or FakeRetriever(),
        llm or FakeLLM(),
    )


def test_context_is_prepared_and_citations_validated(settings: Settings):
    llm = FakeLLM("Eligible farmers receive Rs 6000 per year [S1].")
    chain = _chain(settings, llm=llm)
    result = chain.ask_question("What is PM-KISAN?", language="en")
    assert result["retrieval_status"] == "success"
    assert result["citation_status"] == "verified"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_title"] == "PM-KISAN FAQ"
    assert result["sources"][0]["page"] == 2
    assert "PM-KISAN FAQ" in llm.last_user_prompt
    assert "UNTRUSTED" in llm.last_user_prompt


def test_no_relevant_documents_avoids_llm(settings: Settings):
    llm = FakeLLM()
    chain = _chain(settings, FakeRetriever(chunks=[], count=1), llm)
    result = chain.ask_question("What is PM-KISAN?", language="en")
    assert result["retrieval_status"] == "no_relevant_documents"
    assert result["sources"] == []
    assert llm.calls == 0


def test_empty_collection_avoids_llm(settings: Settings):
    llm = FakeLLM()
    chain = _chain(settings, FakeRetriever(chunks=[], count=0), llm)
    result = chain.ask_question("What is PM-KISAN?", language="en")
    assert result["retrieval_status"] == "no_documents"
    assert llm.calls == 0


def test_llm_failure_is_handled(settings: Settings):
    llm = FakeLLM(LLMError("quota exceeded", code="quota"))
    result = _chain(settings, llm=llm).ask_question("What is PM-KISAN?", language="en")
    assert result["retrieval_status"] == "llm_error"
    assert result["error_code"] == "quota"
    assert result["sources"] == []


def test_empty_llm_response_is_handled(settings: Settings):
    result = _chain(settings, llm=FakeLLM("")).ask_question(
        "What is PM-KISAN?", language="en"
    )
    assert result["retrieval_status"] == "empty_response"


def test_invalid_source_ids_are_not_shown_as_citations(settings: Settings):
    result = _chain(settings, llm=FakeLLM("Invented claim [S9].")).ask_question(
        "What is PM-KISAN?", language="en"
    )
    assert "[S9]" not in result["answer"]
    assert result["citation_status"] == "unverified"
    assert all(source["source_id"] in {"S1", "S2"} for source in result["sources"])


def test_insufficient_context_response(settings: Settings):
    result = _chain(settings, llm=FakeLLM(NOT_FOUND_MESSAGES["en"])).ask_question(
        "What is the current MSP for wheat?", language="en"
    )
    assert result["retrieval_status"] == "insufficient_context"
    assert result["sources"] == []


def test_marathi_language_is_preserved(settings: Settings):
    result = _chain(settings, llm=FakeLLM("पीएम किसान योजना [S1].")).ask_question(
        "पीएम किसान योजनेची माहिती द्या."
    )
    assert result["language"] == "mr"
    assert result["retrieval_status"] == "success"


def test_invalid_question_is_rejected(settings: Settings):
    result = _chain(settings).ask_question(" ", language="en")
    assert result["retrieval_status"] == "invalid_question"


def test_sources_are_deduplicated(settings: Settings):
    chunks = make_chunks(2)
    chunks[1].metadata = dict(chunks[0].metadata)
    llm = FakeLLM("Answer [S1] [S2].")
    result = _chain(settings, FakeRetriever(chunks=chunks), llm).ask_question(
        "What is PM-KISAN?", language="en"
    )
    assert len(result["sources"]) == 1


def test_provider_and_model_reported(settings: Settings):
    result = _chain(settings).ask_question("What is PM-KISAN?", language="en")
    assert result["provider"] == "fake"
    assert result["model"] == "fake-model"


def test_private_identifiers_are_redacted_before_llm(settings: Settings):
    llm = FakeLLM("Information about PM-KISAN [S1].")
    chain = _chain(settings, llm=llm)
    chain.ask_question(
        "My Aadhaar is 1234 5678 9012. What is PM-KISAN?", language="en"
    )
    assert "1234 5678 9012" not in llm.last_user_prompt
    assert "[REDACTED-ID]" in llm.last_user_prompt
