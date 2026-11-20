"""Tests for the optional, clearly labeled general-LLM fallback."""

from __future__ import annotations

from config.prompts import GENERAL_FALLBACK_NOTICES, NOT_FOUND_MESSAGES
from config.settings import Settings
from src.llm.base import LLMError, LLMResponse
from src.rag.chain import RAGChain
from src.rag.fallback import clean_general_answer, is_general_fallback_blocked
from tests.conftest import FakeLLM, FakeRetriever


class SequentialLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=self.answers.pop(0), provider=self.name, model=self.model
        )


def _enabled(settings: Settings) -> Settings:
    return settings.model_copy(update={"enable_general_llm_fallback": True})


def _chain(settings: Settings, llm, chunks=None) -> RAGChain:
    return RAGChain(settings, FakeRetriever(chunks=chunks or [], count=1), llm)


def test_fallback_is_off_by_default(settings: Settings):
    llm = FakeLLM("General answer [S1].")
    result = _chain(settings, llm).ask_question(
        "Explain crop rotation.", language="en"
    )
    assert result["retrieval_status"] == "no_relevant_documents"
    assert llm.calls == 0


def test_high_risk_questions_are_never_fallen_back_to(settings: Settings):
    llm = FakeLLM("General answer.")
    chain = _chain(_enabled(settings), llm)
    result = chain.ask_question(
        "आज पुण्यात पाऊस पडेल का?", language="mr"
    )
    assert result["retrieval_status"] == "no_relevant_documents"
    assert llm.calls == 0


def test_allowed_question_gets_labeled_ungrounded_answer(settings: Settings):
    llm = FakeLLM("Crop rotation can help soil health [S1]. See https://example.com")
    result = _chain(_enabled(settings), llm).ask_question(
        "Explain crop rotation.", language="en"
    )
    assert result["retrieval_status"] == "general_llm_fallback"
    assert result["is_general_fallback"] is True
    assert result["citation_status"] == "not_applicable"
    assert result["sources"] == []
    assert GENERAL_FALLBACK_NOTICES["en"] in result["answer"]
    assert "[S1]" not in result["answer"]
    assert "https://" not in result["answer"]
    assert llm.calls == 1


def test_fallback_runs_after_insufficient_context(settings: Settings):
    llm = SequentialLLM(
        NOT_FOUND_MESSAGES["en"],
        "In general, crop rotation means changing crops across seasons.",
    )
    result = RAGChain(
        _enabled(settings), FakeRetriever(count=1), llm
    ).ask_question("Explain crop rotation.", language="en")
    assert llm.calls == 2
    assert result["retrieval_status"] == "general_llm_fallback"


def test_failure_of_fallback_keeps_original_refusal(settings: Settings):
    llm = FakeLLM(LLMError("quota", code="quota"))
    result = _chain(_enabled(settings), llm).ask_question(
        "Explain crop rotation.", language="en"
    )
    assert result["retrieval_status"] == "no_relevant_documents"
    assert result["sources"] == []


def test_fallback_blocklist_examples():
    assert is_general_fallback_blocked("What is today's tomato price in Pune?")
    assert is_general_fallback_blocked("What is the current MSP for wheat?")
    assert is_general_fallback_blocked("आज पुण्यात पाऊस पडेल का?")
    assert is_general_fallback_blocked("आज टमाटर का बाजार भाव क्या है?")
    assert not is_general_fallback_blocked("Explain crop rotation.")
    assert not is_general_fallback_blocked("What is PM-KISAN?")


def test_clean_general_answer_removes_citations_and_urls():
    cleaned = clean_general_answer("Facts [S1, S2] at https://example.com here")
    assert "[S1" not in cleaned
    assert "https://" not in cleaned
