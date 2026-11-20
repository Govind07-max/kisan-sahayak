"""Chat service tests."""

from __future__ import annotations

import pytest

from config.settings import Settings
from src.services.chat_service import ChatService


class FakeChain:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, response=None, error=None):
        self.response = response or {
            "answer": "Answer",
            "language": "en",
            "sources": [],
            "retrieval_status": "success",
        }
        self.error = error
        self.calls = []

    def ask_question(self, question, language=None):
        self.calls.append((question, language))
        if self.error:
            raise self.error
        return self.response


def test_ask_delegates_to_chain(settings: Settings):
    chain = FakeChain()
    service = ChatService(settings, chain=chain)
    result = service.ask("What is PM-KISAN?", language="en")
    assert result["answer"] == "Answer"
    assert chain.calls == [("What is PM-KISAN?", "en")]


def test_unexpected_error_returns_structured_response(settings: Settings):
    service = ChatService(settings, chain=FakeChain(error=RuntimeError("boom")))
    result = service.ask("What is PM-KISAN?", language="en")
    assert result["retrieval_status"] == "internal_error"
    assert result["answer"]
    assert result["sources"] == []


def test_status_contains_safe_fields(settings: Settings):
    service = ChatService(settings, chain=FakeChain())
    status = service.status()
    assert status["chunks"] == 0
    assert status["provider"] == "fake"
    assert "gemini_api_key" not in status


def test_missing_gemini_key_raises_configuration_error(settings: Settings):
    from config.settings import SettingsError

    with pytest.raises(SettingsError):
        ChatService(settings)


def test_model_listing_does_not_require_a_model_id(settings: Settings):
    from pydantic import SecretStr
    from config.settings import SettingsError

    listing_settings = settings.model_copy(
        update={"gemini_api_key": SecretStr("AIza-test-key-not-used")}
    )
    listing_settings.validate_llm_settings(require_model=False)
    with pytest.raises(SettingsError):
        listing_settings.validate_llm_settings()
