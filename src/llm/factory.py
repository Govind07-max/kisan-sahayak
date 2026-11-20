"""Provider factory.  Failures are explicit; we never silently switch providers."""

from __future__ import annotations

from config.settings import Settings
from src.llm.base import BaseLLMProvider, LLMError
from src.llm.gemini import GeminiProvider
from src.llm.ollama import OllamaProvider


def create_llm_provider(
    settings: Settings, require_model: bool = True
) -> BaseLLMProvider:
    if settings.llm_provider == "gemini":
        return GeminiProvider(settings, require_model=require_model)
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings, require_model=require_model)
    raise LLMError(
        f"Unsupported LLM_PROVIDER: {settings.llm_provider}",
        code="configuration",
    )
