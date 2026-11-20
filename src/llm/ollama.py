"""Optional local Ollama provider (no cloud API key required)."""

from __future__ import annotations

import httpx

from config.settings import Settings
from src.llm.base import BaseLLMProvider, LLMError, LLMResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    def __init__(self, settings: Settings, require_model: bool = True) -> None:
        settings.validate_llm_settings(require_model=require_model)
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model

    @property
    def model(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.settings.llm_temperature,
                "num_predict": self.settings.llm_max_output_tokens,
            },
        }
        try:
            with httpx.Client(timeout=self.settings.ollama_timeout_seconds) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise LLMError(
                "The Ollama request timed out.", code="timeout", detail=str(exc)
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response is not None else ""
            if exc.response is not None and exc.response.status_code == 404:
                raise LLMError(
                    "The configured Ollama model is not installed.",
                    code="model_not_found",
                    detail=body,
                ) from exc
            raise LLMError(
                "Ollama returned an HTTP error.",
                code="http_error",
                detail=f"{exc.response.status_code if exc.response is not None else '?'} {body}",
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                "Could not reach Ollama. Is `ollama serve` running?",
                code="network",
                detail=str(exc),
            ) from exc

        text = ""
        if isinstance(data, dict):
            message = data.get("message") or {}
            text = message.get("content", "") if isinstance(message, dict) else ""
            if not text:
                text = data.get("response", "")
        if not text or not str(text).strip():
            raise LLMError(
                "Ollama returned an empty response.",
                code="empty_response",
                detail=str(data)[:500],
            )
        logger.info("Ollama response generated (model=%s, chars=%d)", self._model, len(text))
        return LLMResponse(text=str(text).strip(), provider=self.name, model=self._model)

    def list_models(self) -> list[str]:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise LLMError(
                "Could not list Ollama models. Is `ollama serve` running?",
                code="network",
                detail=str(exc),
            ) from exc
        return sorted(
            model.get("name", "")
            for model in data.get("models", [])
            if isinstance(model, dict) and model.get("name")
        )
