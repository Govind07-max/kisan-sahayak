"""Google Gemini provider using the official google-genai SDK."""

from __future__ import annotations

import re

from config.settings import Settings
from src.llm.base import BaseLLMProvider, LLMError, LLMResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)

_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


class GeminiProvider(BaseLLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings, require_model: bool = True) -> None:
        settings.validate_llm_settings(require_model=require_model)
        self.settings = settings
        self._api_key = settings.gemini_api_key_value
        self._model = settings.gemini_model
        try:
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(
                timeout=settings.gemini_timeout_seconds * 1000
            )
            self._client = genai.Client(api_key=self._api_key, http_options=http_options)
            self._types = types
        except Exception as exc:
            raise LLMError(
                "Gemini client could not be initialized.",
                code="configuration",
                detail=self._sanitize(str(exc)),
            ) from exc

    @property
    def model(self) -> str:
        return self._model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            config = self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.settings.llm_temperature,
                max_output_tokens=self.settings.llm_max_output_tokens,
                response_mime_type="text/plain",
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=config,
            )
        except Exception as exc:
            raise self._translate_error(exc) from exc

        text = getattr(response, "text", None)
        if not text:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None)
                content = getattr(candidates[0], "content", None)
                parts = getattr(content, "parts", None) or []
                text = "".join(getattr(part, "text", "") or "" for part in parts)
                if not text:
                    raise LLMError(
                        "Gemini returned no text.",
                        code="empty_response",
                        detail=f"finish_reason={finish_reason}",
                    )
        if not text or not str(text).strip():
            raise LLMError(
                "Gemini returned an empty response.",
                code="empty_response",
                detail="No text in response.",
            )
        logger.info(
            "Gemini response generated (model=%s, chars=%d)", self._model, len(str(text))
        )
        return LLMResponse(text=str(text).strip(), provider=self.name, model=self._model)

    def list_models(self) -> list[str]:
        try:
            names: list[str] = []
            for model in self._client.models.list():
                methods = getattr(model, "supported_generation_methods", None) or []
                name = getattr(model, "name", "") or ""
                if methods and "generateContent" not in methods:
                    continue
                names.append(name.replace("models/", ""))
            return sorted(set(names))
        except Exception as exc:
            raise self._translate_error(exc) from exc

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def _sanitize(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace(self._api_key, "[REDACTED]") if self._api_key else text
        return _KEY_PATTERN.sub("[REDACTED]", text)

    def _translate_error(self, exc: Exception) -> LLMError:
        message = self._sanitize(str(exc))
        lowered = message.lower()
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)

        if any(token in lowered for token in ("api key", "api_key", "unauthenticated", "401", "403", "permission")):
            code = "authentication"
            user_message = (
                "Gemini authentication failed. Check GEMINI_API_KEY and whether the "
                "key has access to the selected model."
            )
        elif any(token in lowered for token in ("quota", "resource_exhausted", "429", "rate limit")):
            code = "quota"
            user_message = (
                "The Gemini free-tier quota or rate limit was exceeded. Please wait "
                "and try again, or check your quota in Google AI Studio."
            )
        elif any(token in lowered for token in ("timeout", "timed out", "deadline")):
            code = "timeout"
            user_message = "The Gemini request timed out. Please try again."
        elif any(token in lowered for token in ("not found", "404", "model")):
            code = "model_not_found"
            user_message = (
                "The configured Gemini model is not available to this API key. "
                "Check GEMINI_MODEL or list available models."
            )
        elif any(token in lowered for token in ("connection", "network", "dns", "ssl", "unavailable")):
            code = "network"
            user_message = "Could not reach the Gemini service. Check your internet connection."
        else:
            code = "unknown"
            user_message = "The Gemini request failed. Please try again."

        return LLMError(
            user_message,
            code=code if status is None else f"{code}:{status}",
            detail=message[:1000],
        )
