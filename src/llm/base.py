"""Provider-neutral LLM interface and error model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class LLMError(RuntimeError):
    """A safe, user-facing LLM error.

    ``detail`` is sanitized and is intended for logs/debugging.  API keys must
    never appear in it.
    """

    def __init__(self, message: str, code: str = "unknown", detail: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


class BaseLLMProvider(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> list[str]:
        raise NotImplementedError
