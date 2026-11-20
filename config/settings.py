"""Application configuration.

All configuration is read from environment variables (and an optional `.env`
file).  Paths are resolved relative to the project root so the application can
be started from any working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SettingsError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "Kisan Sahayak"
    log_level: str = "INFO"
    log_file: Path = PROJECT_ROOT / "logs" / "kisan_sahayak.log"

    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = ""
    gemini_timeout_seconds: int = Field(default=60, ge=10, le=300)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = Field(default=120, ge=10, le=600)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    llm_max_output_tokens: int = Field(default=1400, ge=128, le=8192)

    # ------------------------------------------------------------------
    # Embeddings and vector store
    # ------------------------------------------------------------------
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_device: str = "cpu"
    chroma_path: Path = PROJECT_ROOT / "storage" / "chroma_db"
    collection_name: str = "farmer_knowledge"

    # ------------------------------------------------------------------
    # Documents and chunking
    # ------------------------------------------------------------------
    document_dir: Path = PROJECT_ROOT / "data" / "raw"
    inventory_path: Path = (
        PROJECT_ROOT / "data" / "metadata" / "document_inventory.csv"
    )
    manifest_path: Path = PROJECT_ROOT / "storage" / "ingestion_manifest.json"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    failed_dir: Path = PROJECT_ROOT / "data" / "failed"

    chunk_size: int = Field(default=200, ge=50, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)

    # ------------------------------------------------------------------
    # Retrieval / RAG
    # ------------------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=20)
    relevance_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    max_chunks_per_document: int = Field(default=2, ge=1, le=20)
    query_expansion_enabled: bool = True
    max_context_chars: int = Field(default=14000, ge=1000, le=60000)
    min_question_chars: int = Field(default=2, ge=1)
    max_question_chars: int = Field(default=1500, ge=50)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    ocr_enabled: bool = True
    ocr_languages: str = "mar+hin+eng"
    ocr_dpi: int = Field(default=180, ge=100, le=400)
    ocr_min_chars: int = Field(default=50, ge=0)
    ocr_page_timeout_seconds: int = Field(default=120, ge=10, le=600)
    tesseract_cmd: str = ""

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator(
        "chroma_path",
        "document_dir",
        "inventory_path",
        "manifest_path",
        "processed_dir",
        "failed_dir",
        "log_file",
        mode="after",
    )
    @classmethod
    def _resolve_paths(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        return (PROJECT_ROOT / value).resolve()

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _empty_key_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value if value.get_secret_value().strip() else None
        text = str(value).strip()
        return text or None

    @field_validator("gemini_model", mode="before")
    @classmethod
    def _strip_model(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("relevance_threshold", mode="before")
    @classmethod
    def _empty_threshold_to_none(cls, value: object) -> object:
        if value in (None, ""):
            return None
        return value

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def gemini_api_key_value(self) -> str:
        """Return the plain key without ever logging it."""
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else ""

    def ensure_directories(self) -> None:
        for path in (
            self.chroma_path,
            self.processed_dir,
            self.failed_dir,
            self.manifest_path.parent,
            self.log_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_llm_settings(self, require_model: bool = True) -> None:
        """Fail early with a clear message when the selected provider is incomplete."""
        if self.llm_provider == "gemini":
            if not self.gemini_api_key_value:
                raise SettingsError(
                    "GEMINI_API_KEY is not set. Add it to your .env file or set "
                    "LLM_PROVIDER=ollama."
                )
            if require_model and not self.gemini_model:
                raise SettingsError(
                    "GEMINI_MODEL is not set. Choose a model currently available "
                    "to your API key (see README: listing Gemini models)."
                )
        elif self.llm_provider == "ollama":
            if not self.ollama_base_url.strip():
                raise SettingsError("OLLAMA_BASE_URL is not set.")
            if require_model and not self.ollama_model.strip():
                raise SettingsError("OLLAMA_MODEL is not set.")

    def public_summary(self) -> dict[str, object]:
        """Configuration values that are safe to display in logs or the UI."""
        return {
            "llm_provider": self.llm_provider,
            "gemini_model": self.gemini_model,
            "ollama_model": self.ollama_model,
            "embedding_model": self.embedding_model,
            "collection_name": self.collection_name,
            "top_k": self.top_k,
            "relevance_threshold": self.relevance_threshold,
            "max_chunks_per_document": self.max_chunks_per_document,
            "query_expansion_enabled": self.query_expansion_enabled,
            "document_dir": str(self.document_dir),
            "chroma_path": str(self.chroma_path),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
