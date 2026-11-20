"""Chat service used by the Streamlit UI."""

from __future__ import annotations

from typing import Any

from config.prompts import LLM_ERROR_MESSAGES
from config.settings import Settings
from src.embeddings.embedder import get_embedding_service
from src.llm.base import BaseLLMProvider
from src.llm.factory import create_llm_provider
from src.rag.chain import RAGChain
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.retriever import Retriever
from src.utils.document_utils import detect_language
from src.utils.logging import get_logger
from src.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)


class ChatService:
    def __init__(
        self,
        settings: Settings,
        store: ChromaStore | None = None,
        llm: BaseLLMProvider | None = None,
        chain: RAGChain | None = None,
    ) -> None:
        self.settings = settings
        settings.ensure_directories()
        self.store = store or ChromaStore(settings)
        if chain is not None:
            self.chain = chain
        else:
            embedder = get_embedding_service(
                settings.embedding_model,
                settings.embedding_batch_size,
                settings.embedding_device,
            )
            retriever = Retriever(settings, self.store, embedder)
            llm = llm or create_llm_provider(settings)
            self.chain = RAGChain(
                settings,
                retriever,
                llm,
                ContextBuilder(settings.max_context_chars),
            )

    def ask(self, question: str, language: str | None = None) -> dict[str, Any]:
        try:
            return self.chain.ask_question(question, language=language)
        except Exception as exc:
            resolved_language = (
                language
                if language in {"mr", "hi", "en"}
                else detect_language(question or "")
            )
            logger.exception("Unexpected chat service failure: %s", exc)
            return {
                "answer": LLM_ERROR_MESSAGES[resolved_language],
                "language": resolved_language,
                "sources": [],
                "retrieval_status": "internal_error",
                "error_code": "internal_error",
            }

    def status(self) -> dict[str, Any]:
        return {
            "collection": self.settings.collection_name,
            "chunks": self.store.count(),
            "provider": self.chain.provider_name,
            "model": self.chain.model_name,
            "embedding_model": self.settings.embedding_model,
            "top_k": self.settings.top_k,
            "relevance_threshold": self.settings.relevance_threshold,
            "llm_provider": self.settings.llm_provider,
            "enable_general_llm_fallback": self.settings.enable_general_llm_fallback,
        }


def create_chat_service(settings: Settings) -> ChatService:
    return ChatService(settings)
