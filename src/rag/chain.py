"""End-to-end RAG chain with citation validation and explicit failure states."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from config.prompts import (
    CITATION_ERROR_MESSAGES,
    GENERAL_FALLBACK_NOTICES,
    GENERAL_FALLBACK_SYSTEM_PROMPT,
    INVALID_QUESTION_MESSAGES,
    LLM_ERROR_MESSAGES,
    NOT_FOUND_MESSAGES,
    RETRIEVAL_ERROR_MESSAGES,
    SYSTEM_PROMPT,
)
from config.settings import Settings, SettingsError, get_settings
from src.llm.base import BaseLLMProvider, LLMError
from src.rag.fallback import clean_general_answer, is_general_fallback_blocked
from src.rag.prompts import build_general_fallback_prompt, build_user_prompt
from src.retrieval.context_builder import ContextBuilder, ContextBundle
from src.retrieval.retriever import RetrievalError, Retriever
from src.utils.document_utils import (
    detect_language,
    redact_sensitive_text,
    sanitize_question,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Accepts both single markers ([S1]) and combined markers ([S1, S2, S5]).
_SOURCE_MARKER = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def _marker_source_ids(token: str) -> list[int]:
    """Extract unique source numbers from a citation marker token."""
    return list(dict.fromkeys(int(value) for value in re.findall(r"S(\d+)", token)))


def _error_response(
    answer: str,
    language: str,
    status: str,
    error_code: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "answer": answer,
        "language": language,
        "sources": sources or [],
        "retrieval_status": status,
    }
    if error_code:
        response["error_code"] = error_code
    return response


def _dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any, Any]] = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        key = (
            source.get("document_title"),
            source.get("source"),
            source.get("page"),
            source.get("page_end"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_not_found_answer(answer: str) -> bool:
    normalized = _normalize_for_match(answer)
    for message in NOT_FOUND_MESSAGES.values():
        if _normalize_for_match(message) in normalized:
            return True
    return False


class RAGChain:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        llm: BaseLLMProvider,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self.context_builder = context_builder or ContextBuilder(settings.max_context_chars)

    @property
    def provider_name(self) -> str:
        return self.llm.name

    @property
    def model_name(self) -> str:
        return self.llm.model

    def _try_general_fallback(
        self,
        question: str,
        language: str,
        retrieved_chunks: int = 0,
    ) -> dict[str, Any] | None:
        """Return a labeled general answer, or None if the fallback cannot be used."""
        if not self.settings.enable_general_llm_fallback:
            return None
        if is_general_fallback_blocked(question):
            logger.info("General fallback blocked for this question")
            return None

        prompt = build_general_fallback_prompt(question, language)
        try:
            response = self.llm.generate(GENERAL_FALLBACK_SYSTEM_PROMPT, prompt)
        except LLMError as exc:
            logger.error(
                "General fallback LLM failure provider=%s code=%s detail=%s",
                self.provider_name,
                exc.code,
                exc.detail,
            )
            return None
        except Exception as exc:
            logger.exception("Unexpected general fallback failure: %s", exc)
            return None

        answer = clean_general_answer((response.text or "").strip())
        if not answer:
            return None

        notice = GENERAL_FALLBACK_NOTICES.get(
            language, GENERAL_FALLBACK_NOTICES["en"]
        )
        return {
            "answer": f"{notice}\n\n{answer}",
            "language": language,
            "sources": [],
            "retrieval_status": "general_llm_fallback",
            "citation_status": "not_applicable",
            "provider": self.provider_name,
            "model": self.model_name,
            "retrieved_chunks": retrieved_chunks,
            "is_general_fallback": True,
            "fallback_notice": notice,
        }

    def ask_question(
        self,
        question: str,
        language: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        resolved_language = (
            language if language in {"mr", "hi", "en"} else detect_language(question or "")
        )

        try:
            cleaned_question = sanitize_question(
                question,
                min_chars=self.settings.min_question_chars,
                max_chars=self.settings.max_question_chars,
            )
        except ValueError:
            return _error_response(
                INVALID_QUESTION_MESSAGES[resolved_language],
                resolved_language,
                "invalid_question",
                error_code="invalid_question",
            )

        # Never send obvious identifiers from the farmer's question to a remote
        # model.  The original question is only used for language detection.
        safe_question = redact_sensitive_text(cleaned_question)

        try:
            if self.retriever.store.count() == 0:
                return _error_response(
                    NOT_FOUND_MESSAGES[resolved_language],
                    resolved_language,
                    "no_documents",
                    error_code="no_documents",
                )
            chunks = self.retriever.retrieve(
                safe_question,
                top_k=top_k,
                threshold=threshold,
                category=category,
            )
        except RetrievalError as exc:
            logger.error("Retrieval failed: %s", exc)
            return _error_response(
                RETRIEVAL_ERROR_MESSAGES[resolved_language],
                resolved_language,
                "retrieval_error",
                error_code="retrieval_error",
            )
        except Exception as exc:
            logger.exception("Unexpected retrieval failure: %s", exc)
            return _error_response(
                RETRIEVAL_ERROR_MESSAGES[resolved_language],
                resolved_language,
                "retrieval_error",
                error_code="retrieval_error",
            )

        if not chunks:
            refusal = _error_response(
                NOT_FOUND_MESSAGES[resolved_language],
                resolved_language,
                "no_relevant_documents",
                error_code="no_relevant_documents",
            )
            return self._try_general_fallback(safe_question, resolved_language) or refusal

        bundle: ContextBundle = self.context_builder.build(chunks)
        if not bundle.sources or not bundle.context_text:
            return _error_response(
                NOT_FOUND_MESSAGES[resolved_language],
                resolved_language,
                "no_relevant_documents",
                error_code="empty_context",
            )

        user_prompt = build_user_prompt(
            question=safe_question,
            language=resolved_language,
            context_text=bundle.context_text,
            source_count=len(bundle.sources),
        )

        try:
            llm_response = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except LLMError as exc:
            logger.error(
                "LLM failure provider=%s code=%s detail=%s",
                self.provider_name,
                exc.code,
                exc.detail,
            )
            return _error_response(
                LLM_ERROR_MESSAGES[resolved_language],
                resolved_language,
                "llm_error",
                error_code=exc.code,
            )
        except Exception as exc:
            logger.exception("Unexpected LLM failure: %s", exc)
            return _error_response(
                LLM_ERROR_MESSAGES[resolved_language],
                resolved_language,
                "llm_error",
                error_code="unknown",
            )

        answer = (llm_response.text or "").strip()
        if not answer:
            return _error_response(
                LLM_ERROR_MESSAGES[resolved_language],
                resolved_language,
                "empty_response",
                error_code="empty_response",
            )

        if _is_not_found_answer(answer):
            refusal = {
                "answer": NOT_FOUND_MESSAGES[resolved_language],
                "language": resolved_language,
                "sources": [],
                "retrieval_status": "insufficient_context",
                "citation_status": "not_applicable",
                "provider": self.provider_name,
                "model": self.model_name,
                "retrieved_chunks": len(chunks),
            }
            return (
                self._try_general_fallback(safe_question, resolved_language, len(chunks))
                or refusal
            )

        answer, sources, citation_status = self._validate_citations(answer, bundle)

        # A model can occasionally ignore the citation instruction. Give it one
        # explicit retry; never display an uncited answer or fabricated markers.
        if citation_status in {"not_provided", "unverified"}:
            logger.warning(
                "Citation validation failed (%s); retrying once", citation_status
            )
            retry_prompt = (
                user_prompt
                + "\n\nCITATION RETRY: Your previous answer did not contain at least "
                "one valid [S#] marker. Rewrite the same answer using only valid "
                "source IDs from the context and cite every factual paragraph."
            )
            try:
                retry_response = self.llm.generate(SYSTEM_PROMPT, retry_prompt)
            except LLMError as exc:
                logger.error(
                    "LLM retry failure provider=%s code=%s detail=%s",
                    self.provider_name,
                    exc.code,
                    exc.detail,
                )
                return _error_response(
                    LLM_ERROR_MESSAGES[resolved_language],
                    resolved_language,
                    "llm_error",
                    error_code=exc.code,
                )
            except Exception as exc:
                logger.exception("Unexpected LLM retry failure: %s", exc)
                return _error_response(
                    LLM_ERROR_MESSAGES[resolved_language],
                    resolved_language,
                    "llm_error",
                    error_code="unknown",
                )

            retry_answer = (retry_response.text or "").strip()
            if _is_not_found_answer(retry_answer):
                return {
                    "answer": NOT_FOUND_MESSAGES[resolved_language],
                    "language": resolved_language,
                    "sources": [],
                    "retrieval_status": "insufficient_context",
                    "citation_status": "not_applicable",
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "retrieved_chunks": len(chunks),
                }
            if retry_answer:
                answer, sources, citation_status = self._validate_citations(
                    retry_answer, bundle
                )

        if citation_status in {"not_provided", "unverified"}:
            return _error_response(
                CITATION_ERROR_MESSAGES[resolved_language],
                resolved_language,
                "citation_error",
                error_code="citation_missing",
            )

        return {
            "answer": answer,
            "language": resolved_language,
            "sources": sources,
            "retrieval_status": "success",
            "citation_status": citation_status,
            "provider": self.provider_name,
            "model": self.model_name,
            "retrieved_chunks": len(chunks),
        }

    def _validate_citations(
        self, answer: str, bundle: ContextBundle
    ) -> tuple[str, list[dict[str, Any]], str]:
        valid_ids = {
            int(source["source_id"][1:])
            for source in bundle.sources
            if str(source.get("source_id", "")).startswith("S")
        }
        cited_ids: set[int] = set()
        for match in _SOURCE_MARKER.finditer(answer):
            cited_ids.update(_marker_source_ids(match.group(1)))

        if not cited_ids:
            return answer, _dedupe_sources(bundle.sources), "not_provided"

        invalid_ids = cited_ids - valid_ids
        if invalid_ids:
            logger.warning("LLM referenced invalid source IDs: %s", sorted(invalid_ids))

            def replace_invalid(match: re.Match[str]) -> str:
                kept = [
                    source_id
                    for source_id in _marker_source_ids(match.group(1))
                    if source_id in valid_ids
                ]
                if not kept:
                    return ""
                return "[" + ", ".join(f"S{source_id}" for source_id in kept) + "]"

            answer = _SOURCE_MARKER.sub(replace_invalid, answer)
            answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
            cited_ids &= valid_ids

        selected = [
            source
            for source in bundle.sources
            if int(source["source_id"][1:]) in cited_ids
        ]
        if not selected:
            # The model produced markers that do not exist.  Do not display
            # fabricated citations; show the real retrieved sources instead and
            # mark the citation status as unverified.
            return answer, _dedupe_sources(bundle.sources), "unverified"

        return answer, _dedupe_sources(selected), "verified"


# ----------------------------------------------------------------------
# Convenience entry point required by the project specification
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_default_chain() -> RAGChain:
    from src.embeddings.embedder import get_embedding_service
    from src.llm.factory import create_llm_provider
    from src.vectorstore.chroma_store import ChromaStore

    settings = get_settings()
    settings.validate_llm_settings()
    settings.ensure_directories()
    store = ChromaStore(settings)
    embedder = get_embedding_service(
        settings.embedding_model,
        settings.embedding_batch_size,
        settings.embedding_device,
    )
    retriever = Retriever(settings, store, embedder)
    llm = create_llm_provider(settings)
    return RAGChain(settings, retriever, llm)


def reset_default_chain() -> None:
    get_default_chain.cache_clear()


def ask_question(question: str, language: str | None = None) -> dict[str, Any]:
    """Central entry point: question in, structured answer and citations out."""
    resolved_language = (
        language if language in {"mr", "hi", "en"} else detect_language(question or "")
    )
    try:
        chain = get_default_chain()
    except (SettingsError, LLMError) as exc:
        logger.error("RAG configuration error: %s", exc)
        return _error_response(
            LLM_ERROR_MESSAGES[resolved_language],
            resolved_language,
            "configuration_error",
            error_code=getattr(exc, "code", "configuration"),
        )
    except Exception as exc:
        logger.exception("Could not initialize RAG chain: %s", exc)
        return _error_response(
            LLM_ERROR_MESSAGES[resolved_language],
            resolved_language,
            "configuration_error",
            error_code="configuration",
        )
    return chain.ask_question(question, language=resolved_language)
