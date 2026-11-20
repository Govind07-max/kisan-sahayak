"""Kisan Sahayak Streamlit application."""

from __future__ import annotations

import streamlit as st

from config.prompts import (
    APP_NAME,
    APP_SUBTITLE,
    GENERAL_FALLBACK_NOTICES,
    SUGGESTED_QUESTIONS,
    UI_LANGUAGES,
)
from config.settings import SettingsError, get_settings
from src.utils.logging import configure_logging, redact_secrets

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded",
)

_styles = """
<style>
    .block-container {max-width: 900px; padding-top: 2rem;}
    .kisan-subtitle {color: #4b5563; font-size: 1.05rem; margin-top: -0.8rem;}
    .kisan-source {border-left: 3px solid #2e7d32; padding-left: 0.8rem; margin: 0.6rem 0;}
    .kisan-note {font-size: 0.85rem; color: #6b7280;}
    [data-testid="stChatMessage"] {border-radius: 0.8rem;}
</style>
"""
st.markdown(_styles, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_service():
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_file)
    try:
        from src.services.chat_service import create_chat_service

        return create_chat_service(settings), None
    except SettingsError as exc:
        return None, redact_secrets(str(exc))
    except Exception as exc:
        return None, redact_secrets(f"Application initialization failed: {exc}")


settings = get_settings()
configure_logging(settings.log_level, settings.log_file)
service, service_error = load_service()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "mr"
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def add_message(
    role: str,
    content: str,
    sources: list[dict] | None = None,
    status: str | None = None,
    citation_status: str | None = None,
    is_general_fallback: bool = False,
    fallback_notice: str | None = None,
) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
            "status": status,
            "citation_status": citation_status,
            "is_general_fallback": is_general_fallback,
            "fallback_notice": fallback_notice,
        }
    )


def handle_question(question: str) -> None:
    question = question.strip()
    if not question:
        return
    language = st.session_state.language
    add_message("user", question)
    if service is None:
        add_message(
            "assistant",
            "The chatbot is not configured yet.\n\n"
            f"**{service_error or 'Unknown configuration error'}**\n\n"
            "Add `GEMINI_API_KEY` and `GEMINI_MODEL` to `.env`, then restart the app.",
            status="configuration_error",
        )
        st.session_state.pending_question = None
        st.rerun()

    with st.spinner("Searching government documents and preparing an answer..."):
        result = service.ask(question, language=language)
    add_message(
        "assistant",
        result.get("answer", ""),
        sources=result.get("sources", []),
        status=result.get("retrieval_status"),
        citation_status=result.get("citation_status"),
        is_general_fallback=bool(result.get("is_general_fallback")),
        fallback_notice=result.get("fallback_notice"),
    )
    st.session_state.pending_question = None
    st.rerun()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Supporting sources ({len(sources)})"):
        for source in sources:
            source_id = source.get("source_id", "")
            title = source.get("document_title") or source.get("source", "Source")
            page = source.get("page", "?")
            page_end = source.get("page_end", page)
            page_label = str(page) if str(page) == str(page_end) else f"{page}–{page_end}"
            st.markdown(f"**[{source_id}] {title}**")
            details = [f"File: `{source.get('source', '')}`", f"Page: {page_label}"]
            if source.get("category"):
                details.append(f"Category: {source['category']}")
            if source.get("validity_status"):
                details.append(f"Validity: {source['validity_status']}")
            st.caption(" · ".join(details))
            source_url = (source.get("source_url") or "").strip()
            if source_url:
                st.markdown(f"[Open official source]({source_url})")
            st.divider()


def render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                status = message.get("status")
                if message.get("is_general_fallback"):
                    notice = message.get("fallback_notice") or GENERAL_FALLBACK_NOTICES.get(
                        st.session_state.language, GENERAL_FALLBACK_NOTICES["en"]
                    )
                    st.warning(notice)
                elif status == "insufficient_context":
                    st.caption("The documents did not contain enough verified detail for a confident answer.")
                elif status in {"no_documents", "no_relevant_documents"}:
                    st.caption("No sufficiently relevant document was found for this question.")
                elif status == "llm_error":
                    st.caption("The answer service failed; no answer was generated from the documents.")
                citation_status = None if message.get("is_general_fallback") else message.get("citation_status")
                if citation_status == "unverified":
                    st.caption(
                        "The model's inline citation markers could not be matched; "
                        "the retrieved sources are shown for verification."
                    )
                elif citation_status == "not_provided":
                    st.caption(
                        "The answer did not include inline citations; the retrieved "
                        "sources are shown for verification."
                    )
                render_sources(message.get("sources", []))


st.title(f"🌾 {APP_NAME}")
st.markdown(
    f'<div class="kisan-subtitle">{APP_SUBTITLE}</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Document-based answers are preferred. If unverified general guidance is enabled, "
    "it is clearly labeled. Always verify scheme details, deadlines, and benefits "
    "with the official department."
)

with st.sidebar:
    st.header(APP_NAME)
    selected_label = st.radio(
        "Language / भाषा",
        options=list(UI_LANGUAGES.values()),
        index=list(UI_LANGUAGES.keys()).index(st.session_state.language),
    )
    for code, label in UI_LANGUAGES.items():
        if label == selected_label:
            if st.session_state.language != code:
                st.session_state.language = code
            break

    st.divider()
    if service is not None:
        status = service.status()
        st.metric("Indexed chunks", status["chunks"])
        st.caption(f"Collection: `{status['collection']}`")
        st.caption(f"LLM: `{status['provider']}` / `{status['model']}`")
        st.caption(f"Embedding: `{status['embedding_model']}`")
        st.caption(
            "General fallback: "
            + ("enabled" if status.get("enable_general_llm_fallback") else "disabled")
        )
    else:
        st.error("Chat service is not configured.")
        st.caption(service_error or "Check your .env configuration.")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    with st.expander("About this assistant"):
        st.write(
            "Kisan Sahayak is a retrieval-augmented assistant for Central and "
            "Maharashtra government schemes, crop advisories, soil health, and "
            "irrigation. It supports Marathi, Hindi, and English."
        )
        st.write(
            "It does not provide live weather, market prices, or personalized "
            "financial advice. It does not ask for Aadhaar or bank details."
        )

if service is None:
    st.error(
        "The chatbot cannot generate answers until it is configured.\n\n"
        f"**{service_error or 'Unknown configuration error'}**\n\n"
        "Create a `.env` file from `.env.example`, add your Gemini API key and a "
        "currently available model ID, then restart Streamlit."
    )

with st.expander("💡 Suggested questions", expanded=not st.session_state.messages):
    for question in SUGGESTED_QUESTIONS[st.session_state.language]:
        if st.button(question, key=f"suggested-{st.session_state.language}-{question}"):
            st.session_state.pending_question = question
            st.rerun()

render_messages()

if st.session_state.pending_question:
    handle_question(st.session_state.pending_question)

prompt = st.chat_input("Type your question in Marathi, Hindi, or English...")
if prompt:
    handle_question(prompt)
