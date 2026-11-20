"""Guardrails for the optional general-LLM fallback.

This module keeps the fallback rules separate from the grounded RAG chain.
The fallback is intentionally conservative: it is blocked for questions that
need live, current, official, or personal information.
"""

from __future__ import annotations

import re

_CURRENT_MARKERS = re.compile(
    r"\b(?:today(?:'s)?|current|currently|latest|live|now|as\s+of)\b|"
    r"आज|सध्या|अद्ययावत|वर्तमान|ताज्या|आजचा|आजची|आजचे",
    re.IGNORECASE,
)

_HIGH_RISK_TOPIC = re.compile(
    r"\b(?:weather|forecast|temperature|rain(?:fall)?|market\s+price|"
    r"mandi|msp|minimum\s+support\s+price|deadline|last\s+date|"
    r"cut-?off|how\s+much|amount|subsidy|financial\s+support|"
    r"eligib\w*|price|prices|cost)\b|"
    r"पाऊस|हवामान|तापमान|बाजार|भाव|किंमती?|मंडी|एमएसपी|"
    r"अंतिम\s+मुदत|अंतिम\s+तारीख|कट-?ऑफ|किती|रक्कम|अनुदान|"
    r"पात्र|पात्रता|दर|मूल्य|"
    r"मौसम|बारिश|बाजार\s+भाव|न्यूनतम\s+समर्थन\s+मूल्य|"
    r"कितना|कितने|राशि|सब्सिडी|अंतिम\s+दिनांक",
    re.IGNORECASE,
)

_DIRECT_LIVE_REQUEST = re.compile(
    r"\b(?:weather\s+forecast|current\s+weather|market\s+price|"
    r"mandi\s+price|today'?s\s+price|live\s+price)\b|"
    r"हवामान\s+अंदाज|मौसम\s+पूर्वानुमान|बाजार\s+भाव|मंडी\s+भाव|आजचा\s+भाव",
    re.IGNORECASE,
)

_PERSONAL_DECISION = re.compile(
    r"\b(?:am\s+i\s+eligible|do\s+i\s+qualify|my\s+eligibility|"
    r"i\s+am\s+eligible)\b|"
    r"मी\s+पात्र|माझी\s+पात्रता|मला\s+पात्र|"
    r"क्या\s+मैं\s+पात्र|मेरी\s+पात्रता|मुझे\s+पात्र",
    re.IGNORECASE,
)

_CITATION_MARKER = re.compile(r"\[(?:S\d+(?:\s*,\s*S\d+)*)\]")
_URL = re.compile(r"https?://\S+")


def is_general_fallback_blocked(question: str) -> bool:
    """Return True when the fallback must not answer this question."""
    text = (question or "").strip()
    if not text:
        return True
    if _DIRECT_LIVE_REQUEST.search(text) or _PERSONAL_DECISION.search(text):
        return True
    return bool(_CURRENT_MARKERS.search(text) and _HIGH_RISK_TOPIC.search(text))


def clean_general_answer(answer: str) -> str:
    """Remove source markers and URLs from an ungrounded fallback answer."""
    text = _CITATION_MARKER.sub("", answer or "")
    text = _URL.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
