"""Small, transparent multilingual retrieval-name expansion.

The corpus is mostly English, while farmers ask questions in Marathi and Hindi.
The multilingual embedding model handles cross-lingual meaning, but explicit
scheme names and section-level intents are still easy to miss when they are
written in a different script or as a local abbreviation.  These expansions add
only scheme/topic names and generic procedure headings to the retrieval query;
they never add answer facts, amounts, eligibility rules, or dates.
"""

from __future__ import annotations

import re

# (trigger phrases, English retrieval terms)
_ALIAS_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "पीएम किसान",
            "पीएम-किसान",
            "पंतप्रधान किसान सन्मान",
            "प्रधानमंत्री किसान सम्मान",
            "pm-kisan",
            "pm kisan",
            "pmkisan",
        ),
        "PM-KISAN Pradhan Mantri Kisan Samman Nidhi farmer income support scheme",
    ),
    (
        (
            "पीक विमा",
            "पंतप्रधान पीक विमा",
            "फसल बीमा",
            "प्रधानमंत्री फसल बीमा",
            "pmfby",
            "fasal bima",
        ),
        "PMFBY Pradhan Mantri Fasal Bima Yojana crop insurance",
    ),
    (
        (
            "मृदा आरोग्य",
            "मृदा स्वास्थ्य",
            "माती परीक्षण",
            "मिट्टी परीक्षण",
            "soil health",
            "soil testing",
        ),
        "Soil Health Card soil sampling and testing",
    ),
    (
        ("किसान क्रेडिट", "kisan credit", "kcc"),
        "Kisan Credit Card KCC farmer credit",
    ),
    (
        ("कापूस", "कपास", "cotton"),
        "cotton integrated pest management IPM",
    ),
    (
        (
            "महा-डीबीटी",
            "महाडीबीटी",
            "महा डीबीटी",
            "maha-dbt",
            "mahadbt",
            "maha dbt",
        ),
        "Maha-DBT MahaDBT Farmer Application Portal registration Farmer ID OTP",
    ),
    (
        (
            "महाॲग्री",
            "महा एआय",
            "महाएआय",
            "mahaagri",
            "maha-agri",
            "maha agri",
        ),
        "MahaAgri-AI Policy 2025-2029 artificial intelligence",
    ),
    (
        (
            "कृषि अवसंरचना",
            "कृषी अवसंरचना",
            "agriculture infrastructure",
            "aif",
        ),
        "Agriculture Infrastructure Fund AIF interest subvention",
    ),
)

# (scheme/topic triggers, intent triggers, English retrieval hint)
# These hints name document sections or generic concepts (for example,
# "cultural practices" or "claim intimation").  They deliberately contain no
# answer facts such as amounts, deadlines, or eligibility thresholds.
_INTENT_RULES: tuple[
    tuple[tuple[str, ...], tuple[str, ...], str], ...
] = (
    (
        ("कापूस", "कपास", "cotton"),
        ("कीड", "कीट", "pest", "ipm", "एकात्मिक", "integrated"),
        "cotton integrated pest management cultural practices mechanical "
        "practices biocontrol practices recommended methods",
    ),
    (
        (
            "पीक विमा",
            "पंतप्रधान पीक विमा",
            "फसल बीमा",
            "प्रधानमंत्री फसल बीमा",
            "pmfby",
            "fasal bima",
            "crop insurance",
        ),
        ("दावा", "claim", "नुकसान", "loss", "मुआवजा", "मुआवज़ा"),
        "PMFBY crop insurance claim intimation procedure documents loss "
        "assessment farmer",
    ),
    (
        (
            "कृषि अवसंरचना",
            "कृषी अवसंरचना",
            "agriculture infrastructure",
            "aif",
        ),
        ("ब्याज", "interest", "सब्सिडी", "subvention"),
        "Agriculture Infrastructure Fund AIF interest subvention norms "
        "maximum loan period",
    ),
    (
        (
            "महाॲग्री",
            "महा एआय",
            "महाएआय",
            "mahaagri",
            "maha-agri",
            "maha agri",
        ),
        ("उद्दिष्ट", "उद्दिष्टे", "objective", "ध्येय", "vision", "कालावधी", "period"),
        "MahaAgri AI Policy 2025-2029 vision objectives mission implementation "
        "period",
    ),
)


def _has_latin_trigger(question: str, trigger: str) -> bool:
    """Avoid matching short Latin abbreviations inside unrelated words."""
    if re.fullmatch(r"[a-z0-9-]+", trigger):
        return bool(re.search(rf"\b{re.escape(trigger)}\b", question))
    return trigger in question


def expand_query(question: str) -> str:
    """Append English scheme/topic names when a known local alias is present."""
    if not question or not question.strip():
        return question
    lowered = question.casefold()
    extras: list[str] = []
    for triggers, expansion in _ALIAS_RULES:
        if any(_has_latin_trigger(lowered, trigger) for trigger in triggers):
            if expansion not in extras:
                extras.append(expansion)
    if not extras:
        return question
    return (
        question.rstrip()
        + "\nRelevant scheme/topic names for retrieval: "
        + "; ".join(extras)
    )


def build_retrieval_queries(question: str) -> list[str]:
    """Return the primary query plus any matching intent-level hints.

    The first item is always the normal scheme-expanded question.  Additional
    English hints are used as separate retrieval queries, so a Marathi or Hindi
    question can still reach an English procedure section without diluting the
    original query.  The retriever fuses the best results from each query.
    """
    if not question or not question.strip():
        return []
    primary = expand_query(question)
    queries = [primary]
    lowered = question.casefold()
    for scheme_triggers, intent_triggers, hint in _INTENT_RULES:
        if not any(
            _has_latin_trigger(lowered, trigger) for trigger in scheme_triggers
        ):
            continue
        if not any(
            _has_latin_trigger(lowered, trigger) for trigger in intent_triggers
        ):
            continue
        if hint.casefold() not in primary.casefold() and hint not in queries:
            queries.append(hint)
    return queries
