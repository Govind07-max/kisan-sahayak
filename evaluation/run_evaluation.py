"""Evaluate retrieval (no API key) and optionally live RAG answers.

Examples:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --top-k 5 --threshold 0.30
    python evaluation/run_evaluation.py --live --limit 5

Retrieval mode does not need an LLM key.  Live mode consumes Gemini quota and
requires GEMINI_API_KEY and GEMINI_MODEL.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SettingsError, get_settings  # noqa: E402
from src.embeddings.embedder import get_embedding_service  # noqa: E402
from src.retrieval.retriever import Retriever  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402
from src.vectorstore.chroma_store import ChromaStore  # noqa: E402

DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "questions.jsonl"
_CITATION_RE = re.compile(r"\[(?:S\d+\s*,\s*)*S\d+\]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation questions not found: {path}")
    questions: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            for field in ("id", "language", "question", "answerable"):
                if field not in item:
                    raise ValueError(f"Missing field '{field}' on line {line_number}")
            if item["language"] not in {"mr", "hi", "en"}:
                raise ValueError(f"Unsupported language on line {line_number}: {item['language']}")
            questions.append(item)
    return questions


def _validate_balance(questions: list[dict]) -> None:
    counts = {language: 0 for language in ("mr", "hi", "en")}
    ids: set[str] = set()
    for item in questions:
        counts[item["language"]] += 1
        if item["id"] in ids:
            raise ValueError(f"Duplicate evaluation ID: {item['id']}")
        ids.add(item["id"])
    if len(questions) != 30 or any(count != 10 for count in counts.values()):
        raise ValueError(
            f"Expected 30 questions with 10 per language; got {len(questions)} and {counts}"
        )


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _language_script_ok(answer: str, language: str) -> bool:
    has_devanagari = bool(_DEVANAGARI_RE.search(answer))
    if language in {"mr", "hi"}:
        return has_devanagari
    # For English, allow a small amount of Devanagari in a quoted scheme name,
    # but require mostly Latin text.
    devanagari = len(_DEVANAGARI_RE.findall(answer))
    return devanagari < max(10, len(answer) * 0.10)


def _keyword_coverage(answer: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    lowered = answer.lower()
    matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return matched / len(keywords)


def retrieval_evaluation(
    questions: list[dict],
    retriever: Retriever,
    top_k: int,
    threshold: float | None,
) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    answerable_scores: list[float] = []
    unsupported_scores: list[float] = []
    hits = 0
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    answerable_count = 0

    for item in questions:
        chunks = retriever.retrieve(
            item["question"],
            top_k=top_k,
            threshold=threshold,
        )
        retrieved = [
            {
                "document_id": chunk.metadata.get("document_id", ""),
                "document_title": chunk.metadata.get("document_title", ""),
                "page": chunk.metadata.get("page_number"),
                "page_end": chunk.metadata.get("page_end"),
                "score": round(chunk.score, 4),
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        ]
        expected = item.get("expected_doc_ids", [])
        record = {
            **item,
            "retrieved": retrieved,
            "top_score": round(chunks[0].score, 4) if chunks else None,
        }

        if item["answerable"]:
            answerable_count += 1
            rank = None
            for index, chunk in enumerate(chunks, start=1):
                if chunk.metadata.get("document_id") in expected:
                    rank = index
                    answerable_scores.append(chunk.score)
                    break
            if rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
                record["first_relevant_rank"] = rank
                record["relevant_score"] = round(chunks[rank - 1].score, 4)
            else:
                reciprocal_ranks.append(0.0)
                record["first_relevant_rank"] = None
                record["relevant_score"] = None

            retrieved_ids = {
                chunk.metadata.get("document_id") for chunk in chunks
            }
            recall = (
                len(retrieved_ids.intersection(expected)) / len(expected)
                if expected
                else 0.0
            )
            recalls.append(recall)
            record["expected_recall"] = round(recall, 4)
        else:
            if chunks:
                unsupported_scores.append(chunks[0].score)
            record["first_relevant_rank"] = None
            record["expected_recall"] = None

        records.append(record)

    by_language = {}
    for language in ("mr", "hi", "en"):
        language_records = [
            record for record in records if record["language"] == language
        ]
        language_answerable = [
            record for record in language_records if record["answerable"]
        ]
        language_hits = sum(
            1 for record in language_answerable if record["first_relevant_rank"]
        )
        by_language[language] = {
            "questions": len(language_records),
            "answerable": len(language_answerable),
            "hit_rate": round(language_hits / len(language_answerable), 4)
            if language_answerable
            else 0.0,
            "mean_top_score": _mean(
                [
                    record["top_score"]
                    for record in language_records
                    if record["top_score"] is not None
                ]
            ),
        }

    summary = {
        "mode": "retrieval",
        "questions": len(questions),
        "answerable_questions": answerable_count,
        "unsupported_questions": len(questions) - answerable_count,
        "top_k": top_k,
        "threshold": threshold,
        "hit_at_k": round(hits / answerable_count, 4) if answerable_count else 0.0,
        "mrr": _mean(reciprocal_ranks),
        "mean_expected_document_recall": _mean(recalls),
        "mean_relevant_score": _mean(answerable_scores),
        "mean_unsupported_top_score": _mean(unsupported_scores),
        "by_language": by_language,
    }
    return summary, records


def live_evaluation(
    questions: list[dict],
    settings,
    retriever: Retriever,
    top_k: int,
    threshold: float | None,
    limit: int | None,
) -> tuple[dict, list[dict]]:
    from src.llm.factory import create_llm_provider
    from src.rag.chain import RAGChain

    llm = create_llm_provider(settings)
    chain = RAGChain(settings, retriever, llm)
    selected = questions[:limit] if limit else questions
    records: list[dict] = []

    for item in selected:
        result = chain.ask_question(
            item["question"],
            language=item["language"],
            top_k=top_k,
            threshold=threshold,
        )
        sources = result.get("sources", [])
        source_doc_ids = {source.get("document_id") for source in sources}
        expected = set(item.get("expected_doc_ids", []))
        markers = _CITATION_RE.findall(result.get("answer", ""))
        keyword_coverage = _keyword_coverage(
            result.get("answer", ""), item.get("expected_keywords", [])
        )
        record = {
            "id": item["id"],
            "language": item["language"],
            "question": item["question"],
            "answerable": item["answerable"],
            "answer": result.get("answer", ""),
            "retrieval_status": result.get("retrieval_status"),
            "citation_status": result.get("citation_status"),
            "citation_count": len(markers),
            "source_doc_ids": sorted(doc for doc in source_doc_ids if doc),
            "expected_doc_ids": sorted(expected),
            "source_document_hit": bool(source_doc_ids.intersection(expected))
            if expected
            else None,
            "expected_keyword_coverage": round(keyword_coverage, 4)
            if keyword_coverage is not None
            else None,
            "language_script_ok": _language_script_ok(
                result.get("answer", ""), item["language"]
            ),
            "provider": result.get("provider"),
            "model": result.get("model"),
        }
        if item["answerable"]:
            record["correct_abstention"] = None
        else:
            record["correct_abstention"] = result.get("retrieval_status") in {
                "insufficient_context",
                "no_relevant_documents",
                "no_documents",
            }
        records.append(record)

    answerable = [record for record in records if record["answerable"]]
    unsupported = [record for record in records if not record["answerable"]]
    citation_valid = [
        record
        for record in answerable
        if record["citation_status"] == "verified" and record["citation_count"] > 0
    ]
    summary = {
        "mode": "live",
        "threshold": threshold,
        "questions": len(records),
        "answerable_questions": len(answerable),
        "unsupported_questions": len(unsupported),
        "successful_answers": sum(
            1 for record in answerable if record["retrieval_status"] == "success"
        ),
        "valid_citation_rate": round(len(citation_valid) / len(answerable), 4)
        if answerable
        else 0.0,
        "source_document_hit_rate": round(
            sum(1 for record in answerable if record["source_document_hit"])
            / len(answerable),
            4,
        )
        if answerable
        else 0.0,
        "language_script_ok_rate": round(
            sum(1 for record in records if record["language_script_ok"]) / len(records),
            4,
        )
        if records
        else 0.0,
        "correct_abstention_rate": round(
            sum(1 for record in unsupported if record["correct_abstention"])
            / len(unsupported),
            4,
        )
        if unsupported
        else 0.0,
        "mean_keyword_coverage": _mean(
            [
                record["expected_keyword_coverage"]
                for record in answerable
                if record["expected_keyword_coverage"] is not None
            ]
        ),
        "provider": llm.name,
        "model": llm.model,
    }
    return summary, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help=(
            "Relevance threshold for evaluation. Default: -1.0 (effectively no "
            "filter, so raw scores can be inspected)."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured LLM for answers (consumes API quota).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of live questions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the JSON result path.",
    )
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    _validate_balance(questions)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_file)
    settings.ensure_directories()

    # Retrieval mode defaults to -1.0 so raw scores are visible.  Live mode
    # must use the configured relevance threshold unless the caller explicitly
    # supplies another one, otherwise unsupported questions would reach the LLM.
    threshold = args.threshold
    if args.live and threshold == -1.0:
        threshold = settings.relevance_threshold

    if args.live:
        try:
            settings.validate_llm_settings()
        except SettingsError as exc:
            print(f"Live evaluation needs a configured LLM: {exc}", file=sys.stderr)
            return 2

    store = ChromaStore(settings)
    if store.count() == 0:
        print(
            "The ChromaDB collection is empty. Run `python ingest.py` first.",
            file=sys.stderr,
        )
        return 2
    embedder = get_embedding_service(
        settings.embedding_model,
        settings.embedding_batch_size,
        settings.embedding_device,
    )
    retriever = Retriever(settings, store, embedder)

    if args.live:
        summary, records = live_evaluation(
            questions,
            settings,
            retriever,
            args.top_k,
            threshold,
            args.limit,
        )
    else:
        summary, records = retrieval_evaluation(
            questions, retriever, args.top_k, args.threshold
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        PROJECT_ROOT
        / "evaluation"
        / "results"
        / f"{summary['mode']}_{timestamp}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "settings": settings.public_summary(),
                "summary": summary,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== EVALUATION SUMMARY ==========")
    for key, value in summary.items():
        if key != "by_language":
            print(f"{key}: {value}")
    print("\nBy language:")
    for language, values in summary.get("by_language", {}).items():
        print(f"  {language}: {values}")
    print(f"\nResult file: {output_path}")

    if summary["mode"] == "retrieval":
        print("\nTop scores by question (inspect values before setting RELEVANCE_THRESHOLD):")
        for record in records:
            print(
                f"  {record['id']:5s} {record['language']} "
                f"answerable={str(record['answerable']):5s} "
                f"top={record['top_score']} "
                f"relevant={record.get('relevant_score')} "
                f"rank={record.get('first_relevant_rank')}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
