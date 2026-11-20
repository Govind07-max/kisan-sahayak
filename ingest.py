"""Ingestion CLI.

Examples:
    python ingest.py
    python ingest.py --force
    python ingest.py --document GRD-001
    python ingest.py --stats
    python ingest.py --ocr-health
"""

from __future__ import annotations

import argparse
import json
import sys

from config.settings import SettingsError, get_settings
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest government PDFs into the persistent ChromaDB collection."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess documents even when their SHA-256 hash is unchanged.",
    )
    parser.add_argument(
        "--document",
        action="append",
        default=None,
        help="Limit ingestion to a document ID or filename. Repeat for multiple documents.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Disable OCR for this run (not recommended for GRD-008/GRD-009).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print collection statistics and exit.",
    )
    parser.add_argument(
        "--ocr-health",
        action="store_true",
        help="Print Tesseract availability/languages and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models available to the configured LLM provider and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_file)
    settings.ensure_directories()

    if args.skip_ocr:
        settings = settings.model_copy(update={"ocr_enabled": False})

    if args.ocr_health:
        from src.ingestion.ocr import get_ocr_service

        service = get_ocr_service(
            settings.ocr_languages,
            settings.ocr_page_timeout_seconds,
            settings.tesseract_cmd,
        )
        print(json.dumps(service.health(), indent=2, ensure_ascii=False))
        return 0

    if args.list_models:
        from src.llm.factory import create_llm_provider
        from src.llm.base import LLMError

        try:
            provider = create_llm_provider(settings, require_model=False)
            models = provider.list_models()
        except (SettingsError, LLMError) as exc:
            logger.error("Could not list models: %s", exc)
            return 1
        print(f"Provider: {settings.llm_provider}")
        for name in models:
            print(name)
        return 0

    if args.stats:
        from src.vectorstore.chroma_store import ChromaStore

        store = ChromaStore(settings)
        print(json.dumps(store.statistics(), indent=2, ensure_ascii=False))
        return 0

    from src.ingestion.pipeline import IngestionPipeline

    only = set(args.document) if args.document else None
    pipeline = IngestionPipeline(settings)
    summary = pipeline.run(force=args.force, only=only)

    print("\n========== INGESTION SUMMARY ==========")
    print(f"Discovered:          {summary.discovered}")
    print(f"Processed:           {summary.succeeded}")
    print(f"Skipped unchanged:   {summary.skipped_unchanged}")
    print(f"Failed:              {summary.failed}")
    print(f"Chunks indexed:      {summary.chunks_added}")
    print(f"OCR pages:           {summary.ocr_pages}")
    print(f"Collection count:    {summary.collection_count}")
    if summary.failures:
        print("\nFailures:")
        for failure in summary.failures:
            print(f"  - {failure['document_id']}: {failure['error']}")
    print("=======================================")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
