"""Incremental ingestion pipeline tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pymupdf

from config.settings import Settings
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.chroma_store import ChromaStore


def _write_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=11)
    document.save(path)
    document.close()


def _inventory(settings: Settings, rows: list[dict[str, str]]) -> None:
    settings.inventory_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "document_id",
        "document_title",
        "source_filename",
        "source_path",
        "category",
        "scheme_or_crop",
        "language",
        "publication_date",
        "validity_status",
        "source_url",
        "pages",
        "sha256",
        "duplicate_of",
        "include",
        "ocr_mode",
        "notes",
    ]
    with settings.inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _row(document_id: str, filename: str, category: str = "central_schemes"):
    return {
        "document_id": document_id,
        "document_title": f"Document {document_id}",
        "source_filename": filename,
        "source_path": f"data/raw/{category}/{filename}",
        "category": category,
        "scheme_or_crop": "PM-KISAN",
        "language": "en",
        "publication_date": "",
        "validity_status": "undated",
        "source_url": "",
        "include": "true",
        "ocr_mode": "auto",
    }


def test_incremental_ingestion_and_manifest(settings: Settings, fake_embedder):
    settings.document_dir.mkdir(parents=True, exist_ok=True)
    first = settings.document_dir / "central_schemes" / "a.pdf"
    second = settings.document_dir / "central_schemes" / "b.pdf"
    _write_pdf(
        first,
        "PM-KISAN provides income support. " * 40,
    )
    _write_pdf(
        second,
        "PMFBY provides crop insurance to farmers. " * 40,
    )
    _inventory(
        settings,
        [
            _row("DOC-A", "a.pdf"),
            _row("DOC-B", "b.pdf"),
        ],
    )
    store = ChromaStore(settings)
    pipeline = IngestionPipeline(settings, store=store, embedder=fake_embedder)

    summary = pipeline.run()
    assert summary.succeeded == 2
    assert summary.failed == 0
    assert summary.collection_count > 0
    first_count = summary.collection_count
    assert settings.manifest_path.exists()

    summary_again = pipeline.run()
    assert summary_again.skipped_unchanged == 2
    assert summary_again.chunks_added == 0
    assert summary_again.collection_count == first_count

    # Change one document and make sure its old chunks are replaced, not
    # duplicated.  The file hash changes, so the pipeline must reprocess it.
    _write_pdf(first, "PM-KISAN revised information for farmers. " * 40)
    summary_changed = pipeline.run()
    assert summary_changed.succeeded == 1
    assert summary_changed.skipped_unchanged == 1
    assert summary_changed.collection_count == first_count

    manifest = __import__("json").loads(settings.manifest_path.read_text())
    assert manifest["documents"]["DOC-A"]["status"] == "success"
    assert manifest["documents"]["DOC-A"]["chunks"] >= 1
    assert len(store.ids_for_document("DOC-A")) == manifest["documents"]["DOC-A"]["chunks"]


def test_no_documents_is_safe(settings: Settings, fake_embedder):
    settings.document_dir.mkdir(parents=True, exist_ok=True)
    store = ChromaStore(settings)
    pipeline = IngestionPipeline(settings, store=store, embedder=fake_embedder)
    summary = pipeline.run()
    assert summary.discovered == 0
    assert summary.failed == 0
    assert summary.collection_count == 0
