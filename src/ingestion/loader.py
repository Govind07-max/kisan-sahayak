"""Document discovery and inventory handling."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import PROJECT_ROOT
from src.utils.document_utils import relative_to_project
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALID_CATEGORIES = {
    "central_schemes",
    "maharashtra_schemes",
    "crop_advisory",
    "soil_and_fertilizer",
    "weather_advisory",
}

VALID_OCR_MODES = {"auto", "force", "skip"}


@dataclass
class DocumentInfo:
    document_id: str
    document_title: str
    path: Path
    source_filename: str
    source_path: str
    category: str
    scheme_or_crop: str
    language: str
    publication_date: str
    validity_status: str
    source_url: str
    ocr_mode: str = "auto"
    notes: str = ""
    inventory_found: bool = True


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def load_inventory(inventory_path: Path) -> dict[str, dict[str, str]]:
    """Load the CSV inventory keyed by source filename."""
    if not inventory_path.exists():
        logger.warning("Document inventory not found: %s", inventory_path)
        return {}

    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    inventory: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = (row.get("source_filename") or "").strip()
        if not filename:
            continue
        inventory[filename] = row
    logger.info("Loaded inventory with %d rows from %s", len(inventory), inventory_path)
    return inventory


def _fallback_document(path: Path, document_dir: Path) -> DocumentInfo:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8].upper()
    parent = path.parent.name
    category = parent if parent in VALID_CATEGORIES else "uncategorized"
    return DocumentInfo(
        document_id=f"DOC-{digest}",
        document_title=path.stem.replace("_", " ").replace("-", " ").strip(),
        path=path,
        source_filename=path.name,
        source_path=relative_to_project(path, PROJECT_ROOT),
        category=category,
        scheme_or_crop="",
        language="unknown",
        publication_date="",
        validity_status="unknown",
        source_url="",
        ocr_mode="auto",
        notes="Automatically discovered; not present in document_inventory.csv",
        inventory_found=False,
    )


def discover_documents(
    document_dir: Path,
    inventory_path: Path,
    only: set[str] | None = None,
) -> list[DocumentInfo]:
    """Recursively find PDFs and enrich them with inventory metadata.

    Unsupported files are skipped with a warning.  Duplicate inventory rows and
    rows marked ``include=false`` are not indexed.
    """
    if not document_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {document_dir}")

    inventory = load_inventory(inventory_path)
    documents: list[DocumentInfo] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for path in sorted(document_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() != ".pdf":
            logger.warning("Skipping unsupported file (not a PDF): %s", path)
            continue

        row = inventory.get(path.name)
        if row is not None:
            if not _as_bool(row.get("include"), default=True):
                logger.info(
                    "Skipping %s because include=false (duplicate/not indexed)", path.name
                )
                continue
            if (row.get("duplicate_of") or "").strip():
                logger.info("Skipping duplicate file %s", path.name)
                continue

            document_id = (row.get("document_id") or "").strip() or path.stem
            if document_id in seen_ids:
                logger.warning(
                    "Duplicate document_id %s for %s; skipping", document_id, path.name
                )
                continue

            category = (row.get("category") or "").strip() or path.parent.name
            if category not in VALID_CATEGORIES:
                category = path.parent.name if path.parent.name in VALID_CATEGORIES else "uncategorized"

            ocr_mode = (row.get("ocr_mode") or "auto").strip().lower()
            if ocr_mode not in VALID_OCR_MODES:
                logger.warning(
                    "Invalid ocr_mode %r for %s; using auto", ocr_mode, path.name
                )
                ocr_mode = "auto"

            doc = DocumentInfo(
                document_id=document_id,
                document_title=(row.get("document_title") or "").strip()
                or path.stem.replace("_", " "),
                path=path,
                source_filename=path.name,
                source_path=relative_to_project(path, PROJECT_ROOT),
                category=category,
                scheme_or_crop=(row.get("scheme_or_crop") or "").strip(),
                language=(row.get("language") or "").strip() or "unknown",
                publication_date=(row.get("publication_date") or "").strip(),
                validity_status=(row.get("validity_status") or "").strip() or "unknown",
                source_url=(row.get("source_url") or "").strip(),
                ocr_mode=ocr_mode,
                notes=(row.get("notes") or "").strip(),
                inventory_found=True,
            )
        else:
            logger.warning(
                "No inventory row for %s; using filename-derived metadata", path.name
            )
            doc = _fallback_document(path, document_dir)
            if doc.document_id in seen_ids:
                logger.warning("Duplicate fallback ID %s; skipping", doc.document_id)
                continue

        if only:
            normalized = {item.lower() for item in only}
            if (
                doc.document_id.lower() not in normalized
                and doc.source_filename.lower() not in normalized
                and doc.path.stem.lower() not in normalized
            ):
                continue

        # Hash-level dedupe protects against two differently named copies that
        # are not marked as duplicates in the inventory.
        file_digest = hashlib.sha256(doc.path.read_bytes()).hexdigest()
        if file_digest in seen_hashes:
            logger.warning(
                "Skipping %s: identical file content already discovered", path.name
            )
            continue
        seen_hashes.add(file_digest)

        seen_ids.add(doc.document_id)
        documents.append(doc)

    logger.info("Discovered %d indexable PDF(s)", len(documents))
    return documents
