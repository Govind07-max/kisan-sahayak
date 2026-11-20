"""Incremental ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.settings import Settings
from src.embeddings.embedder import EmbeddingService, get_embedding_service
from src.ingestion.chunker import Chunk, TextChunker
from src.ingestion.cleaner import clean_pages
from src.ingestion.extractor import ExtractedDocument, PDFExtractor
from src.ingestion.loader import DocumentInfo, discover_documents
from src.utils.document_utils import (
    read_json,
    sha256_file,
    stable_chunk_id,
    write_json_atomic,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

MANIFEST_VERSION = 1


class IngestionError(RuntimeError):
    """A document could not be ingested."""


@dataclass
class IngestionSummary:
    discovered: int = 0
    skipped_unchanged: int = 0
    succeeded: int = 0
    failed: int = 0
    chunks_added: int = 0
    ocr_pages: int = 0
    collection_count: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    skipped_documents: list[str] = field(default_factory=list)
    processed_documents: list[str] = field(default_factory=list)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        store=None,
        embedder: EmbeddingService | None = None,
        extractor: PDFExtractor | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.store = store or self._build_store()
        self.embedder = embedder or get_embedding_service(
            settings.embedding_model,
            settings.embedding_batch_size,
            settings.embedding_device,
        )
        self.extractor = extractor or PDFExtractor(settings)
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def _build_store(self):
        from src.vectorstore.chroma_store import ChromaStore

        return ChromaStore(self.settings)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def _load_manifest(self) -> dict:
        manifest = read_json(
            self.settings.manifest_path,
            {"version": MANIFEST_VERSION, "documents": {}},
        )
        if not isinstance(manifest, dict):
            manifest = {"version": MANIFEST_VERSION, "documents": {}}
        manifest.setdefault("version", MANIFEST_VERSION)
        manifest.setdefault("documents", {})
        return manifest

    def _save_manifest(self, manifest: dict) -> None:
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(self.settings.manifest_path, manifest)

    def _is_unchanged(self, document: DocumentInfo, file_hash: str, manifest: dict) -> bool:
        previous = manifest["documents"].get(document.document_id)
        return bool(
            previous
            and previous.get("status") == "success"
            and previous.get("sha256") == file_hash
        )

    # ------------------------------------------------------------------
    # Document processing
    # ------------------------------------------------------------------
    def _build_metadatas(
        self, document: DocumentInfo, chunks: list[Chunk]
    ) -> list[dict[str, object]]:
        metadatas: list[dict[str, object]] = []
        for chunk in chunks:
            metadatas.append(
                {
                    "document_id": document.document_id,
                    "document_title": document.document_title,
                    "source_filename": document.source_filename,
                    "source_path": document.source_path,
                    "source_url": document.source_url,
                    "page_number": chunk.page_start,
                    "page_end": chunk.page_end,
                    "category": document.category,
                    "scheme_or_crop": document.scheme_or_crop,
                    "language": document.language,
                    "publication_date": document.publication_date,
                    "validity_status": document.validity_status,
                    "chunk_index": chunk.chunk_index,
                }
            )
        return metadatas

    def _store_chunks(
        self,
        document: DocumentInfo,
        extracted: ExtractedDocument,
        chunks: list[Chunk],
        existing_ids: list[str],
    ) -> int:
        if not chunks:
            raise IngestionError("No non-empty chunks were produced.")

        texts = [chunk.text for chunk in chunks]
        # The short title prefix makes scheme and crop names searchable even
        # when a chunk starts in the middle of a long procedure.  The stored
        # document text remains the original chunk so the LLM sees natural
        # citations and context.
        embedding_texts = [
            (
                f"Document: {document.document_title}\n"
                f"Scheme or crop: {document.scheme_or_crop}\n\n{text}"
            )
            for text in texts
        ]
        embeddings = self.embedder.encode_documents(embedding_texts)
        if len(embeddings) != len(chunks):
            raise IngestionError(
                f"Embedding count mismatch: {len(embeddings)} vectors for {len(chunks)} chunks."
            )

        metadatas = self._build_metadatas(document, chunks)
        ids: list[str] = []
        selected_texts: list[str] = []
        selected_embeddings: list[list[float]] = []
        selected_metadatas: list[dict[str, object]] = []
        seen: set[str] = set()
        for chunk, text, embedding, metadata in zip(
            chunks, texts, embeddings, metadatas
        ):
            chunk_id = stable_chunk_id(
                document.document_id, chunk.page_start, chunk.chunk_index, text
            )
            if chunk_id in seen:
                logger.warning(
                    "Duplicate chunk ID %s in %s; skipping duplicate",
                    chunk_id,
                    document.document_id,
                )
                continue
            seen.add(chunk_id)
            ids.append(chunk_id)
            selected_texts.append(text)
            selected_embeddings.append(list(embedding))
            selected_metadatas.append(metadata)

        if not ids:
            raise IngestionError("All generated chunks were duplicates.")

        # Add/update new chunks first, then remove stale chunks.  If the upsert
        # fails, the previous version remains in the collection.
        self.store.upsert_chunks(
            ids=ids,
            documents=selected_texts,
            embeddings=selected_embeddings,
            metadatas=selected_metadatas,
        )
        stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in set(ids)]
        if stale_ids:
            self.store.delete_ids(stale_ids)
            logger.info(
                "Deleted %d stale chunk(s) for %s", len(stale_ids), document.document_id
            )

        logger.info(
            "Indexed %s: %d chunks, pages %d, OCR pages %d",
            document.document_id,
            len(ids),
            extracted.page_count,
            len(extracted.ocr_pages),
        )
        return len(ids)

    def run(self, force: bool = False, only: set[str] | None = None) -> IngestionSummary:
        documents = discover_documents(
            self.settings.document_dir,
            self.settings.inventory_path,
            only=only,
        )
        summary = IngestionSummary(discovered=len(documents))
        manifest = self._load_manifest()

        if not documents:
            logger.warning("No indexable PDFs found in %s", self.settings.document_dir)
            summary.collection_count = self.store.count()
            return summary

        for document in documents:
            logger.info("Processing %s", document.source_filename)
            file_hash = sha256_file(document.path)
            if not force and self._is_unchanged(document, file_hash, manifest):
                logger.info("Skipping unchanged document %s", document.document_id)
                summary.skipped_unchanged += 1
                summary.skipped_documents.append(document.document_id)
                continue

            existing_ids = self.store.ids_for_document(document.document_id)
            try:
                extracted = self.extractor.extract(
                    document.path,
                    ocr_mode=document.ocr_mode,
                    language=document.language,
                )
                cleaned_pages = clean_pages(extracted.pages)
                if not any(page.text.strip() for page in cleaned_pages):
                    raise IngestionError(
                        "No extractable text found. OCR may be unavailable or the "
                        "PDF may contain only images."
                    )
                chunks = self.chunker.chunk_pages(cleaned_pages)
                if not chunks:
                    raise IngestionError("No chunks were produced from extracted text.")

                chunk_count = self._store_chunks(
                    document, extracted, chunks, existing_ids
                )

                manifest["documents"][document.document_id] = {
                    "document_title": document.document_title,
                    "source_filename": document.source_filename,
                    "source_path": document.source_path,
                    "category": document.category,
                    "language": document.language,
                    "sha256": file_hash,
                    "last_processed": datetime.now(timezone.utc).isoformat(),
                    "status": "success",
                    "pages": extracted.page_count,
                    "chunks": chunk_count,
                    "ocr_pages": extracted.ocr_pages,
                    "text_chars": extracted.text_char_count,
                    "warnings": extracted.warnings[-20:],
                    "errors": [],
                }
                summary.succeeded += 1
                summary.chunks_added += chunk_count
                summary.ocr_pages += len(extracted.ocr_pages)
                summary.processed_documents.append(document.document_id)
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed to ingest %s: %s", document.document_id, exc)
                manifest["documents"][document.document_id] = {
                    "document_title": document.document_title,
                    "source_filename": document.source_filename,
                    "source_path": document.source_path,
                    "category": document.category,
                    "language": document.language,
                    "sha256": file_hash,
                    "last_processed": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "pages": None,
                    "chunks": 0,
                    "ocr_pages": [],
                    "text_chars": 0,
                    "warnings": [],
                    "errors": [error_message],
                }
                summary.failed += 1
                summary.failures.append(
                    {
                        "document_id": document.document_id,
                        "source_filename": document.source_filename,
                        "error": error_message,
                    }
                )
            finally:
                self._save_manifest(manifest)

        summary.collection_count = self.store.count()
        return summary
