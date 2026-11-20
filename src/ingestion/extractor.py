"""Page-level PDF text extraction with OCR fallback."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image

from config.settings import Settings
from src.ingestion.ocr import OCRService, get_ocr_service
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PDFExtractionError(RuntimeError):
    """A PDF could not be opened or read."""


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    extraction_method: str
    char_count: int
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    path: Path
    page_count: int
    pages: list[ExtractedPage]
    metadata: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    @property
    def ocr_pages(self) -> list[int]:
        return [page.page_number for page in self.pages if page.ocr_used]

    @property
    def text_char_count(self) -> int:
        return sum(page.char_count for page in self.pages)

    @property
    def has_text(self) -> bool:
        return any(page.text.strip() for page in self.pages)


class PDFExtractor:
    def __init__(self, settings: Settings, ocr_service: OCRService | None = None) -> None:
        self.settings = settings
        self.ocr_service = ocr_service or get_ocr_service(
            settings.ocr_languages,
            settings.ocr_page_timeout_seconds,
            settings.tesseract_cmd,
        )

    def extract(self, path: Path, ocr_mode: str = "auto", language: str = "unknown") -> ExtractedDocument:
        path = Path(path)
        if not path.exists():
            raise PDFExtractionError(f"PDF does not exist: {path}")
        if not path.is_file():
            raise PDFExtractionError(f"Not a file: {path}")

        try:
            document = pymupdf.open(path)
        except Exception as exc:
            raise PDFExtractionError(f"Could not open PDF {path.name}: {exc}") from exc

        warnings: list[str] = []
        try:
            if document.needs_pass and not document.authenticate(""):
                raise PDFExtractionError(
                    f"PDF is password-protected and cannot be opened: {path.name}"
                )

            metadata: dict[str, str] = {}
            for key, value in (document.metadata or {}).items():
                if value:
                    metadata[str(key)] = str(value)

            pages: list[ExtractedPage] = []
            for page_index in range(document.page_count):
                page_number = page_index + 1
                try:
                    page = document.load_page(page_index)
                    text = page.get_text("text", sort=True) or ""
                    text = text.strip()
                    method = "text" if text else "empty"
                    page_warnings: list[str] = []
                    ocr_used = False

                    should_ocr = (
                        self.settings.ocr_enabled
                        and self.ocr_service.available
                        and (
                            ocr_mode == "force"
                            or (ocr_mode == "auto" and len(text) < self.settings.ocr_min_chars)
                        )
                    )

                    if should_ocr:
                        image = self._render_page(page)
                        ocr_result = self.ocr_service.image_to_text(
                            image, language_hint=language
                        )
                        page_warnings.extend(ocr_result.warnings)
                        if ocr_result.text.strip():
                            if ocr_mode == "force" or len(ocr_result.text) > len(text):
                                text = ocr_result.text.strip()
                                method = "ocr"
                                ocr_used = True
                            else:
                                page_warnings.append(
                                    "OCR produced less text than the PDF text layer; "
                                    "kept the PDF text layer."
                                )
                        elif ocr_mode == "force":
                            page_warnings.append(
                                "Forced OCR returned no text; kept the original PDF "
                                "text layer (which may be garbled)."
                            )
                        elif not text:
                            page_warnings.append("Page has no extractable text and OCR returned nothing.")
                    elif not text:
                        if not self.settings.ocr_enabled:
                            page_warnings.append("Page has no extractable text; OCR is disabled.")
                        elif not self.ocr_service.available:
                            page_warnings.append(
                                "Page has no extractable text; Tesseract OCR is unavailable."
                            )
                        else:
                            page_warnings.append("Page has no extractable text.")

                    pages.append(
                        ExtractedPage(
                            page_number=page_number,
                            text=text,
                            extraction_method=method,
                            char_count=len(text),
                            ocr_used=ocr_used,
                            warnings=page_warnings,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not extract page %d from %s: %s", page_number, path.name, exc
                    )
                    pages.append(
                        ExtractedPage(
                            page_number=page_number,
                            text="",
                            extraction_method="error",
                            char_count=0,
                            warnings=[f"Page extraction failed: {exc}"],
                        )
                    )
                    warnings.append(f"Page {page_number} extraction failed: {exc}")

            if not pages:
                warnings.append("PDF contains no pages.")
            if not any(page.text.strip() for page in pages):
                warnings.append("PDF contains no extractable text after OCR fallback.")

            if ocr_mode == "force" and not self.ocr_service.available:
                warnings.append(
                    "This document needs OCR, but Tesseract is unavailable; its "
                    "text may be garbled or empty."
                )

            return ExtractedDocument(
                path=path,
                page_count=document.page_count,
                pages=pages,
                metadata=metadata,
                warnings=warnings,
            )
        finally:
            document.close()

    def _render_page(self, page: pymupdf.Page) -> Image.Image:
        pixmap = page.get_pixmap(dpi=self.settings.ocr_dpi, alpha=False)
        return Image.open(io.BytesIO(pixmap.tobytes("png")))
