"""PDF extraction tests, including scanned-page OCR."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from src.ingestion.extractor import PDFExtractionError, PDFExtractor
from src.ingestion.ocr import get_ocr_service


def test_extract_multiple_pages(settings: Settings, make_pdf):
    path = make_pdf(
        "multipage.pdf",
        [
            "Page one: PM-KISAN provides income support to eligible farmers.",
            "Page two: PMFBY is a crop insurance scheme.",
            "Page three: Soil Health Card provides soil testing information.",
        ],
    )
    extractor = PDFExtractor(settings)
    result = extractor.extract(path)
    assert result.page_count == 3
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert "PM-KISAN" in result.pages[0].text
    assert "crop insurance" in result.pages[1].text
    assert result.ocr_pages == []


def test_invalid_pdf_raises(settings: Settings, tmp_path: Path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a PDF")
    extractor = PDFExtractor(settings)
    with pytest.raises(PDFExtractionError):
        extractor.extract(bad)


def test_empty_pdf_is_handled(settings: Settings, make_pdf):
    path = make_pdf("blank.pdf", [""])
    extractor = PDFExtractor(settings)
    result = extractor.extract(path)
    assert result.page_count == 1
    assert not result.has_text
    assert result.pages[0].text == ""
    assert result.pages[0].warnings


def test_scanned_pdf_uses_ocr(settings: Settings, tmp_path: Path):
    ocr = get_ocr_service(
        settings.ocr_languages,
        settings.ocr_page_timeout_seconds,
        settings.tesseract_cmd,
    )
    if not ocr.available or "eng" not in ocr.available_languages:
        pytest.skip("Tesseract OCR is not available in this environment.")

    import pymupdf
    from PIL import Image

    source = pymupdf.open()
    page = source.new_page()
    page.insert_textbox(
        pymupdf.Rect(80, 80, 520, 200),
        "PM-KISAN OCR TEST 2026",
        fontsize=22,
        fontname="helv",
    )
    pixmap = page.get_pixmap(dpi=160)
    image_path = tmp_path / "scan.png"
    pixmap.save(image_path)
    source.close()

    image_pdf = pymupdf.open()
    image_page = image_pdf.new_page(width=pixmap.width, height=pixmap.height)
    image_page.insert_image(image_page.rect, filename=str(image_path))
    scanned_path = tmp_path / "scanned.pdf"
    image_pdf.save(scanned_path)
    image_pdf.close()

    extractor = PDFExtractor(settings, ocr_service=ocr)
    extract_settings = settings.model_copy(update={"ocr_enabled": True})
    extractor = PDFExtractor(extract_settings, ocr_service=ocr)
    result = extractor.extract(scanned_path, language="en")
    assert result.page_count == 1
    assert result.pages[0].ocr_used is True
    assert "KISAN" in result.pages[0].text.upper()


def test_page_metadata_is_preserved(settings: Settings, make_pdf):
    path = make_pdf("meta.pdf", ["First page text", "Second page text"])
    result = PDFExtractor(settings).extract(path)
    assert result.pages[1].page_number == 2
    assert result.pages[1].extraction_method == "text"
