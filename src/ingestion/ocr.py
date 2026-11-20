"""Tesseract OCR support with graceful degradation.

The project can use a system Tesseract installation, or the self-contained
copy under ``.local/tesseract`` that was extracted from official Ubuntu packages.
No document is sent to a cloud OCR service.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

from config.settings import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)

BUNDLED_TESSERACT_DIR = PROJECT_ROOT / ".local" / "tesseract"


@dataclass
class OCRResult:
    text: str
    languages: str
    warnings: list[str] = field(default_factory=list)
    used_ocr: bool = False


class OCRUnavailableError(RuntimeError):
    """Raised when Tesseract or its language data cannot be used."""


class OCRService:
    def __init__(
        self,
        languages: str = "mar+hin+eng",
        page_timeout_seconds: int = 120,
        tesseract_cmd: str = "",
    ) -> None:
        self.requested_languages = [
            item for item in languages.split("+") if item.strip()
        ]
        self.page_timeout_seconds = page_timeout_seconds
        self.tesseract_cmd = self._resolve_binary(tesseract_cmd)
        self.tessdata_dir = self._resolve_tessdata()
        self._configure_environment()
        self.available = self._check_available()
        self.available_languages = self._read_languages() if self.available else []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _resolve_binary(self, configured: str) -> str:
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.exists():
                return str(candidate)
            logger.warning("Configured TESSERACT_CMD does not exist: %s", configured)

        bundled = BUNDLED_TESSERACT_DIR / "usr" / "bin" / "tesseract"
        if bundled.exists():
            return str(bundled)

        system = shutil.which("tesseract")
        if system:
            return system
        return ""

    def _resolve_tessdata(self) -> str:
        env_value = os.environ.get("TESSDATA_PREFIX", "").strip()
        if env_value:
            return env_value

        candidates = [
            BUNDLED_TESSERACT_DIR / "usr" / "share" / "tesseract-ocr" / "5" / "tessdata",
            BUNDLED_TESSERACT_DIR / "usr" / "share" / "tesseract-ocr" / "tessdata",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        if self.tesseract_cmd:
            prefix = Path(self.tesseract_cmd).resolve().parent.parent
            for candidate in [
                prefix / "share" / "tesseract-ocr" / "5" / "tessdata",
                prefix / "share" / "tesseract-ocr" / "tessdata",
            ]:
                if candidate.exists():
                    return str(candidate)
        return ""

    def _configure_environment(self) -> None:
        if not self.tesseract_cmd:
            return

        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        except Exception as exc:  # pragma: no cover - import failures are rare
            logger.warning("pytesseract could not be configured: %s", exc)

        if self.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = self.tessdata_dir

        # The bundled binary needs its bundled shared libraries.  This only
        # affects this process and its children.
        bundled_lib = BUNDLED_TESSERACT_DIR / "usr" / "lib" / "x86_64-linux-gnu"
        if bundled_lib.exists():
            old = os.environ.get("LD_LIBRARY_PATH", "")
            if str(bundled_lib) not in old.split(":"):
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{bundled_lib}:{old}" if old else str(bundled_lib)
                )

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------
    def _check_available(self) -> bool:
        if not self.tesseract_cmd:
            logger.error(
                "Tesseract is not installed. OCR fallback is disabled. Install "
                "tesseract-ocr with mar/hin language data."
            )
            return False
        try:
            import pytesseract

            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR available: %s", version)
            return True
        except Exception as exc:
            logger.error("Tesseract is not usable: %s", exc)
            return False

    def _read_languages(self) -> list[str]:
        try:
            import pytesseract

            return list(pytesseract.get_languages(config=""))
        except Exception as exc:
            logger.warning("Could not read Tesseract languages: %s", exc)
            try:
                output = subprocess.run(
                    [self.tesseract_cmd, "--list-langs"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                ).stdout
                return [
                    line.strip()
                    for line in output.splitlines()[1:]
                    if line.strip() and not line.lower().startswith("list of")
                ]
            except Exception:
                return []

    def languages_for(self, hint: str | None = None) -> str:
        """Choose an appropriate language set for a page.

        ``mar+hin+eng`` on every page is slow and can reduce English accuracy,
        so we prefer a primary language plus English when the document language
        is known.
        """
        preferred: list[str]
        if hint == "mr":
            preferred = ["mar", "eng"]
        elif hint == "hi":
            preferred = ["hin", "eng"]
        elif hint == "en":
            preferred = ["eng"]
        else:
            preferred = self.requested_languages

        available = self.available_languages
        selected = [lang for lang in preferred if lang in available]
        if not selected and available:
            # Never silently claim a language we do not have.  English is the
            # safest fallback and the warning is recorded by the caller.
            selected = ["eng"] if "eng" in available else [available[0]]
        return "+".join(selected)

    def missing_requested_languages(self) -> list[str]:
        return [
            lang for lang in self.requested_languages if lang not in self.available_languages
        ]

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def image_to_text(self, image: Image.Image, language_hint: str | None = None) -> OCRResult:
        if not self.available:
            return OCRResult(
                text="",
                languages="",
                warnings=["Tesseract OCR is unavailable; page left without OCR text."],
                used_ocr=False,
            )

        languages = self.languages_for(language_hint)
        if not languages:
            return OCRResult(
                text="",
                languages="",
                warnings=["No Tesseract language data is available."],
                used_ocr=False,
            )

        warnings: list[str] = []
        missing = self.missing_requested_languages()
        if missing:
            warnings.append(
                "Missing configured OCR language data: " + ", ".join(missing)
            )
        language_code = {"mr": "mar", "hi": "hin", "en": "eng"}.get(
            language_hint or "", language_hint
        )
        if language_hint in {"mr", "hi"} and language_code not in languages.split("+"):
            warnings.append(
                f"Requested language {language_hint} is unavailable; OCR used {languages}."
            )

        try:
            import pytesseract

            prepared = ImageOps.autocontrast(ImageOps.grayscale(image))
            text = pytesseract.image_to_string(
                prepared,
                lang=languages,
                config="--oem 3 --psm 3",
                timeout=self.page_timeout_seconds,
            )
            return OCRResult(
                text=text.strip(),
                languages=languages,
                warnings=warnings,
                used_ocr=bool(text.strip()),
            )
        except Exception as exc:
            logger.warning("OCR failed for a page (%s): %s", languages, exc)
            return OCRResult(
                text="",
                languages=languages,
                warnings=warnings
                + [f"OCR failed with languages {languages}: {exc}"],
                used_ocr=False,
            )

    def health(self) -> dict[str, object]:
        return {
            "available": self.available,
            "binary": self.tesseract_cmd or None,
            "tessdata": self.tessdata_dir or None,
            "available_languages": self.available_languages,
            "missing_requested_languages": self.missing_requested_languages(),
        }


@lru_cache(maxsize=2)
def get_ocr_service(
    languages: str = "mar+hin+eng",
    page_timeout_seconds: int = 120,
    tesseract_cmd: str = "",
) -> OCRService:
    return OCRService(
        languages=languages,
        page_timeout_seconds=page_timeout_seconds,
        tesseract_cmd=tesseract_cmd,
    )
