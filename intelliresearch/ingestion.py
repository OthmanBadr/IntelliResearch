import hashlib
import json
from collections.abc import Callable

import pymupdf

from .models import Page, ResearchError, Upload

MAX_FILES = 5
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 100


def document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def corpus_id(uploads: list[Upload], options: dict) -> str:
    manifest = sorted((u.name, document_id(u.data)) for u in uploads)
    return hashlib.sha256(json.dumps([manifest, options], sort_keys=True).encode()).hexdigest()


def validate_uploads(uploads: list[Upload]) -> None:
    if not 1 <= len(uploads) <= MAX_FILES:
        raise ResearchError(f"Choose between 1 and {MAX_FILES} PDF files.")
    seen = set()
    page_count = 0
    for upload in uploads:
        if not upload.name.lower().endswith(".pdf") or not upload.data.startswith(b"%PDF-"):
            raise ResearchError("Every upload must be a PDF.")
        if len(upload.data) > MAX_FILE_BYTES:
            raise ResearchError("Each PDF must be at most 20 MB.")
        identity = document_id(upload.data)
        if identity in seen:
            raise ResearchError("The same PDF was uploaded more than once. Remove the duplicate.")
        seen.add(identity)
        try:
            with pymupdf.open(stream=upload.data, filetype="pdf") as doc:
                if doc.needs_pass or not len(doc):
                    raise ResearchError("Use nonempty PDFs without password protection.")
                page_count += len(doc)
        except ResearchError:
            raise
        except Exception:
            raise ResearchError("A PDF could not be opened. Check that it is not damaged.") from None
    if page_count > MAX_PAGES:
        raise ResearchError(f"Upload at most {MAX_PAGES} pages in total.")


def render_page(page: pymupdf.Page, max_side: int = 1800) -> bytes:
    scale = min(2.0, max_side / max(page.rect.width, page.rect.height))
    return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")


def tesseract_ocr(png: bytes, languages: str) -> str:
    import io

    import pytesseract
    from PIL import Image

    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(png)), lang=languages, timeout=45)
    except Exception:
        raise ResearchError("OCR failed. Install Tesseract and the selected language packs.") from None


def extract_pages(
    uploads: list[Upload],
    *,
    ocr_languages: str = "eng+ara",
    ocr: Callable | None = None,
    analyze_visual: Callable[[bytes], str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Page], list[str]]:
    validate_uploads(uploads)
    pages, warnings = [], []
    for upload in uploads:
        identity = document_id(upload.data)
        with pymupdf.open(stream=upload.data, filetype="pdf") as doc:
            for index, pdf_page in enumerate(doc):
                text = pdf_page.get_text(sort=True).strip()
                png = None
                if len(text) < 30 and ocr is not None:
                    png = render_page(pdf_page)
                    text = ocr(png, ocr_languages).strip() or text
                if not text:
                    warnings.append(f"{upload.name}, page {index + 1}: no text extracted.")
                notes = ""
                # Vector drawings matter too: many academic plots contain no raster images.
                if analyze_visual and (pdf_page.get_images() or pdf_page.get_drawings()):
                    png = png or render_page(pdf_page)
                    notes = analyze_visual(png)
                pages.append(Page(identity, upload.name, index + 1, text, notes))
                if progress:
                    progress(index + 1, len(doc))
    if not any(p.text or p.visual_notes for p in pages):
        raise ResearchError("No usable content was extracted. Enable OCR for scanned PDFs.")
    return pages, warnings


def preview_page(uploads: list[Upload], identity: str, number: int) -> bytes:
    upload = next(u for u in uploads if document_id(u.data) == identity)
    with pymupdf.open(stream=upload.data, filetype="pdf") as doc:
        return render_page(doc[number - 1])
