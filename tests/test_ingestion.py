import pymupdf
import pytest

from intelliresearch.ingestion import corpus_id, extract_pages, preview_page, validate_uploads
from intelliresearch.models import ResearchError, Upload
from conftest import pdf_bytes


def test_sources_keep_document_and_exact_page_identity():
    first = Upload("same.pdf", pdf_bytes(*[f"Paper A page {i} contains a different research result." for i in range(1, 11)]))
    second = Upload("same.pdf", pdf_bytes("Paper B page 1 contains an unrelated research result."))
    pages, _ = extract_pages([first, second])
    assert pages[0].number == 1 and pages[9].number == 10
    assert pages[0].document_id != pages[10].document_id
    assert "Paper B" in pages[10].text and "Paper A" not in pages[10].text
    assert preview_page([first, second], pages[10].document_id, 1).startswith(b"\x89PNG")


def test_corpus_changes_when_content_or_options_change(uploads):
    assert corpus_id(uploads, {}) == corpus_id(list(reversed(uploads)), {})
    assert corpus_id(uploads, {}) != corpus_id(uploads, {"ocr": True})
    assert corpus_id(uploads, {}) != corpus_id([Upload(uploads[0].name, pdf_bytes("Changed text"))], {})


@pytest.mark.parametrize("uploads", [[], [Upload("x.pdf", b"not a pdf")], [Upload("x.txt", b"%PDF-")]])
def test_invalid_uploads_fail_before_processing(uploads):
    with pytest.raises(ResearchError):
        validate_uploads(uploads)


def test_duplicates_and_page_limits(uploads):
    with pytest.raises(ResearchError, match="duplicate"):
        validate_uploads([uploads[0], uploads[0]])
    with pytest.raises(ResearchError, match="100"):
        validate_uploads([Upload("long.pdf", pdf_bytes(*(["Page"] * 101)))])


def test_ocr_is_called_only_for_sparse_pages():
    calls = []
    def ocr(png, languages):
        calls.append(languages)
        return "نتائج البحث تشير إلى دقة مرتفعة في التجربة."
    upload = Upload("scan.pdf", pdf_bytes("", "This page has sufficient native text and must not use OCR."))
    pages, warnings = extract_pages([upload], ocr=ocr)
    assert calls == ["eng+ara"]
    assert "نتائج" in pages[0].text
    assert not warnings


def test_vector_figures_are_attached_to_their_own_page():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 70), "This figure belongs only to the first research paper.")
        page.draw_rect(pymupdf.Rect(50, 100, 150, 200))
        first = Upload("a.pdf", doc.tobytes())
    second = Upload("b.pdf", pdf_bytes("This second paper has no figures and no visual description."))
    calls = []
    def vision(png):
        calls.append(png)
        return "A rectangle is shown."
    pages, _ = extract_pages([first, second], analyze_visual=vision)
    assert len(calls) == 1
    assert pages[0].visual_notes == "A rectangle is shown."
    assert pages[1].visual_notes == ""
