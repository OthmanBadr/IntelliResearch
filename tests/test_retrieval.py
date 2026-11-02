import pytest

from intelliresearch.models import Page
from intelliresearch.retrieval import Index, split_pages


def test_token_chunks_overlap_without_losing_text_or_metadata(encoder):
    pages = [Page("a" * 64, "arabic.pdf", 10, "هذه دراسة عن تعلم الآلة وتقييم دقة النموذج", "رسم يوضح النتائج")]
    chunks = split_pages(pages, encoder.tokenizer, size=4, overlap=1)
    assert chunks[0].text.endswith("تعلم")
    assert chunks[1].text.startswith("تعلم")
    assert "النموذج" in " ".join(c.text for c in chunks)
    assert all(c.page == 10 and c.document_id == "a" * 64 for c in chunks)
    assert chunks[-1].kind == "visual interpretation"
    assert len({c.id for c in chunks}) == len(chunks)


def test_index_built_once_and_e5_prefixes_used(encoder):
    chunks = split_pages([Page("a", "a.pdf", 1, "battery accuracy results"), Page("b", "b.pdf", 1, "camera latency results")], encoder.tokenizer)
    index = Index(chunks, encoder)
    assert index.search("camera latency", k=1)[0].filename == "b.pdf"
    index.search("battery accuracy")
    assert len(encoder.calls) == 3
    assert all(s.startswith("passage: ") for s in encoder.calls[0])
    assert encoder.calls[1] == ["query: camera latency"]


def test_indexes_do_not_share_document_state(encoder):
    a = Index(split_pages([Page("a", "a.pdf", 1, "battery accuracy")], encoder.tokenizer), encoder)
    b = Index(split_pages([Page("b", "b.pdf", 1, "camera latency")], encoder.tokenizer), encoder)
    assert a.search("camera")[0].filename == "a.pdf"
    assert b.search("battery")[0].filename == "b.pdf"


def test_bad_chunk_size_rejected(encoder):
    with pytest.raises(ValueError):
        split_pages([], encoder.tokenizer, size=10, overlap=10)
