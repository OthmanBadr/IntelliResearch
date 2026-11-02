import re
from dataclasses import dataclass, field

from .grounding import render_answer, validate_chart
from .ingestion import corpus_id, extract_pages, tesseract_ocr
from .models import Chunk, ResearchError, Upload
from .retrieval import Index, split_pages


@dataclass
class Workspace:
    fingerprint: str
    uploads: list[Upload]
    index: Index
    warnings: list[str]
    page_count: int
    messages: list[dict] = field(default_factory=list)
    summary: object = None
    questions: list[str] = field(default_factory=list)


def prepare(uploads, options, encoder, provider=None) -> Workspace:
    if options.get("visual") and provider is None:
        raise ResearchError("Figure analysis requires a configured Gemini client.")
    pages, warnings = extract_pages(
        uploads, ocr=tesseract_ocr if options.get("ocr") else None,
        ocr_languages=options.get("ocr_languages", "eng+ara"),
        analyze_visual=provider.visual if options.get("visual") and provider else None,
    )
    index = Index(split_pages(pages, encoder.tokenizer), encoder)
    return Workspace(corpus_id(uploads, options), uploads, index, warnings, len(pages))


def ask(workspace: Workspace, question: str, provider):
    question = question.strip()
    if not question or len(question) > 2000:
        raise ResearchError("Enter a question between 1 and 2,000 characters.")
    history = [{"role": m["role"], "content": m["content"][:4000]} for m in workspace.messages[-8:]]
    previous = [m["content"][:300] for m in history if m["role"] == "user"][-2:]
    # Put the new question first so long history cannot displace it at the encoder's token limit.
    query = question + ("\nEarlier questions: " + "\n".join(previous) if previous else "")
    chunks = workspace.index.search(query)
    raw = provider.answer(question, chunks, history)
    answer = render_answer(raw, chunks, arabic=bool(re.search(r"[\u0600-\u06ff]", question)))
    # Mutate history only after the whole operation succeeds, so failures can be retried.
    workspace.messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer.markdown, "sources": answer.sources},
    ])
    return answer


def summarize(workspace: Workspace, provider, language: str = "English"):
    chunks = workspace.index.chunks
    evidence = {}
    for start in range(0, len(chunks), 8):
        batch = chunks[start:start + 8]
        result = provider.answer(
            f"In {language}, extract the main research objectives, methods, findings and limitations "
            "from this batch. Include at most five claims. Leave questions empty.", batch,
        )
        render_answer(result, batch)  # Validate every intermediate quote before synthesis.
        if result.supported:
            for claim in result.claims:
                for item in claim.evidence:
                    evidence.setdefault(item.source_id, []).append(item.quote)
    if not evidence:
        raise ResearchError("There was not enough grounded evidence to create a summary.")
    original = {c.id: c for c in chunks}
    reduced = [Chunk(c.id, c.document_id, c.filename, c.page, "\n".join(dict.fromkeys(quotes)), c.kind)
               for identity, quotes in evidence.items() for c in [original[identity]]]
    if sum(len(c.text) for c in reduced) > 60_000:
        raise ResearchError("These documents are too large for one combined summary. Process fewer PDFs together.")
    result = provider.answer(
        f"In {language}, write a concise combined research summary from these extracted quotes. "
        "Distinguish papers, explain objectives, methods, key findings and limitations. "
        "Only compare compatible results. Generate five useful follow-up questions in the same language.",
        reduced,
    )
    # Check final quotes against the original chunks as well as the reduced context.
    render_answer(result, reduced)
    answer = render_answer(result, chunks, arabic=language == "Arabic")
    workspace.summary = answer
    workspace.questions = result.questions if result.supported else []
    return answer


def make_chart(workspace: Workspace, question: str, provider):
    if not question.strip():
        raise ResearchError("Describe the chart you want to extract.")
    chunks = workspace.index.search(question, k=10)
    spec = provider.chart(question, chunks)
    validate_chart(spec, chunks)
    return spec, chunks
