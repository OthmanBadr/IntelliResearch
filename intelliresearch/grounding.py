import re
import unicodedata

from .models import Answer, ChartSpec, Chunk, Evidence, GroundedAnswer, ResearchError


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def check_evidence(evidence: Evidence, sources: dict[str, Chunk]) -> Chunk:
    source = sources.get(evidence.source_id)
    if source is None or normalize(evidence.quote) not in normalize(source.text):
        raise ResearchError("The generated answer failed its source check. Try a more specific question.")
    return source


def render_answer(answer: GroundedAnswer, chunks: list[Chunk], *, arabic: bool = False) -> Answer:
    if not answer.supported or not answer.claims:
        message = (
            "لم أجد أدلة كافية في المقاطع المسترجعة للإجابة. جرّب سؤالاً أكثر تحديداً أو راجع المستندات."
            if arabic else
            "I could not find enough evidence in the retrieved passages to answer. "
            "Try a more specific question or inspect the documents."
        )
        return Answer(message)
    sources = {c.id: c for c in chunks}
    used, lines = {}, []
    for claim in answer.claims:
        citations = []
        for evidence in claim.evidence:
            source = check_evidence(evidence, sources)
            if source.id not in used:
                used[source.id] = (len(used) + 1, source)
            number = used[source.id][0]
            if f"[{number}]" not in citations:
                citations.append(f"[{number}]")
        lines.append(f"{claim.text} {' '.join(citations)}")
    lines.append("\n**Sources**")
    for number, source in used.values():
        note = " — AI visual interpretation; verify against the page" if source.kind != "text" else ""
        # Display metadata as text; uploaded names are never filesystem paths or HTML.
        filename = re.sub(r"([\\`*_{}\[\]<>])", r"\\\1", source.filename)
        lines.append(f"[{number}] {filename}, page {source.page}{note}")
    return Answer("\n\n".join(lines), tuple(source for _, source in used.values()))


def validate_chart(chart: ChartSpec, chunks: list[Chunk]) -> None:
    sources = {c.id: c for c in chunks if c.kind == "text"}
    labels = set()
    for point in chart.points:
        check_evidence(point.evidence, sources)
        quote = normalize(point.evidence.quote)
        if normalize(point.label).casefold() not in quote.casefold():
            raise ResearchError("A chart label could not be matched to its source excerpt.")
        # Arabic and Persian digits are accepted by float; normalize decimal separators.
        numeric_text = quote.replace("٬", "").replace("٫", ".")
        numeric_text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", numeric_text)
        values = [float(v) for v in re.findall(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?", numeric_text)]
        if point.value not in values:
            raise ResearchError("A chart value could not be matched to a printed source value.")
        if point.label in labels:
            raise ResearchError("Chart labels must be unique. Ask for a single comparison.")
        labels.add(point.label)
