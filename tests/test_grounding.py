import pytest
from pydantic import ValidationError

from intelliresearch.grounding import render_answer, validate_chart
from intelliresearch.models import ChartPoint, ChartSpec, Chunk, Claim, Evidence, GroundedAnswer, ResearchError


def source(kind="text"):
    return Chunk("doc-p1", "doc", "paper.pdf", 1, "Model A accuracy is 92 percent. Model B accuracy is 85 percent.", kind)


def test_only_cited_sources_are_rendered():
    chunk = source()
    raw = GroundedAnswer(supported=True, claims=[Claim(text="Model A achieved 92%.", evidence=[Evidence(source_id=chunk.id, quote="Model A accuracy is 92 percent.")])])
    unused = Chunk("other", "other", "uncited.pdf", 3, "An irrelevant passage")
    result = render_answer(raw, [chunk, unused])
    assert "[1]" in result.markdown and "uncited.pdf" not in result.markdown
    assert result.sources == (chunk,)


@pytest.mark.parametrize("identity, quote", [("invented", "Model A accuracy is 92 percent."), ("doc-p1", "Model A accuracy is 99 percent.")])
def test_hallucinated_citation_or_quote_rejected(identity, quote):
    raw = GroundedAnswer(supported=True, claims=[Claim(text="Unsupported claim", evidence=[Evidence(source_id=identity, quote=quote)])])
    with pytest.raises(ResearchError):
        render_answer(raw, [source()])


def test_abstention_does_not_append_retrieved_references():
    answer = render_answer(GroundedAnswer(supported=False), [source()])
    assert not answer.sources and "Sources" not in answer.markdown


def chart(value=92, label="Model A"):
    return ChartSpec(title="Accuracy", kind="bar", unit="percent", points=[ChartPoint(label=label, value=value, evidence=Evidence(source_id="doc-p1", quote="Model A accuracy is 92 percent."))])


def test_printed_chart_values_only():
    validate_chart(chart(), [source()])
    with pytest.raises(ResearchError):
        validate_chart(chart(value=99), [source()])
    with pytest.raises(ResearchError):
        validate_chart(chart(label="Model C"), [source()])
    with pytest.raises(ResearchError):
        validate_chart(chart(), [source(kind="visual interpretation")])


def test_chart_schema_rejects_code_and_nonfinite_values():
    with pytest.raises(ValidationError):
        ChartSpec.model_validate({**chart().model_dump(), "code": "print('unsafe')"})
    with pytest.raises(ValidationError):
        chart(value=float("nan"))
