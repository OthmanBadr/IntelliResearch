import pytest

from conftest import FakeProvider
from intelliresearch.models import Claim, Evidence, GroundedAnswer, ResearchError
from intelliresearch.service import ask, prepare, summarize


def test_followup_gets_history_and_actual_question_without_reindexing(uploads, encoder):
    workspace = prepare(uploads, {}, encoder)
    provider = FakeProvider()
    ask(workspace, "What is the camera latency?", provider)
    ask(workspace, "How was it evaluated?", provider)
    assert provider.calls[-1][0] == "How was it evaluated?"
    assert provider.calls[-1][2][0]["content"] == "What is the camera latency?"
    assert len(workspace.messages) == 4
    assert sum(any(t.startswith("passage:") for t in call) for call in encoder.calls) == 1


def test_failed_answer_does_not_mutate_history(uploads, encoder):
    workspace = prepare(uploads, {}, encoder)
    class BadProvider:
        def answer(self, *args):
            return GroundedAnswer(supported=True, claims=[Claim(text="A fabricated claim", evidence=[Evidence(source_id="made-up", quote="Invented evidence")])])
    with pytest.raises(ResearchError):
        ask(workspace, "What is the accuracy?", BadProvider())
    assert workspace.messages == []


def test_summary_visits_all_chunks_before_synthesis(uploads, encoder):
    workspace = prepare(uploads, {}, encoder)
    provider = FakeProvider()
    summarize(workspace, provider)
    visited = {c.id for _, batch, _ in provider.calls[:-1] for c in batch}
    assert visited == {c.id for c in workspace.index.chunks}
    assert workspace.summary.sources
    assert workspace.questions
