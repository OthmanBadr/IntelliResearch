from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from intelliresearch.models import ResearchError
from intelliresearch.provider import Gemini


def test_provider_uses_structured_sdk_and_serializes_history():
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text='{"supported":false,"claims":[],"questions":[]}')
    provider = Gemini("test-only", client=client)
    answer = provider.answer("A follow-up?", [], [{"role": "user", "content": "Earlier question"}])
    call = client.models.generate_content.call_args.kwargs
    assert "Earlier question" in call["contents"]
    assert call["config"]["response_mime_type"] == "application/json"
    assert not answer.supported


def test_raw_provider_errors_are_not_exposed():
    client = Mock()
    client.models.generate_content.side_effect = RuntimeError("private source text and secret credentials")
    with pytest.raises(ResearchError) as error:
        Gemini("test-only", client=client).answer("Question", [])
    assert "secret" not in str(error.value) and "private source" not in str(error.value)
