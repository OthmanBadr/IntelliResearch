import io
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

import app
from conftest import FakeProvider


def test_empty_app_and_compatibility_entrypoint():
    for path in ("app.py", "IntelliResearch.py"):
        result = AppTest.from_file(Path(__file__).resolve().parents[1] / path).run(timeout=15)
        assert not result.exception
        assert not result.error
        assert result.title[0].value == "IntelliResearch"


def test_upload_chat_reruns_and_corpus_change(monkeypatch, uploads, encoder):
    files = []
    for upload in uploads:
        file = io.BytesIO(upload.data)
        file.name = upload.name
        files.append(file)
    monkeypatch.setattr(app.st, "file_uploader", lambda *args, **kwargs: files)
    monkeypatch.setattr(app, "get_encoder", lambda: encoder)
    fake = FakeProvider()
    @contextmanager
    def provider(*args):
        yield fake
    monkeypatch.setattr(app, "provider", provider)
    prepare = Mock(wraps=app.prepare)
    monkeypatch.setattr(app, "prepare", prepare)
    result = AppTest.from_string("from app import main\nmain()").run()
    result.button[0].click().run()
    assert not result.exception
    assert prepare.call_count == 1
    result.chat_input[0].set_value("What is camera latency?").run()
    assert not result.exception
    assert fake.calls[-1][0] == "What is camera latency?"
    assert prepare.call_count == 1
    result.chat_input[0].set_value("What about accuracy?").run()
    assert fake.calls[-1][2][0]["content"] == "What is camera latency?"
    result.button[0].click().run()
    assert prepare.call_count == 1
    files.pop()
    result.run()
    assert "workspace" not in result.session_state
    assert not result.exception
