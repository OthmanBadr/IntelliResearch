import re

import numpy as np
import pymupdf
import pytest

from intelliresearch.models import Claim, Evidence, GroundedAnswer, Upload


def pdf_bytes(*texts):
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page()
            page.insert_text((50, 70), text)
        return doc.tobytes()


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


class FakeEncoder:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return np.asarray([[text.lower().count(word) for word in ("battery", "camera", "accuracy", "latency")] + [0.1]
                           for text in texts], dtype=float)


class FakeProvider:
    def __init__(self):
        self.calls = []

    def answer(self, question, chunks, history=()):
        self.calls.append((question, chunks, history))
        return GroundedAnswer(supported=True, claims=[
            Claim(text="The paper reports this finding.", evidence=[Evidence(source_id=chunks[0].id, quote=chunks[0].text[:500])])
        ], questions=["What does the paper report?"])

    def close(self):
        pass


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def uploads():
    return [Upload("battery.pdf", pdf_bytes("Battery accuracy is 92 percent. The evaluation used 100 samples.")),
            Upload("camera.pdf", pdf_bytes("Camera latency is 40 milliseconds. The evaluation used 200 samples."))]
