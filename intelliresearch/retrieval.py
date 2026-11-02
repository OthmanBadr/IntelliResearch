import numpy as np

from .models import Chunk, Page, ResearchError

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class E5Encoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        except Exception:
            raise ResearchError("The embedding model could not be loaded. Check Hugging Face connectivity and the local model cache.") from None
        self.tokenizer = self.model.tokenizer

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def split_pages(pages: list[Page], tokenizer, size: int = 384, overlap: int = 64) -> list[Chunk]:
    if not 0 <= overlap < size <= 480:
        raise ValueError("Chunk size must leave space for the E5 prefix and special tokens.")
    chunks = []
    for page in pages:
        for kind, text in (("text", page.text), ("visual interpretation", page.visual_notes)):
            if not text.strip():
                continue
            offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for start in range(0, len(offsets), size - overlap):
                window = offsets[start:start + size]
                content = text[window[0][0]:window[-1][1]].strip()
                if content:
                    chunks.append(Chunk(
                        f"{page.document_id[:16]}-p{page.number}-{kind.split()[0]}-{start}",
                        page.document_id, page.filename, page.number, content, kind,
                    ))
                if start + size >= len(offsets):
                    break
    return chunks


class Index:
    """An in-memory index owned by one browser session, never a shared collection."""

    def __init__(self, chunks: list[Chunk], encoder):
        if not chunks:
            raise ResearchError("No text chunks are available to index.")
        self.chunks = chunks
        self.encoder = encoder
        self.vectors = self._normalize(encoder.encode(["passage: " + c.text for c in chunks]))

    @staticmethod
    def _normalize(vectors):
        values = np.asarray(vectors, dtype=np.float32)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def search(self, question: str, k: int = 6) -> list[Chunk]:
        if not question.strip():
            return []
        vector = self._normalize(self.encoder.encode(["query: " + question]))[0]
        scores = self.vectors @ vector
        return [self.chunks[int(i)] for i in np.argsort(-scores, kind="stable")[:k]]
