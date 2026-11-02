# Evaluation

The bundled cases are **synthetic smoke tests**, not evidence of research-paper accuracy.
They cover two English documents, one Arabic document, overlapping page numbers,
cross-language retrieval, printed numeric values, and unanswerable questions.

```bash
python -m evaluation.run
# Optional: uses your configured key and incurs Gemini usage charges.
python -m evaluation.run --live
```

The first command downloads E5 weights if needed, but makes no Gemini calls.
It measures page recall@3 on answerable cases and retrieval latency. Unanswerable
questions have no gold retrieval pages and are excluded from recall. `--k` changes
the retrieval metric only; live answers use the app's default top six passages.

Live runs record answer text, abstention, errors and a simple expected-term check.
Expected-term matching is **not** a correctness or faithfulness score. Inspect
whether each claim follows from its cited excerpt; matching quotes alone cannot
prove that. Errors should be counted as failures, not abstentions.

For a meaningful portfolio evaluation, replace the synthetic material with a
small, legally shareable set of real papers and manually label 30–50 questions.
Include Arabic, scanned pages, tables, vector figures, multi-paper questions,
follow-ups, and questions with no answer in the corpus. Keep evaluation questions
separate from those used to tune chunking or prompts. Record:

| Measure | How to score |
|---|---|
| Page recall@k | Fraction of gold evidence pages retrieved |
| Answer correctness | Human review against a reference answer |
| Citation support | Fraction of factual claims supported by their cited excerpts |
| Abstention | Whether unanswerable questions are declined, without declining answerable ones |
| Numeric fidelity | Exact agreement with printed table values and units |
| OCR quality | Transcription errors on a manually checked page sample |
| Latency and cost | Separate ingestion, summary, and question-answering measurements |

Use the structure of `cases.json` for custom evaluation data:
`python -m evaluation.run --cases /path/to/cases.json --output /path/to/results.json`.
Do not commit private papers, API keys, or reports containing private excerpts.
The text fixtures exercise retrieval and generation. PDF parsing, OCR wiring,
figure association, and chat/session regressions are covered separately by tests.
