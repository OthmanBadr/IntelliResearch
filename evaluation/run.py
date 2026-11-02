"""Run: python -m evaluation.run [--live] [--cases path] [--output path]."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from intelliresearch.models import Page, ResearchError
from intelliresearch.provider import Gemini
from intelliresearch.retrieval import E5Encoder, Index, split_pages
from intelliresearch.service import Workspace, ask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--live", action="store_true", help="Make billable Gemini calls using GEMINI_API_KEY.")
    parser.add_argument("--k", type=int, default=3, choices=range(1, 11))
    args = parser.parse_args()
    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    index = Index(split_pages([Page(**p) for p in dataset["pages"]], (encoder := E5Encoder()).tokenizer), encoder)
    provider = Gemini(os.environ.get("GEMINI_API_KEY", ""), os.getenv("GEMINI_MODEL", "gemini-flash-latest")) if args.live else None
    rows = []
    try:
        for case in dataset["questions"]:
            start = time.perf_counter()
            retrieved = index.search(case["question"], args.k)
            actual = {(c.document_id, c.page) for c in retrieved}
            gold = {tuple(p) for p in case["expected_pages"]}
            row = {"id": case["id"], "retrieved_pages": sorted(actual),
                   "recall_at_k": len(actual & gold) / len(gold) if gold else None,
                   "retrieval_seconds": time.perf_counter() - start}
            if provider:
                # Fresh conversation per case, same immutable index.
                workspace = Workspace("evaluation", [], index, [], len(dataset["pages"]))
                start = time.perf_counter()
                try:
                    result = ask(workspace, case["question"], provider)
                    row.update(answer=result.markdown, abstained=not bool(result.sources),
                               expected_terms_present=all(t.casefold() in result.markdown.casefold() for t in case["answer_contains"]) if gold else None)
                except ResearchError as error:
                    row["error"] = str(error)
                row["answer_seconds"] = time.perf_counter() - start
            rows.append(row)
    finally:
        if provider:
            provider.close()
    recalls = [r["recall_at_k"] for r in rows if r["recall_at_k"] is not None]
    report = {"dataset": dataset["description"], "embedding_model": "intfloat/multilingual-e5-small",
              "k": args.k, "mean_recall_at_k": float(np.mean(recalls)) if recalls else None,
              "retrieval_p50_seconds": float(np.median([r["retrieval_seconds"] for r in rows])),
              "live_generation": args.live, "cases": rows}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    print(f"Details written to {args.output}")


if __name__ == "__main__":
    main()
