"""
Case file: reranker demotions.

For specific regression cases, prints the gold passage's text and cross-encoder score
next to whatever the cross-encoder placed above it, to show why
ms-marco-MiniLM-L-6-v2 disagreed with the fusion ordering.

Run with: python -m scripts.casefiles.inspect_reranker_demotion
"""

import json

from doctrace.search.lexical import build_lexical_index, lexical_search
from doctrace.search.semantic import semantic_search
from doctrace.search.fusion import rrf_merge, rescore

POOL_SIZE = 20

CASES = [
    {
        "question": "What underlying library is FastAPI's `TestClient` built on?",
        "gt_ids": {495},
    },
    {
        "question": (
            "If you return a `Response` object directly from a path operation instead of a "
            "`dict`/Pydantic model, does FastAPI still validate or serialize its contents "
            "using your declared `response_model`? And relatedly, what tool would you need "
            "to use manually if you wanted to put non-JSON-serializable data (like a "
            "`datetime`) into that raw `Response`?"
        ),
        "gt_ids": {649, 650, 149},
    },
]


def main() -> None:
    passages = json.loads(open("data/processed/passages.json", encoding="utf-8").read())
    build_lexical_index(passages)

    for case in CASES:
        question = case["question"]
        gt_ids = case["gt_ids"]

        print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")

        dense_results = semantic_search(question, top_k=POOL_SIZE)
        bm25_results = lexical_search(question, top_k=POOL_SIZE)
        fused = rrf_merge([dense_results, bm25_results])

        reranked = rescore(question, fused[:POOL_SIZE], top_k=len(fused[:POOL_SIZE]))

        print("\nFull reranked order (all candidates, with scores):")
        for i, r in enumerate(reranked, start=1):
            marker = " <-- GROUND TRUTH" if r["id"] in gt_ids else ""
            print(f"  #{i:2d}  id={r['id']:4d}  ce_score={r['ce_score']:7.3f}  "
                  f"{r['source_path']} > {r['header_path']}{marker}")

        print("\nActual text of ground truth passage(s):")
        for r in reranked:
            if r["id"] in gt_ids:
                print(f"\n  [id={r['id']}] {r['source_path']} > {r['header_path']}")
                print(f"  {r['text'][:400]}")

        print("\nActual text of the TOP-RANKED passage (what beat the ground truth):")
        top = reranked[0]
        if top["id"] not in gt_ids:
            print(f"\n  [id={top['id']}] {top['source_path']} > {top['header_path']}")
            print(f"  {top['text'][:400]}")
        else:
            print("  (ground truth was actually ranked #1 here)")


if __name__ == "__main__":
    main()