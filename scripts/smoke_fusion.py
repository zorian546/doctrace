"""
Smoke test for fused search: semantic-only versus fusion plus cross-encoder
rescoring on sample questions, with a per-stage latency breakdown.

Run with:
    python -m scripts.smoke_fusion
"""

import json

from doctrace.search.lexical import build_lexical_index
from doctrace.search.semantic import semantic_search
from doctrace.search.fusion import fused_search, _load_cross_encoder

TEST_QUERIES = [
    "How do I define a path parameter with a type hint?",
    "How does dependency injection work in FastAPI?",
    "What is the difference between async and sync path operations?",
    "How do I set a response_model in a path operation decorator?",  # exact-term-heavy, good BM25 test
]


def main() -> None:
    passages = json.loads(open("data/processed/passages.json", encoding="utf-8").read())
    print(f"Building BM25 index over {len(passages)} passages...")
    build_lexical_index(passages)
    print("Done.\n")

    # Warm up the cross-encoder (triggers model download/load) before
    # timing anything, so the first real query's rescore_ms isn't
    # polluted by one-time download/load overhead.
    print("Warming up cross-encoder...")
    _load_cross_encoder()
    print("Done.\n")

    for query in TEST_QUERIES:
        print("=" * 80)
        print("QUERY:", query)

        print("\n-- Dense-only --")
        for r in semantic_search(query, top_k=3):
            print(f"  [{r['score']:.3f}] {r['source_path']} > {r['header_path']}")

        print("\n-- Hybrid + reranked --")
        out = fused_search(query, top_k=3)
        for r in out["results"]:
            print(f"  [rescore={r['ce_score']:.3f}, rrf={r['fusion_score']:.4f}] "
                  f"{r['source_path']} > {r['header_path']}")
        print(f"\n  latency: {out['latency_ms']}")
        print()


if __name__ == "__main__":
    main()