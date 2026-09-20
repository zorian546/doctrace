"""
Smoke test for semantic search.

Uploads every passage to Qdrant, then runs a few hand-picked questions to confirm the
vector search returns sensible hits before the keyword search and fusion are layered on.

Run with: python -m scripts.smoke_semantic
"""

import json

from doctrace.search.semantic import ensure_collection, index_passages, semantic_search

TEST_QUERIES = [
    "How do I define a path parameter with a type hint?",
    "How does dependency injection work in FastAPI?",
    "What is the difference between async and sync path operations?",
]


def main() -> None:
    passages = json.loads(open("data/processed/passages.json", encoding="utf-8").read())
    print(f"Indexing {len(passages)} passages...")

    ensure_collection(recreate=True)
    index_passages(passages)
    print("Done indexing.")

    for query in TEST_QUERIES:
        print()
        print("QUERY:", query)
        for r in semantic_search(query, top_k=3):
            print(f"  [{r['score']:.3f}] {r['source_path']} > {r['header_path']}")


if __name__ == "__main__":
    main()