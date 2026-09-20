"""
Indexing entry point: fetch the docs, split them, write data/processed/passages.json,
and optionally upload the vectors to Qdrant.

Run with:  python -m scripts.build_index
"""

import argparse
import json
from pathlib import Path

from doctrace.corpus.fetcher import pull_docs, read_docs
from doctrace.corpus.splitter import split_by_headings

PASSAGES_PATH = Path("data/processed/passages.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the FastAPI docs into passages.json and optionally upload to Qdrant.")
    parser.add_argument("--sync-qdrant", action="store_true", help="Also encode the passages and upload them to the Qdrant collection")
    parser.add_argument("--recreate-collection", action="store_true", help="Drop and recreate the Qdrant collection before uploading")
    args = parser.parse_args()

    pull_docs()
    docs = read_docs()
    print(f"Loaded {len(docs)} docs")

    blank = [d for d in docs if not d["text"].strip()]
    assert not blank, f"{len(blank)} docs loaded with empty text"

    unresolved = sum(d["text"].count("[MISSING SNIPPET") for d in docs)
    assert unresolved == 0, f"{unresolved} snippet includes failed to resolve"

    passages = []
    for d in docs:
        passages.extend(split_by_headings(d["text"], d["source_path"]))
    print(f"Total passages: {len(passages)}")

    unbalanced = [p for p in passages if p["text"].count("`" * 3) % 2 != 0]
    assert not unbalanced, f"{len(unbalanced)} passages have an unclosed code fence"

    PASSAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PASSAGES_PATH.write_text(json.dumps(passages, indent=2), encoding="utf-8")
    print(f"Wrote {len(passages)} passages to {PASSAGES_PATH}")
    print("\nChecks passed.")

    if args.sync_qdrant:
        from doctrace.search.semantic import ensure_collection, index_passages
        print("\nUploading passages to Qdrant...")
        ensure_collection(recreate=args.recreate_collection)
        index_passages(passages)
        print("Qdrant upload complete.")


if __name__ == "__main__":
    main()
