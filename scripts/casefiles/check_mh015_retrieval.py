"""
Did mh_015's required source passages actually appear in the context it was
given? Reads the saved retrieval stash; nothing is re-run.

Run with: python -m scripts.casefiles.check_mh015_retrieval
"""

import json

from doctrace.audit.goldset import load_gold_pairs, attach_gold_ids


def main() -> None:
    stash = {r["id"]: r for r in json.loads(open("data/processed/retrieval_stash.json", encoding="utf-8").read())}
    gold_pairs = load_gold_pairs()
    resolved = {p["id"]: p for p in attach_gold_ids(gold_pairs)}

    pair = resolved["mh_015"]
    retrieved = stash["mh_015"]["retrieved"]
    retrieved_ids = {r["id"] for r in retrieved}

    print("Ground truth passage ids needed:", pair["ground_truth_ids"])
    print("Retrieved passage ids (top-5):", sorted(retrieved_ids))
    print()
    for r in retrieved:
        print(f"  id={r['id']}  {r['source_path']} > {r['header_path']}")

    missing = set(pair["ground_truth_ids"]) - retrieved_ids
    print(f"\nGround truth passages MISSING from retrieved context: {missing}")


if __name__ == "__main__":
    main()
