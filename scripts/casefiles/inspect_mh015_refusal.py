"""
Case file: a closer look at mh_015's over-refusal, using saved data only.

Run with: python -m scripts.casefiles.inspect_mh015_refusal
"""

import json

from doctrace.audit.goldset import load_gold_pairs

def main() -> None:
    results = {r["id"]: r for r in json.loads(open("data/processed/generation_report.json", encoding="utf-8").read())}
    gold_pairs = {p["id"]: p for p in load_gold_pairs()}

    r = results["mh_015"]
    p = gold_pairs["mh_015"]
    print("Q:", p["question"])
    print("\nExpected answer:", p["answer"])
    print("\nExpected source_chunks:", p["source_chunks"])
    print("\nLocal model's actual answer:", r["answer"])

if __name__ == "__main__":
    main()