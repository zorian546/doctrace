"""
Case file: which no_answer questions did the local model answer with an
invention instead of declining, and what does mh_025's unexpected refusal look
like? Works from saved results; no model calls.

Run with: python -m scripts.casefiles.inspect_local_model_regressions
"""

import json

from doctrace.audit.goldset import load_gold_pairs

REPORT_PATH = "data/processed/generation_report.json"


def main() -> None:
    results = {r["id"]: r for r in json.loads(open(REPORT_PATH, encoding="utf-8").read())}
    gold_pairs = {p["id"]: p for p in load_gold_pairs()}

    print("=" * 70)
    print("no_answer pairs that were HALLUCINATED instead of refused:")
    print("=" * 70)
    for pid, r in results.items():
        if r["category"] == "no_answer" and not r.get("correctly_refused"):
            print(f"\n[{pid}]")
            print(f"  Q: {gold_pairs[pid]['question']}")
            print(f"  A (local model): {r['answer']}")

    print(f"\n{'=' * 70}")
    print("mh_025's unexpected refusal (should have been answerable):")
    print("=" * 70)
    r = results.get("mh_025")
    if r:
        print(f"\n  Q: {gold_pairs['mh_025']['question']}")
        print(f"  A (local model): {r['answer']}")
        print(f"  Expected answer: {gold_pairs['mh_025']['answer']}")


if __name__ == "__main__":
    main()