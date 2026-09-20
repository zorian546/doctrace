"""
Case file: unsupported claims in multi-hop answers.

Reads the saved generation_report.json (no new model calls) and prints the specific
unsupported claims behind every multi_hop score under 1.0, so ENGINEERING_NOTES.md can quote
real failures instead of just the 0.914 average.

Run with: python -m scripts.casefiles.inspect_multihop_failures
"""

import json

REPORT_PATH = "data/processed/generation_report.json"


def main() -> None:
    results = json.loads(open(REPORT_PATH, encoding="utf-8").read())
    multihop = [r for r in results if r["category"] == "multi_hop" and not r.get("abstained")]

    below_perfect = [r for r in multihop if r["grounding_score"] < 1.0]
    print(f"{len(below_perfect)}/{len(multihop)} multi_hop pairs scored below 1.0\n")

    for r in sorted(below_perfect, key=lambda x: x["grounding_score"]):
        print(f"{'=' * 70}")
        print(f"{r['id']}  faithfulness={r['grounding_score']:.2f}")
        print(f"{'=' * 70}")
        for c in r["claims"]:
            if not c["supported"]:
                print(f"  [UNSUPPORTED] {c['claim']}")
        print(f"\nFull answer:\n  {r['answer']}\n")


if __name__ == "__main__":
    main()