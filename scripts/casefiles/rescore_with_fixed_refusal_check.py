"""
Case file: re-score saved generation results with a sturdier refusal detector.
Nothing is regenerated, so no GPU is needed.

Run with: python -m scripts.casefiles.rescore_with_fixed_refusal_check
"""

import json

REPORT_PATH = "data/processed/generation_report.json"
ABSTAIN_MARKER = "don't have enough information in the provided context"


def is_refusal_robust(answer: str) -> bool:
    return ABSTAIN_MARKER in answer.lower()


def main() -> None:
    results = json.loads(open(REPORT_PATH, encoding="utf-8").read())

    corrections = []
    for r in results:
        answer = r["answer"]
        robust_refusal = is_refusal_robust(answer)

        if r["category"] == "no_answer":
            old = r.get("correctly_refused")
            if old != robust_refusal:
                corrections.append((r["id"], "correctly_refused", old, robust_refusal))
            r["correctly_refused"] = robust_refusal
        else:
            old = r.get("abstained", False)
            if old != robust_refusal and robust_refusal:
                corrections.append((r["id"], "abstained", old, robust_refusal))
                r["abstained"] = True
                r["grounding_score"] = None

    print(f"Corrections made: {len(corrections)}")
    for pid, field, old, new in corrections:
        print(f"  {pid}: {field} {old} -> {new}")

    for cat in ("single_hop", "multi_hop"):
        rows = [r for r in results if r["category"] == cat and not r.get("abstained")]
        scores = [r["grounding_score"] for r in rows]
        avg = sum(scores) / len(scores) if scores else 0.0
        refused = sum(1 for r in results if r["category"] == cat and r.get("abstained"))
        print(f"\n{cat}: avg faithfulness = {avg:.3f} (n={len(rows)}, {refused} unexpectedly refused)")

    no_answer_rows = [r for r in results if r["category"] == "no_answer"]
    refusal_accuracy = sum(r["correctly_refused"] for r in no_answer_rows) / len(no_answer_rows)
    print(f"no_answer: refusal accuracy = {refusal_accuracy:.3f} (n={len(no_answer_rows)})")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nCorrected results saved back to {REPORT_PATH}")


if __name__ == "__main__":
    main()