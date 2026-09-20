"""
Cross-check: RAGAS faithfulness against the claim-level grounding score.

Scores the answers saved in data/processed/generation_report.json with RAGAS and
lines the results up with our own metric for a direct methodology comparison.

Run with:
    python -m scripts.bench_ragas
"""

import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness

from doctrace.search.semantic import semantic_search
from doctrace.audit.goldset import load_gold_pairs
from doctrace.audit.ragas_bridge import LocalRagasJudge

TOP_K = 5
GENERATION_REPORT_PATH = "data/processed/generation_report.json"
CROSSCHECK_PATH = "data/processed/ragas_crosscheck.json"


def main() -> None:
    existing_results = {r["id"]: r for r in json.loads(open(GENERATION_REPORT_PATH, encoding="utf-8").read())}
    gold_pairs = {p["id"]: p for p in load_gold_pairs()}

    rows = []
    for pid, r in existing_results.items():
        if r["category"] == "no_answer" or r.get("abstained"):
            continue
        question = gold_pairs[pid]["question"]
        retrieved = semantic_search(question, top_k=TOP_K)
        rows.append({
            "id": pid,
            "question": question,
            "answer": r["answer"],
            "contexts": [c["text"] for c in retrieved],
            "grounding_score": r["faithfulness_score"],
        })

    print(f"Scoring {len(rows)} pairs with RAGAS faithfulness...", flush=True)

    dataset = Dataset.from_list([
        {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]}
        for r in rows
    ])

    print("Using LocalRagasJudge (Local Qwen2.5-3B-Instruct)", flush=True)
    faithfulness.llm = LocalRagasJudge()

    result = evaluate(dataset, metrics=[faithfulness])
    result_df = result.to_pandas()
    ragas_scores = result_df["faithfulness"].tolist()

    parse_failures = sum(1 for s in ragas_scores if s is None or (isinstance(s, float) and s != s))
    if parse_failures:
        print(f"\nNote: RAGAS itself failed to parse its own LLM output on "
              f"{parse_failures}/{len(ragas_scores)} pairs (returned None/NaN) -- "
              f"excluded from the averaged comparison below, not treated as 0.", flush=True)

    def _is_missing(x):
        return x is None or (isinstance(x, float) and x != x)

    comparison = []
    for row, ragas_score in zip(rows, ragas_scores):
        missing = _is_missing(ragas_score)
        comparison.append({
            "id": row["id"],
            "grounding_score": row["grounding_score"],
            "ragas_faithfulness_score": None if missing else ragas_score,
            "diff": None if missing else abs(row["grounding_score"] - ragas_score),
        })

    avg_custom = sum(c["grounding_score"] for c in comparison) / len(comparison)
    valid_ragas = [c["ragas_faithfulness_score"] for c in comparison if c["ragas_faithfulness_score"] is not None]
    avg_ragas = sum(valid_ragas) / len(valid_ragas) if valid_ragas else 0.0

    print(f"\nAverage custom faithfulness: {avg_custom:.3f}", flush=True)
    print(f"Average RAGAS faithfulness:  {avg_ragas:.3f}", flush=True)

    biggest_diffs = sorted([c for c in comparison if c["diff"] is not None],
                            key=lambda c: c["diff"], reverse=True)[:5]
    print("\nBiggest disagreements (custom vs RAGAS):", flush=True)
    for c in biggest_diffs:
        print(f"  {c['id']}: custom={c['grounding_score']:.2f}  "
              f"ragas={c['ragas_faithfulness_score']:.2f}  diff={c['diff']:.2f}", flush=True)

    with open(CROSSCHECK_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved full comparison to {CROSSCHECK_PATH}", flush=True)


if __name__ == "__main__":
    main()