"""
Smoke test for answering plus the grounding score.

Runs grounded answering and claim-level scoring on sample questions, then an
adversarial pass that scores an answer against unrelated context to confirm
unsupported claims are actually caught.

Run with:
    python -m scripts.smoke_grounding
"""

import json

from doctrace.search.lexical import build_lexical_index
from doctrace.search.semantic import semantic_search
from doctrace.answering.responder import answer_question
from doctrace.audit.grounding import grounding_score
from doctrace.audit.goldset import load_gold_pairs

SAMPLE_IDS = ["sh_001", "sh_005", "mh_002", "na_001"]
TOP_K = 5


def main() -> None:
    passages = json.loads(open("data/processed/passages.json", encoding="utf-8").read())
    build_lexical_index(passages)

    gold_pairs = {p["id"]: p for p in load_gold_pairs()}

    for pid in SAMPLE_IDS:
        pair = gold_pairs[pid]
        print(f"\n{'=' * 70}\n{pid} [{pair['category']}]\nQ: {pair['question']}\n{'=' * 70}")

        retrieved = semantic_search(pair["question"], top_k=TOP_K)
        gen = answer_question(pair["question"], retrieved)
        print(f"\nGenerated answer:\n  {gen['answer']}")

        result = grounding_score(gen["answer"], retrieved)
        if result["abstained"]:
            print("\nFaithfulness: N/A (model refused / made no claims)")
        else:
            print(f"\nFaithfulness score: {result['score']:.2f} "
                  f"({sum(c['supported'] for c in result['claims'])}/{len(result['claims'])} claims supported)")
            for c in result["claims"]:
                mark = "✓" if c["supported"] else "✗"
                print(f"  [{mark}] {c['claim']}")

    # Adversarial check: take sh_001's generated answer, but score it
    # against completely unrelated context (a no_answer-category topic)
    # to confirm the scorer actually flags unsupported claims.
    print(f"\n{'=' * 70}\nADVERSARIAL CHECK: sh_001's answer scored against WRONG context\n{'=' * 70}")
    sh_001 = gold_pairs["sh_001"]
    retrieved_correct = semantic_search(sh_001["question"], top_k=TOP_K)
    gen = answer_question(sh_001["question"], retrieved_correct)
    print(f"Answer (generated from CORRECT context): {gen['answer']}")

    wrong_context = semantic_search("How do I use WebSockets in FastAPI?", top_k=TOP_K)
    result = grounding_score(gen["answer"], wrong_context)
    print(f"\nScored against UNRELATED context -- expect a LOW score, not 1.0:")
    if result["abstained"]:
        print("  N/A (no claims extracted)")
    else:
        print(f"  Faithfulness score: {result['score']:.2f}")
        for c in result["claims"]:
            mark = "✓" if c["supported"] else "✗"
            print(f"  [{mark}] {c['claim']}")


if __name__ == "__main__":
    main()