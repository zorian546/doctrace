"""
Generation and grounding benchmark, run in two stages so the retrieval model and
the generation model never share the GPU (Task Manager showed both together used
all 8GB of VRAM).

Run with: python -m scripts.bench_generation
"""

import json
from pathlib import Path

from doctrace.search.semantic import semantic_search
from doctrace.vectors.encoder import release_gpu
from doctrace.answering.responder import answer_question
from doctrace.audit.grounding import grounding_score, is_abstention
from doctrace.audit.goldset import load_gold_pairs, check_gold_pairs

TOP_K = 5
RETRIEVAL_STASH_PATH = Path("data/processed/retrieval_stash.json")
REPORT_PATH = Path("data/processed/generation_report.json")


def _load_json_dict(path: Path) -> dict:
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        return {r["id"]: r for r in rows} if isinstance(rows, list) else rows
    return {}


def _save_dict_as_list(path: Path, d: dict) -> None:
    path.write_text(json.dumps(list(d.values()), indent=2), encoding="utf-8")


def stage_retrieval(gold_pairs: list[dict]) -> dict:
    stash = _load_json_dict(RETRIEVAL_STASH_PATH)
    if stash:
        print(f"Retrieval stash: {len(stash)}/{len(gold_pairs)} pairs already cached.")

    for i, pair in enumerate(gold_pairs, start=1):
        if pair["id"] in stash:
            continue
        print(f"[retrieval {i}/{len(gold_pairs)}] {pair['id']}")
        retrieved = semantic_search(pair["question"], top_k=TOP_K)
        stash[pair["id"]] = {"id": pair["id"], "retrieved": retrieved}
        _save_dict_as_list(RETRIEVAL_STASH_PATH, stash)

    print("Retrieval stage finished. Releasing the encoder's GPU memory...")
    release_gpu()
    return stash


def stage_generation(gold_pairs: list[dict], retrieval_stash: dict) -> None:
    results_by_id = _load_json_dict(REPORT_PATH)
    if results_by_id:
        print(f"Resuming generation: {len(results_by_id)}/{len(gold_pairs)} pairs already done.")

    for i, pair in enumerate(gold_pairs, start=1):
        if pair["id"] in results_by_id:
            continue

        print(f"[generation {i}/{len(gold_pairs)}] {pair['id']} [{pair['category']}]")
        retrieved = retrieval_stash[pair["id"]]["retrieved"]
        gen = answer_question(pair["question"], retrieved)
        answer = gen["answer"]
        print(f"    generated answer ({len(answer)} chars)")

        row = {"id": pair["id"], "category": pair["category"], "answer": answer}

        if pair["category"] == "no_answer":
            row["correctly_refused"] = is_abstention(answer)
        else:
            print("    scoring faithfulness...")
            faith = grounding_score(answer, retrieved)
            row["faithfulness_score"] = faith["score"]
            row["abstained"] = faith["abstained"]
            row["claims"] = faith["claims"]
            if faith["abstained"]:
                print(f"    WARNING: {pair['id']} unexpectedly refused")
            else:
                print(f"    faithfulness: {faith['score']:.2f} ({len(faith['claims'])} claims)")

        results_by_id[pair["id"]] = row
        _save_dict_as_list(REPORT_PATH, results_by_id)

    results = list(results_by_id.values())
    for cat in ("single_hop", "multi_hop"):
        rows = [r for r in results if r["category"] == cat and not r.get("abstained")]
        scores = [r["faithfulness_score"] for r in rows]
        avg = sum(scores) / len(scores) if scores else 0.0
        refused = sum(1 for r in results if r["category"] == cat and r.get("abstained"))
        print(f"\n{cat}: avg faithfulness = {avg:.3f} (n={len(rows)}, {refused} unexpectedly refused)")

    no_answer_rows = [r for r in results if r["category"] == "no_answer"]
    refusal_accuracy = sum(r["correctly_refused"] for r in no_answer_rows) / len(no_answer_rows)
    print(f"no_answer: refusal accuracy = {refusal_accuracy:.3f} (n={len(no_answer_rows)})")
    print(f"\nSaved to {REPORT_PATH}")


def main() -> None:
    gold_pairs = load_gold_pairs()
    check_gold_pairs(gold_pairs)
    retrieval_stash = stage_retrieval(gold_pairs)
    stage_generation(gold_pairs, retrieval_stash)


if __name__ == "__main__":
    main()