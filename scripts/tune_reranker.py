"""
Fine-tune the cross-encoder on this corpus.

Starts from ms-marco-MiniLM-L-6-v2 and trains on gold questions from the docs:
- positives: (question, gold passage text) -> label 1.0
- hard negatives: (question, high-ranking non-gold passage text) -> label 0.0

Ranking quality is measured before and after, which addresses the vocabulary-overlap
bias described in ENGINEERING_NOTES.md.

Run with:
    python -m scripts.tune_reranker
"""

import json
import os
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

from doctrace.audit.ranking import _score_query
from doctrace.audit.goldset import load_gold_pairs, attach_gold_ids
from doctrace.search.lexical import lexical_search, build_lexical_index
from doctrace.search.semantic import semantic_search
from doctrace.search.fusion import rrf_merge

STOCK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TUNED_MODEL_DIR = "models/docs-reranker-minilm"
REPORT_PATH = Path("data/processed/reranker_tuning_report.json")
SEED = 42


def split_questions(gold_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic 80/20 split by QUESTION, so no question appears in both sets."""
    shuffled = list(gold_rows)
    random.Random(SEED).shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    return shuffled[:cut], shuffled[cut:]


def assemble_training_pairs(train_qa: list[dict], val_qa: list[dict], passages: list[dict],
                            hard_negatives_per_query: int = 4) -> tuple[list, list]:
    """Build positive and hard-negative pairs from the merged semantic and BM25 pools."""

    def _create_examples(qa_subset: list[dict]) -> list[InputExample]:
        examples = []
        for pair in qa_subset:
            q = pair["question"]
            gt_ids = set(pair["ground_truth_ids"])
            
            # 1. Positive examples
            for gid in gt_ids:
                examples.append(InputExample(texts=[q, passages[gid]["text"]], label=1.0))

            # 2. Mine hard negatives from dense + BM25 pool
            dense_cand = semantic_search(q, top_k=15)
            bm25_cand = lexical_search(q, top_k=15)
            fused = rrf_merge([dense_cand, bm25_cand], k=60)
            
            negatives = [c for c in fused if c["id"] not in gt_ids][:hard_negatives_per_query]
            for neg in negatives:
                examples.append(InputExample(texts=[q, neg["text"]], label=0.0))
        return examples

    return _create_examples(train_qa), _create_examples(val_qa)


def score_reranker(model: CrossEncoder, gold_rows: list[dict], passages: list[dict], top_k: int = 5) -> dict:
    """Evaluate retrieval precision, recall, and MRR using the given reranker."""
    per_query = []
    for pair in gold_rows:
        q = pair["question"]
        gt_ids = set(pair["ground_truth_ids"])

        dense_cand = semantic_search(q, top_k=20)
        bm25_cand = lexical_search(q, top_k=20)
        fused = rrf_merge([dense_cand, bm25_cand], k=60)[:20]

        pairs = [(q, c["text"]) for c in fused]
        scores = model.predict(pairs)
        scored_candidates = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_ids = [c["id"] for c, _ in scored_candidates]

        m = _score_query(list(gt_ids), top_ids, top_k)
        per_query.append({"id": pair["id"], **m})

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "hit_rate": _avg(per_query, "hit"),
        "recall": _avg(per_query, "recall"),
        "precision": _avg(per_query, "precision"),
        "mrr": _avg(per_query, "mrr"),
    }


def main() -> None:
    print("=" * 85)
    print("RERANKER TUNING ON THE DOCS CORPUS")
    print("=" * 85)

    # 1. Load data
    passages = json.loads(open("data/processed/passages.json", encoding="utf-8").read())
    build_lexical_index(passages)
    gold_pairs = load_gold_pairs()
    gold_rows = [p for p in attach_gold_ids(gold_pairs, passages) if p["category"] in ("single_hop", "multi_hop")]

    print(f"Loaded {len(passages)} passages and {len(gold_rows)} gold pairs.")

    # 2. Mine hard negatives & build datasets
    print("Mining hard negatives and building train/val datasets...")
    train_qa, val_qa = split_questions(gold_rows)
    print(f"Split: {len(train_qa)} training questions, {len(val_qa)} held-out questions.")
    train_samples, val_samples = assemble_training_pairs(train_qa, val_qa, passages)
    print(f"Generated {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Baseline Evaluation (Before Fine-Tuning)
    print("\n[1/3] Evaluating Baseline Off-the-Shelf Reranker...")
    base_model = CrossEncoder(STOCK_MODEL, device=device)
    baseline_metrics = score_reranker(base_model, gold_rows, passages)
    baseline_heldout = score_reranker(base_model, val_qa, passages)
    print(f"  Baseline (all questions): Hit@5 = {baseline_metrics['hit_rate']:.3f} | Recall@5 = {baseline_metrics['recall']:.3f} | MRR = {baseline_metrics['mrr']:.3f}")

    # 4. Train / Fine-tune Reranker
    print("\n[2/3] Fine-tuning Cross-Encoder on FastAPI Technical Documentation...")
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
    
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_samples, name="holdout")

    os.makedirs(TUNED_MODEL_DIR, exist_ok=True)
    base_model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=3,
        evaluation_steps=20,
        warmup_steps=10,
        output_path=TUNED_MODEL_DIR,
        show_progress_bar=True,
    )
    print(f"Fine-tuned model saved to {TUNED_MODEL_DIR}")

    # 5. Post-Fine-Tuning Evaluation
    print("\n[3/3] Evaluating Fine-Tuned Domain-Adapted Reranker...")
    finetuned_model = CrossEncoder(TUNED_MODEL_DIR, device=device)
    finetuned_metrics = score_reranker(finetuned_model, gold_rows, passages)
    finetuned_heldout = score_reranker(finetuned_model, val_qa, passages)

    # 6. Summary Comparison
    print("\n" + "=" * 85)
    print("RERANKER TUNING, ALL 65 QUESTIONS (includes training questions)")
    print("=" * 85)
    print(f"{'Metric':<20} | {'Baseline (Off-the-Shelf)':<25} | {'Fine-Tuned (Domain-Adapted)':<25} | {'Delta':<10}")
    print("-" * 85)
    for m_key, m_name in [("hit_rate", "Hit Rate@5"), ("recall", "Recall@5"), ("precision", "Precision@5"), ("mrr", "MRR")]:
        b = baseline_metrics[m_key]
        f = finetuned_metrics[m_key]
        delta = f - b
        print(f"{m_name:<20} | {b:<25.3f} | {f:<25.3f} | {delta:+10.3f}")
    print("=" * 85)
    print(f"\nHELD-OUT questions only (n={len(val_qa)}, never seen in training) -- the number to quote:")
    for m_key, m_name in [("hit_rate", "Hit Rate@5"), ("recall", "Recall@5"), ("precision", "Precision@5"), ("mrr", "MRR")]:
        b, f = baseline_heldout[m_key], finetuned_heldout[m_key]
        print(f"{m_name:<20} | {b:<25.3f} | {f:<25.3f} | {f - b:+10.3f}")
    print("(The all-questions table above includes training questions and overstates the gain.)")

    comparison_payload = {
        "baseline_model": STOCK_MODEL,
        "finetuned_model": TUNED_MODEL_DIR,
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "heldout_question_count": len(val_qa),
        "baseline_heldout_metrics": baseline_heldout,
        "finetuned_heldout_metrics": finetuned_heldout,
        "training_samples_count": len(train_samples),
        "validation_samples_count": len(val_samples),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
    print(f"\nSaved fine-tuning results report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
