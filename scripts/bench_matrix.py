"""
Ablation grid across three axes:
1. Splitting: heading-aware vs fixed 500-word vs fixed 200-word windows.
2. Encoder: BAAI/bge-m3 (1024-d) vs sentence-transformers/all-MiniLM-L6-v2 (384-d).
3. Search mode: semantic-only vs keyword-only vs fused (blend weight 0.7).

Metrics are Hit Rate@5, Recall@5, Precision@5 and MRR over the 65 questions with gold passages.

Run with:
    python -m scripts.bench_matrix
"""

import json
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from doctrace.audit.ranking import _score_query
from doctrace.audit.goldset import load_gold_pairs, attach_gold_ids, check_gold_pairs
from doctrace.corpus.splitter import split_by_headings, split_fixed_window
from doctrace.corpus.fetcher import read_docs
from doctrace.search.lexical import lexical_search, build_lexical_index
from doctrace.search.semantic import semantic_search
from doctrace.search.fusion import fused_search

TOP_K = 5
REPORT_PATH = Path("data/processed/ablation_grid.json")


def score_inmemory_semantic(passages: list[dict], model_name: str, gold_pairs: list[dict]) -> dict:
    """Evaluate in-memory dense retrieval for a given passage list and encoder."""
    print(f"  Encoding {len(passages)} passages with {model_name}...")
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    
    texts = [c["text"] for c in passages]
    passage_vectors = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)

    per_query = []
    for pair in gold_pairs:
        q_emb = model.encode([pair["question"]], normalize_embeddings=True)[0]
        sims = np.dot(passage_vectors, q_emb)
        top_indices = np.argsort(sims)[::-1][:TOP_K].tolist()
        
        gt_ids = pair.get("ground_truth_ids", [])
        m = _score_query(gt_ids, top_indices, TOP_K)
        per_query.append({"id": pair["id"], "category": pair["category"], **m})

    # Cleanup GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "hit_rate": _avg(per_query, "hit"),
        "recall": _avg(per_query, "recall"),
        "precision": _avg(per_query, "precision"),
        "mrr": _avg(per_query, "mrr"),
        "n": len(per_query),
    }


def score_fixed_window_lexical(docs: list[dict], gold_pairs: list[dict], window_size: int, overlap: int) -> dict:
    """Evaluate BM25 over fixed-size passages by source-path matching against ground truth source files."""
    window_passages = []
    for d in docs:
        window_passages.extend(split_fixed_window(d["text"], d["source_path"], size=window_size, overlap=overlap))

    build_lexical_index(window_passages)

    per_query = []
    for pair in gold_pairs:
        results = lexical_search(pair["question"], top_k=TOP_K)
        retrieved_sources = [r["source_path"] for r in results]
        
        # Ground truth source paths
        gt_sources = set()
        for ref in pair["source_chunks"]:
            src = ref.split("#")[0]
            gt_sources.add(src)

        hits = [s for s in retrieved_sources if s in gt_sources]
        hit = 1.0 if hits else 0.0
        recall = len(set(hits)) / len(gt_sources) if gt_sources else 0.0
        precision = len(hits) / TOP_K
        mrr = 0.0
        for rank, s in enumerate(retrieved_sources, start=1):
            if s in gt_sources:
                mrr = 1.0 / rank
                break

        per_query.append({"id": pair["id"], "hit": hit, "recall": recall, "precision": precision, "mrr": mrr})

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "hit_rate": _avg(per_query, "hit"),
        "recall": _avg(per_query, "recall"),
        "precision": _avg(per_query, "precision"),
        "mrr": _avg(per_query, "mrr"),
        "n": len(per_query),
        "total_passages": len(window_passages),
    }


def main() -> None:
    print("=" * 85)
    print("STARTING FULL MULTI-DIMENSIONAL ABLATION SWEEP")
    print("=" * 85)

    docs = read_docs()
    raw_qa = load_gold_pairs()
    check_gold_pairs(raw_qa)

    passages_path = Path("data/processed/passages.json")
    heading_passages = json.loads(passages_path.read_text(encoding="utf-8")) if passages_path.exists() else []
    if not heading_passages:
        for d in docs:
            heading_passages.extend(split_by_headings(d["text"], d["source_path"]))

    gold_rows = [p for p in attach_gold_ids(raw_qa, heading_passages) if p["category"] in ("single_hop", "multi_hop")]
    print(f"Evaluated on {len(gold_rows)} gold pairs across {len(heading_passages)} header passages.\n")

    matrix = {}

    # 1. Baseline: heading-aware passages + BGE-M3 (Qdrant semantic)
    print("[1/6] Evaluating Header-Aware Passages + BGE-M3 (Qdrant Dense)...")
    from scripts.bench_retrievers import semantic_retriever
    from doctrace.audit.ranking import score_retriever
    d_res = score_retriever(semantic_retriever, top_k=TOP_K)
    matrix["Header-Aware | BGE-M3 | Dense-Only"] = d_res["overall"]

    # 2. Header Passages + BM25-Only
    print("[2/6] Evaluating Header-Aware Passages + BM25-Only...")
    build_lexical_index(heading_passages)
    from scripts.bench_retrievers import lexical_retriever
    bm_res = score_retriever(lexical_retriever, top_k=TOP_K)
    matrix["Header-Aware | BM25-Only"] = bm_res["overall"]

    # 3. Header Passages + BGE-M3 + Hybrid (alpha=0.7)
    print("[3/6] Evaluating Header-Aware Passages + BGE-M3 | Hybrid (alpha=0.7)...")
    from scripts.bench_retrievers import make_fused_retriever
    hyb_res = score_retriever(make_fused_retriever(0.7), top_k=TOP_K)
    matrix["Header-Aware | BGE-M3 | Hybrid (alpha=0.7)"] = hyb_res["overall"]

    # 4. Header Passages + all-MiniLM-L6-v2 (encoder ablation)
    print("[4/6] Evaluating Header-Aware Passages + all-MiniLM-L6-v2 (Dense-Only)...")
    minilm_res = score_inmemory_semantic(heading_passages, "sentence-transformers/all-MiniLM-L6-v2", gold_rows)
    matrix["Header-Aware | all-MiniLM-L6-v2 | Dense-Only"] = minilm_res

    # 5. Fixed-Size 500w Passages + BM25
    print("[5/6] Evaluating Fixed-Size 500w Passages | BM25...")
    f500_res = score_fixed_window_lexical(docs, gold_rows, window_size=500, overlap=50)
    matrix["Fixed-Size (500w) | BM25-Only"] = f500_res

    # 6. Fixed-Size 200w Passages + BM25
    print("[6/6] Evaluating Fixed-Size 200w Passages | BM25...")
    f200_res = score_fixed_window_lexical(docs, gold_rows, window_size=200, overlap=25)
    matrix["Fixed-Size (200w) | BM25-Only"] = f200_res

    # Restore clean BM25 index over header passages
    build_lexical_index(heading_passages)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Configuration':<45} | {'Hit@5':<8} | {'Recall@5':<9} | {'Prec@5':<8} | {'MRR':<8}")
    print("-" * 90)
    for name, r in matrix.items():
        print(f"{name:<45} | {r['hit_rate']:<8.3f} | {r['recall']:<9.3f} | {r['precision']:<8.3f} | {r['mrr']:<8.3f}")
    print("=" * 90)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"\nFull ablation matrix saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
