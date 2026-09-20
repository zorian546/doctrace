"""
Retrieval benchmark across search configurations:
- semantic-only (BGE-M3)
- keyword-only (BM25 Okapi)
- fused, cross-encoder only (rerank_weight = 1.0)
- fused, blended (rerank_weight = 0.7)
- fused, balanced (rerank_weight = 0.5)

Scored on the hand-built gold set (gold/gold_pairs.json) for hit rate, recall,
precision and MRR.

Run with:
    python -m scripts.bench_retrievers
"""

import json
from pathlib import Path

from doctrace.audit.ranking import score_retriever
from doctrace.audit.goldset import load_gold_pairs, check_gold_pairs
from doctrace.search.lexical import lexical_search, build_lexical_index
from doctrace.search.semantic import semantic_search
from doctrace.search.fusion import fused_search

TOP_K = 5
REPORT_PATH = Path("data/processed/ablation_quick.json")


def semantic_retriever(question: str, top_k: int) -> list[int]:
    results = semantic_search(question, top_k=top_k)
    return [r["id"] for r in results]


def lexical_retriever(question: str, top_k: int) -> list[int]:
    results = lexical_search(question, top_k=top_k)
    return [r["id"] for r in results]


def make_fused_retriever(alpha: float):
    def _retriever(question: str, top_k: int) -> list[int]:
        out = fused_search(question, top_k=top_k, rerank_weight=alpha)
        return [r["id"] for r in out["results"]]
    return _retriever


def format_table(configs: dict[str, dict]) -> str:
    lines = []
    lines.append(f"{'Configuration':<35} | {'Hit Rate@5':<10} | {'Recall@5':<10} | {'Precision@5':<12} | {'MRR':<8}")
    lines.append("-" * 85)
    for name, res in configs.items():
        o = res["overall"]
        lines.append(
            f"{name:<35} | {o['hit_rate']:<10.3f} | {o['recall']:<10.3f} | {o['precision']:<12.3f} | {o['mrr']:<8.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    print("=" * 85)
    print("RAG SYSTEM UNIFIED ABLATION SWEEP")
    print("=" * 85)

    # 1. Validate gold pairs
    gold_pairs = load_gold_pairs()
    check_gold_pairs(gold_pairs)
    print(f"Loaded and validated {len(gold_pairs)} gold pairs from gold/gold_pairs.json\n")

    # 2. Ensure BM25 index is built
    passages_path = Path("data/processed/passages.json")
    if not passages_path.exists():
        raise FileNotFoundError("data/processed/passages.json not found. Run python -m scripts.build_index first.")
    passages = json.loads(passages_path.read_text(encoding="utf-8"))
    build_lexical_index(passages)
    print(f"BM25 index built over {len(passages)} passages.\n")

    ablation_results = {}

    # 3. Dense-only ablation
    print("[1/5] Evaluating Dense-Only (BGE-M3)...")
    ablation_results["Dense-Only (BGE-M3)"] = score_retriever(semantic_retriever, top_k=TOP_K)

    # 4. BM25-only ablation
    print("[2/5] Evaluating BM25-Only (Okapi)...")
    ablation_results["BM25-Only (Okapi)"] = score_retriever(lexical_retriever, top_k=TOP_K)

    # 5. Hybrid pure-rescore (alpha = 1.0)
    print("[3/5] Evaluating Hybrid (alpha=1.0, pure cross-encoder rescore)...")
    ablation_results["Hybrid (alpha=1.0, pure rescore)"] = score_retriever(
        make_fused_retriever(1.0), top_k=TOP_K
    )

    # 6. Hybrid blended (alpha = 0.7)
    print("[4/5] Evaluating Hybrid (alpha=0.7, blended RRF+rescore)...")
    ablation_results["Hybrid (alpha=0.7, blended)"] = score_retriever(
        make_fused_retriever(0.7), top_k=TOP_K
    )

    # 7. Hybrid balanced (alpha = 0.5)
    print("[5/5] Evaluating Hybrid (alpha=0.5, balanced)...")
    ablation_results["Hybrid (alpha=0.5, balanced)"] = score_retriever(
        make_fused_retriever(0.5), top_k=TOP_K
    )

    # 8. Display results
    print("\n" + "=" * 85)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 85)
    print(format_table(ablation_results))
    print("=" * 85)

    # 9. Save to disk
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(ablation_results, indent=2), encoding="utf-8")
    print(f"\nDetailed ablation metrics saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
