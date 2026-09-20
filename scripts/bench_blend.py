"""
Retrieval benchmark across blend weights.

Scores semantic-only search, cross-encoder-only fused search (the original behaviour
later diagnosed as regressing) and several blended weights, so the chosen setting
rests on measured numbers.

Run with: python -m scripts.bench_blend
"""

import json

from doctrace.search.lexical import build_lexical_index
from doctrace.search.semantic import semantic_search
from doctrace.search.fusion import fused_search
from doctrace.audit.ranking import score_retriever

TOP_K = 5
WEIGHTS_TO_TEST = [1.0, 0.7, 0.5, 0.3]  # 1.0 = old pure-reranker behavior


def semantic_retriever(question: str, top_k: int) -> list[int]:
    results = semantic_search(question, top_k=top_k)
    return [r["id"] for r in results]


def make_fused_retriever(alpha: float):
    def _retriever(question: str, top_k: int) -> list[int]:
        out = fused_search(question, top_k=top_k, rerank_weight=alpha)
        return [r["id"] for r in out["results"]]
    return _retriever


def print_report(name: str, result: dict) -> None:
    print(f"\n{'=' * 60}\n{name} (top_k={result['top_k']})\n{'=' * 60}")
    for cat, m in result["by_category"].items():
        print(f"  {cat:12s} (n={m['n']:2d})  "
              f"hit_rate={m['hit_rate']:.3f}  recall={m['recall']:.3f}  "
              f"precision={m['precision']:.3f}  mrr={m['mrr']:.3f}")
    o = result["overall"]
    print(f"  {'OVERALL':12s} (n={o['n']:2d})  "
          f"hit_rate={o['hit_rate']:.3f}  recall={o['recall']:.3f}  "
          f"precision={o['precision']:.3f}  mrr={o['mrr']:.3f}")


def main() -> None:
    passages = json.loads(open("data/processed/passages.json", encoding="utf-8").read())
    print(f"Building BM25 index over {len(passages)} passages...")
    build_lexical_index(passages)
    print("Done.")

    all_results = {}

    print("\nRunning dense-only retrieval eval...")
    dense_result = score_retriever(semantic_retriever, top_k=TOP_K)
    print_report("DENSE-ONLY", dense_result)
    all_results["dense_only"] = dense_result

    for alpha in WEIGHTS_TO_TEST:
        label = f"HYBRID (rerank_weight={alpha})" + (" -- OLD pure-rescore behavior" if alpha == 1.0 else "")
        print(f"\nRunning {label}...")
        result = score_retriever(make_fused_retriever(alpha), top_k=TOP_K)
        print_report(label, result)
        all_results[f"hybrid_alpha_{alpha}"] = result

    with open("data/processed/retrieval_report.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved full results to data/processed/retrieval_report.json")


if __name__ == "__main__":
    main()