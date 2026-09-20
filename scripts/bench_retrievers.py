"""
Retrieval benchmark: every search configuration, scored on the gold set.

Configurations
- semantic only (BGE-M3 vectors)
- keyword only (BM25 Okapi)
- fused at several cross-encoder weights, where 1.0 means the cross-encoder alone and
  lower values keep more of the keyword+semantic agreement

Metrics are Hit Rate@5, Recall@5, Precision@5 and MRR over the 65 gold questions that
have source passages, reported per category and overall. Writes
data/processed/retrieval_report.json, which the dashboard and /v1/scorecard read.

Needs the vector store, so run scripts/build_index.py --sync-qdrant first.

Run with:
    python -m scripts.bench_retrievers
"""

import json
from pathlib import Path

from doctrace.audit.goldset import check_gold_pairs, load_gold_pairs
from doctrace.audit.ranking import score_retriever
from doctrace.search.fusion import fused_search
from doctrace.search.lexical import build_lexical_index, lexical_search
from doctrace.search.semantic import semantic_search

TOP_K = 5
RERANK_WEIGHTS = [1.0, 0.7, 0.5, 0.3]   # 1.0 = cross-encoder alone
PASSAGES_PATH = Path("data/processed/passages.json")
REPORT_PATH = Path("data/processed/retrieval_report.json")


def semantic_retriever(question: str, top_k: int) -> list[int]:
    return [r["id"] for r in semantic_search(question, top_k=top_k)]


def lexical_retriever(question: str, top_k: int) -> list[int]:
    return [r["id"] for r in lexical_search(question, top_k=top_k)]


def make_fused_retriever(weight: float):
    def _retriever(question: str, top_k: int) -> list[int]:
        out = fused_search(question, top_k=top_k, rerank_weight=weight)
        return [r["id"] for r in out["results"]]
    return _retriever


def print_report(name: str, result: dict) -> None:
    print(f"\n{'=' * 64}\n{name} (top_k={result['top_k']})\n{'=' * 64}")
    rows = list(result["by_category"].items()) + [("OVERALL", result["overall"])]
    for label, m in rows:
        print(f"  {label:12s} (n={m['n']:2d})  hit_rate={m['hit_rate']:.3f}  "
              f"recall={m['recall']:.3f}  precision={m['precision']:.3f}  mrr={m['mrr']:.3f}")


def main() -> None:
    pairs = load_gold_pairs()
    check_gold_pairs(pairs)
    passages = json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))
    build_lexical_index(passages)
    print(f"Scoring {len(pairs)} gold questions over {len(passages)} passages.")

    configs = [("semantic_only", "SEMANTIC ONLY", semantic_retriever),
               ("lexical_only", "KEYWORD ONLY (BM25)", lexical_retriever)]
    for weight in RERANK_WEIGHTS:
        note = " -- cross-encoder alone" if weight == 1.0 else ""
        configs.append((f"fused_w{weight}", f"FUSED (rerank_weight={weight}){note}",
                        make_fused_retriever(weight)))

    results = {}
    for i, (key, label, retriever) in enumerate(configs, start=1):
        print(f"\n[{i}/{len(configs)}] {label} ...")
        results[key] = score_retriever(retriever, top_k=TOP_K)
        print_report(label, results[key])

    print("\n" + "=" * 64)
    print(f"{'Configuration':<28} | {'Hit@5':<7} | {'Recall@5':<8} | {'Prec@5':<7} | {'MRR':<7}")
    print("-" * 64)
    for key, _, _ in configs:
        o = results[key]["overall"]
        print(f"{key:<28} | {o['hit_rate']:<7.3f} | {o['recall']:<8.3f} | "
              f"{o['precision']:<7.3f} | {o['mrr']:<7.3f}")
    print("=" * 64)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
