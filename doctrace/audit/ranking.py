"""
Ranking metrics for retrievers, scored against the hand-built gold set.

A retriever function is run over the single_hop and multi_hop questions. The
no_answer questions have no gold passages by design and are judged later, at the
answering stage, on whether the system declines instead of inventing something.

Per query, then averaged per category:
- Hit Rate@k: 1 if any gold passage is in the top k, else 0
- Recall@k: gold passages found in the top k / total gold passages
- Precision@k: gold passages found in the top k / k
- MRR: 1 / rank of the first gold passage found, 0 if none

Multi-hop questions have several gold passages, so their Recall@k tells you whether
retrieval surfaced everything needed to answer and not just one piece.
"""

from doctrace.audit.goldset import load_gold_pairs, attach_gold_ids, check_gold_pairs


def _score_query(ground_truth_ids: list[int], retrieved_ids: list[int], k: int) -> dict:
    top = retrieved_ids[:k]
    gold = set(ground_truth_ids)
    matched = [pid for pid in top if pid in gold]

    reciprocal_rank = 0.0
    for rank, pid in enumerate(top, start=1):
        if pid in gold:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": 1.0 if matched else 0.0,
        "recall": len(matched) / len(gold) if gold else 0.0,
        "precision": len(matched) / k if k else 0.0,
        "mrr": reciprocal_rank,
    }


def score_retriever(retriever_fn, top_k: int = 5,
                    categories: tuple[str, ...] = ("single_hop", "multi_hop")) -> dict:
    """Score retriever_fn(question, top_k) -> ordered passage ids (best first) on
    every gold question in `categories`.

    The result holds per-category and overall averages, plus the per-query rows
    (needed to find single failures that an average hides).
    """
    pairs = load_gold_pairs()
    check_gold_pairs(pairs)
    pairs = [p for p in attach_gold_ids(pairs) if p["category"] in categories]

    rows = []
    for pair in pairs:
        ids = retriever_fn(pair["question"], top_k)
        rows.append({
            "id": pair["id"],
            "category": pair["category"],
            **_score_query(pair["ground_truth_ids"], ids, top_k),
        })

    def _mean(subset, key):
        return sum(r[key] for r in subset) / len(subset) if subset else 0.0

    def _summary(subset):
        return {
            "n": len(subset),
            "hit_rate": _mean(subset, "hit"),
            "recall": _mean(subset, "recall"),
            "precision": _mean(subset, "precision"),
            "mrr": _mean(subset, "mrr"),
        }

    return {
        "top_k": top_k,
        "per_query": rows,
        "by_category": {cat: _summary([r for r in rows if r["category"] == cat]) for cat in categories},
        "overall": _summary(rows),
    }
