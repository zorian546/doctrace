"""
Case file: why does the fused pipeline trail semantic-only on recall?

Hypothesis: rrf_merge() ranks every merged candidate, but fused_search() cuts the
list to pool_size=20 before the cross-encoder sees it. A passage that ranks well
semantically yet is missing from the keyword top-20 could be squeezed out of the
merged top-20 by RRF arithmetic alone, and the cross-encoder would never get to rescore it.

For each single_hop/multi_hop question where semantic-only found a gold passage in
its top 5 and the fused pipeline did not, this prints:
- whether the gold passage is anywhere in the full, untruncated RRF ranking
- and if so, whether its rank is inside or beyond the pool_size=20 cutoff

Run with: python -m scripts.casefiles.diagnose_fusion_regression
"""

import json

from doctrace.search.lexical import build_lexical_index, lexical_search
from doctrace.search.semantic import semantic_search
from doctrace.search.fusion import rrf_merge, rescore
from doctrace.audit.goldset import load_gold_pairs, attach_gold_ids

TOP_K = 5
POOL_SIZE = 20


def main() -> None:
    passages = json.loads(open("data/processed/passages.json", encoding="utf-8").read())
    build_lexical_index(passages)

    gold_pairs = load_gold_pairs()
    resolved = attach_gold_ids(gold_pairs)
    resolved = [p for p in resolved if p["category"] in ("single_hop", "multi_hop")]

    cutoff_truncation_count = 0
    reranker_demotion_count = 0
    not_in_fusion_at_all_count = 0
    total_regressions = 0

    for pair in resolved:
        question = pair["question"]
        gt_ids = set(pair["ground_truth_ids"])

        dense_results = semantic_search(question, top_k=POOL_SIZE)
        dense_top5_ids = [r["id"] for r in dense_results[:TOP_K]]
        dense_hit = bool(gt_ids & set(dense_top5_ids))

        if not dense_hit:
            continue  # only care about cases dense-only got right

        bm25_results = lexical_search(question, top_k=POOL_SIZE)
        fused_full = rrf_merge([dense_results, bm25_results])
        fused_full_ids = [r["id"] for r in fused_full]

        hybrid_top5 = rescore(question, fused_full[:POOL_SIZE], top_k=TOP_K)
        hybrid_top5_ids = [r["id"] for r in hybrid_top5]
        hybrid_hit = bool(gt_ids & set(hybrid_top5_ids))

        if hybrid_hit:
            continue  # not a regression case

        total_regressions += 1
        print(f"\n--- REGRESSION: {pair['id']} ---")
        print(f"    Q: {question}")

        for gt_id in gt_ids:
            if gt_id in fused_full_ids:
                rank = fused_full_ids.index(gt_id) + 1
                if rank <= POOL_SIZE:
                    reranker_demotion_count += 1
                    print(f"    GT passage {gt_id}: rank {rank} in full RRF fusion "
                          f"(INSIDE pool={POOL_SIZE}) -- reranker demoted it")
                else:
                    cutoff_truncation_count += 1
                    print(f"    GT passage {gt_id}: rank {rank} in full RRF fusion "
                          f"(OUTSIDE pool={POOL_SIZE}) -- truncated before rescore")
            else:
                not_in_fusion_at_all_count += 1
                print(f"    GT passage {gt_id}: NOT in fused list at all "
                      f"(missed by both dense and BM25's top-{POOL_SIZE})")

    print(f"\n{'=' * 60}")
    print(f"Total regression cases (dense hit, hybrid miss): {total_regressions}")
    print(f"  - Cut off by pool_size truncation before rescore: {cutoff_truncation_count}")
    print(f"  - Demoted BY the reranker despite being in the pool: {reranker_demotion_count}")
    print(f"  - Missing from BOTH dense and BM25 top-{POOL_SIZE} entirely: {not_in_fusion_at_all_count}")


if __name__ == "__main__":
    main()