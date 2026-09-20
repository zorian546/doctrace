"""
Fused search: semantic + lexical candidates, merged by rank, then rescored by a
cross-encoder.

Stages:
1. Candidate pool: pull the same number of hits from the vector search (BGE-M3) and
   the keyword search (BM25).
2. Reciprocal Rank Fusion (k=60): merge the two lists using rank position only, so
   cosine similarities and unbounded BM25 scores never have to be put on one scale.
3. Cross-encoder rescoring with blending: min-max normalise the fusion score and the
   cross-encoder logit over the pool, then combine
     blended_score = rerank_weight * norm_ce + (1 - rerank_weight) * norm_fusion
   The blend stops a stock cross-encoder, which likes heading vocabulary too much,
   from throwing away what the two searches agreed on and demoting the canonical
   answer.
"""

import os
import time

from doctrace.config import get_device
from doctrace.search.lexical import lexical_search
from doctrace.search.semantic import semantic_search

# Cross-encoder tuned on this corpus with mined hard negatives (see
# scripts/tune_reranker.py and ENGINEERING_NOTES.md). The weights are ~87MB and stored in
# Git LFS. A checkout without them (no LFS, or `git lfs pull` never run) falls
# back to the stock model instead of failing.
_TUNED_RERANKER_DIR = "models/docs-reranker-minilm"
_STOCK_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None


def _load_cross_encoder():
    """Load on first use; see doctrace/vectors/encoder.py for why this is deferred."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        weight_file = os.path.join(_TUNED_RERANKER_DIR, "model.safetensors")
        tuned_usable = (
            os.path.isdir(_TUNED_RERANKER_DIR)
            and os.path.isfile(weight_file)
            and os.path.getsize(weight_file) > 1_000_000
        )
        if tuned_usable:
            print(f"Loading tuned cross-encoder from {_TUNED_RERANKER_DIR}")
            _cross_encoder = CrossEncoder(_TUNED_RERANKER_DIR, device=get_device())
        else:
            if os.path.isdir(_TUNED_RERANKER_DIR):
                # a Git LFS pointer stub is only a few hundred bytes
                print(
                    f"{_TUNED_RERANKER_DIR} exists but model.safetensors is absent or tiny "
                    f"(probably an unpulled Git LFS stub) -- using stock {_STOCK_RERANKER}"
                )
            else:
                print(
                    f"No tuned reranker at {_TUNED_RERANKER_DIR} -- using stock {_STOCK_RERANKER}"
                )
            _cross_encoder = CrossEncoder(_STOCK_RERANKER, device=get_device())
    return _cross_encoder


def rrf_merge(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: a passage's score is the sum of 1 / (k + rank) over
    every list it shows up in. k=60 is the constant from the original paper; it
    softens the gap between rank 1 and rank 2 while still favouring passages that
    place well in several lists."""
    totals: dict[int, float] = {}
    by_id: dict[int, dict] = {}

    for hits in result_lists:
        for position, hit in enumerate(hits):
            pid = hit["id"]
            totals[pid] = totals.get(pid, 0.0) + 1.0 / (k + position + 1)
            by_id[pid] = hit

    ordered = sorted(totals, key=lambda pid: totals[pid], reverse=True)
    return [{**by_id[pid], "fusion_score": totals[pid]} for pid in ordered]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rescore(query: str, candidates: list[dict], top_k: int = 5,
            rerank_weight: float | None = 0.7) -> list[dict]:
    """Cross-encoder rescoring of a candidate pool (never the whole corpus, that
    would be far too slow).

    rerank_weight decides how much the cross-encoder counts against the existing
    fusion_score once both are min-max normalised inside this pool:
      blended_score = rerank_weight * norm_ce + (1 - rerank_weight) * norm_fusion

    1.0 (or None) gives the original cross-encoder-only ordering. That mode is kept
    because the ablation needs it: the regression was diagnosed against it.

    Candidates must already carry "fusion_score", i.e. come from rrf_merge().
    """
    if not candidates:
        return []

    ce_scores = _load_cross_encoder().predict([(query, c["text"]) for c in candidates])

    if rerank_weight is None or rerank_weight >= 1.0:
        ranked = sorted(zip(candidates, ce_scores), key=lambda pair: pair[1], reverse=True)
        return [
            {**c, "ce_score": float(s), "blended_score": float(s)}
            for c, s in ranked[:top_k]
        ]

    norm_ce = _minmax(list(ce_scores))
    norm_fusion = _minmax([c.get("fusion_score", 0.0) for c in candidates])

    scored = []
    for cand, raw, n_ce, n_fu in zip(candidates, ce_scores, norm_ce, norm_fusion):
        blended = rerank_weight * n_ce + (1 - rerank_weight) * n_fu
        scored.append({**cand, "ce_score": float(raw), "blended_score": float(blended)})

    scored.sort(key=lambda item: item["blended_score"], reverse=True)
    return scored[:top_k]


def fused_search(query: str, top_k: int = 5, pool_size: int = 20,
                 rerank_weight: float = 0.7) -> dict:
    """End to end: semantic + lexical search (pool_size hits each), RRF merge, then
    cross-encoder rescoring blended with the fusion score (rerank_weight=1.0 means
    cross-encoder only).

    Returns {"results": [...], "latency_ms": {...}} so each stage's cost is visible.
    """
    timings = {}

    started = time.perf_counter()
    semantic_hits = semantic_search(query, top_k=pool_size)
    timings["semantic_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    lexical_hits = lexical_search(query, top_k=pool_size)
    timings["lexical_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    merged = rrf_merge([semantic_hits, lexical_hits])
    timings["merge_ms"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    final = rescore(query, merged[:pool_size], top_k=top_k, rerank_weight=rerank_weight)
    timings["rescore_ms"] = (time.perf_counter() - started) * 1000

    return {"results": final, "latency_ms": timings}
