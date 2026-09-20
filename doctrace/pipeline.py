"""
The one place a question turns into an answer.

Both the HTTP service and the dashboard call `run_query`, so search modes, timing keys
and the GPU lock cannot drift apart between them.
"""

import threading
import time

from doctrace.answering.responder import answer_question
from doctrace.audit.grounding import is_abstention
from doctrace.search.fusion import fused_search
from doctrace.search.lexical import lexical_search
from doctrace.search.semantic import semantic_search

MODES = ("semantic", "lexical", "fused")
POOL_SIZE = 20

# The encoder, cross-encoder and answer model share one GPU and are not thread-safe,
# so only one query may run the models at a time.
_MODEL_LOCK = threading.Lock()


def run_query(query: str, mode: str = "fused", top_k: int = 5,
              rerank_weight: float = 0.7) -> dict:
    """Search, then answer from what was found.

    Returns {"query", "answer", "abstained", "mode", "top_k", "evidence", "latency_ms"}.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    timings: dict[str, float] = {}
    with _MODEL_LOCK:
        t0 = time.perf_counter()
        if mode == "semantic":
            retrieved = semantic_search(query, top_k=top_k)
            timings["semantic_ms"] = (time.perf_counter() - t0) * 1000
        elif mode == "lexical":
            retrieved = lexical_search(query, top_k=top_k)
            timings["lexical_ms"] = (time.perf_counter() - t0) * 1000
        else:
            fused = fused_search(query, top_k=top_k, pool_size=POOL_SIZE,
                                 rerank_weight=rerank_weight)
            retrieved = fused["results"]
            timings.update(fused["latency_ms"])
        timings["total_retrieval_ms"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        answer = answer_question(query, retrieved)["answer"]
        timings["generation_ms"] = (time.perf_counter() - t1) * 1000
        timings["total_pipeline_ms"] = (time.perf_counter() - t0) * 1000

    evidence = [
        {
            "id": p["id"],
            "source_path": p["source_path"],
            "header_path": p["header_path"],
            "text": p["text"],
            "score": p.get("score"),
            "ce_score": p.get("ce_score"),
            "blended_score": p.get("blended_score"),
        }
        for p in retrieved
    ]
    return {
        "query": query,
        "answer": answer,
        "abstained": is_abstention(answer),
        "mode": mode,
        "top_k": top_k,
        "evidence": evidence,
        "latency_ms": timings,
    }
