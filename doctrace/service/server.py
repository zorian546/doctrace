"""
HTTP service for the DocTrace pipeline.

Endpoints:
- POST /v1/ask: grounded answer with selectable search mode, blend weight and per-stage timings.
- GET /v1/status: whether the passage file is present and how many passages are indexed.
- GET /v1/scorecard: summary of the saved benchmark reports.

A single asyncio lock lets only one request use the GPU at a time, which avoids
CUDA out-of-memory errors on an 8GB card.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from doctrace.audit.grounding import is_abstention
from doctrace.answering.responder import answer_question
from doctrace.search.lexical import lexical_search, build_lexical_index
from doctrace.search.semantic import semantic_search
from doctrace.search.fusion import fused_search

# one GPU request at a time
_GPU_GATE = asyncio.Lock()
PASSAGES_PATH = Path("data/processed/passages.json")
RETRIEVAL_REPORT_PATH = Path("data/processed/retrieval_report.json")
GENERATION_REPORT_PATH = Path("data/processed/generation_report.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the keyword index on startup."""
    if PASSAGES_PATH.exists():
        passages = json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))
        build_lexical_index(passages)
        print(f"[startup] Keyword index ready over {len(passages)} passages.")
    else:
        print("[startup WARNING] data/processed/passages.json not found. Run the indexing script first.")
    yield
    print("[shutdown] Releasing resources.")


app = FastAPI(
    title="DocTrace",
    description="Grounded question answering over the FastAPI docs, with selectable search modes and built-in auditing.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---

class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, description="The question to answer")
    mode: Literal["dense", "bm25", "hybrid"] = Field(
        default="hybrid", description="Search mode: dense, bm25, or hybrid"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="How many passages to retrieve")
    rerank_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Cross-encoder weight versus fusion score (hybrid mode only)"
    )


class EvidencePassage(BaseModel):
    id: int
    source_path: str
    header_path: str
    text: str
    score: Optional[float] = None
    ce_score: Optional[float] = None
    blended_score: Optional[float] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    abstained: bool
    mode: str
    top_k: int
    evidence: list[EvidencePassage]
    latency_ms: dict[str, float]


class StatusResponse(BaseModel):
    status: str
    passages_indexed: int
    passages_file_present: bool
    device: str


# --- Routes ---

@app.get("/v1/status", response_model=StatusResponse, tags=["Status"])
async def status_endpoint():
    """Report whether the passage file exists and how many passages it holds."""
    present = PASSAGES_PATH.exists()
    total = 0
    if present:
        try:
            total = len(json.loads(PASSAGES_PATH.read_text(encoding="utf-8")))
        except Exception:
            total = 0

    return StatusResponse(
        status="healthy" if present else "degraded",
        passages_indexed=total,
        passages_file_present=present,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
    )


@app.post("/v1/ask", response_model=AskResponse, tags=["Ask"])
async def ask_endpoint(request: AskRequest):
    """Answer a question about the FastAPI docs using only retrieved passages."""
    latency: dict[str, float] = {}

    async with _GPU_GATE:
        try:
            # 1. search
            t0 = time.perf_counter()
            if request.mode == "dense":
                retrieved = semantic_search(request.query, top_k=request.top_k)
                latency["semantic_ms"] = (time.perf_counter() - t0) * 1000
            elif request.mode == "bm25":
                retrieved = lexical_search(request.query, top_k=request.top_k)
                latency["lexical_ms"] = (time.perf_counter() - t0) * 1000
            else:  # hybrid
                fused = fused_search(
                    request.query,
                    top_k=request.top_k,
                    pool_size=20,
                    rerank_weight=request.rerank_weight,
                )
                retrieved = fused["results"]
                latency.update(fused["latency_ms"])

            latency["total_retrieval_ms"] = (time.perf_counter() - t0) * 1000

            # 2. answer
            t1 = time.perf_counter()
            answer = answer_question(request.query, retrieved)["answer"]
            latency["generation_ms"] = (time.perf_counter() - t1) * 1000
            latency["total_pipeline_ms"] = (time.perf_counter() - t0) * 1000

            evidence = [
                EvidencePassage(
                    id=p["id"],
                    source_path=p["source_path"],
                    header_path=p["header_path"],
                    text=p["text"],
                    score=p.get("score"),
                    ce_score=p.get("ce_score"),
                    blended_score=p.get("blended_score"),
                )
                for p in retrieved
            ]

            return AskResponse(
                query=request.query,
                answer=answer,
                abstained=is_abstention(answer),
                mode=request.mode,
                top_k=request.top_k,
                evidence=evidence,
                latency_ms=latency,
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline failure: {str(e)}",
            )


@app.get("/v1/scorecard", tags=["Scorecard"])
async def scorecard_endpoint():
    """Return the benchmark results saved by earlier evaluation runs."""
    retrieval_data = None
    generation_data = None

    if RETRIEVAL_REPORT_PATH.exists():
        try:
            retrieval_data = json.loads(RETRIEVAL_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    if GENERATION_REPORT_PATH.exists():
        try:
            generation_data = json.loads(GENERATION_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "retrieval_benchmark": retrieval_data,
        "generation_benchmark_sample_count": len(generation_data) if isinstance(generation_data, list) else None,
    }


if __name__ == "__main__":
    uvicorn.run("doctrace.service.server:app", host="0.0.0.0", port=8000, reload=False)
