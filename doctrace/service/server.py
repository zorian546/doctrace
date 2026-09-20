"""
HTTP service for the DocTrace pipeline.

Routes:
- POST /v1/ask: grounded answer with selectable search mode, blend weight and per-stage timings.
- GET /v1/status: whether the passage file is present and how many passages are indexed.
- GET /v1/scorecard: summary of the saved benchmark reports.

Hardening:
- CORS is limited to the origins in DOCTRACE_ALLOWED_ORIGINS (default: the local dashboard).
- If DOCTRACE_API_KEY is set, /v1/ask requires a matching X-API-Key header.
- Question length and top_k are bounded, and internal errors are logged, never echoed to clients.
- The service listens on 127.0.0.1 unless DOCTRACE_HOST says otherwise.

/v1/ask is a plain `def` route, so FastAPI runs it in a worker thread. The models block
for seconds at a time, and an `async def` route would freeze the whole event loop. The
pipeline's own lock keeps GPU use to one query at a time.
"""

import hmac
import json
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from doctrace.config import allowed_origins, api_key, bind_host, bind_port, get_device
from doctrace.pipeline import run_query
from doctrace.search.lexical import build_lexical_index

logger = logging.getLogger("doctrace.service")

PASSAGES_PATH = Path("data/processed/passages.json")
RETRIEVAL_REPORT_PATH = Path("data/processed/retrieval_report.json")
GENERATION_REPORT_PATH = Path("data/processed/generation_report.json")
MAX_QUESTION_CHARS = 500


@lru_cache(maxsize=1)
def _passage_count() -> int:
    try:
        return len(json.loads(PASSAGES_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return 0


def _read_report(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the keyword index once at startup."""
    if PASSAGES_PATH.exists():
        passages = json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))
        build_lexical_index(passages)
        logger.info("Keyword index ready over %d passages", len(passages))
    else:
        logger.warning("%s not found; run scripts/build_index.py first", PASSAGES_PATH)
    yield


app = FastAPI(
    title="DocTrace",
    description="Grounded question answering over the FastAPI docs, with selectable search modes and built-in auditing.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = api_key()
    if expected is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


# --- Schemas ---

class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=MAX_QUESTION_CHARS,
                       description="The question to answer")
    mode: Literal["semantic", "lexical", "fused"] = Field(
        default="fused", description="Search mode: semantic, lexical, or fused"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="How many passages to retrieve")
    rerank_weight: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Cross-encoder weight versus fusion score (fused mode only)",
    )


class EvidencePassage(BaseModel):
    id: int
    source_path: str
    header_path: str
    anchor: Optional[str] = None
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
def status_endpoint():
    """Report whether the passage file exists and how many passages it holds."""
    present = PASSAGES_PATH.exists()
    return StatusResponse(
        status="healthy" if present else "degraded",
        passages_indexed=_passage_count() if present else 0,
        passages_file_present=present,
        device=get_device(),
    )


@app.post("/v1/ask", response_model=AskResponse, tags=["Ask"],
          dependencies=[Depends(require_api_key)])
def ask_endpoint(request: AskRequest):
    """Answer a question about the FastAPI docs using only retrieved passages."""
    try:
        return run_query(request.query, request.mode, request.top_k, request.rerank_weight)
    except Exception:
        logger.exception("Pipeline failure")
        raise HTTPException(status_code=500, detail="The pipeline failed to produce an answer.")


@app.get("/v1/scorecard", tags=["Scorecard"])
def scorecard_endpoint():
    """Return the benchmark results saved by earlier evaluation runs."""
    generation = _read_report(GENERATION_REPORT_PATH)
    return {
        "retrieval_benchmark": _read_report(RETRIEVAL_REPORT_PATH),
        "generation_benchmark_sample_count": len(generation) if isinstance(generation, list) else None,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("doctrace.service.server:app", host=bind_host(), port=bind_port(), reload=False)
