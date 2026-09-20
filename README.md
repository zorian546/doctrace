---
title: DocTrace
sdk: docker
app_port: 8501
pinned: false
---

# DocTrace

Question answering over the official FastAPI documentation, built so that every claim about its quality can be checked. DocTrace retrieves passages, answers only from them, and ships the harness that measures whether retrieval found the right material, whether the answer stayed grounded in it, and whether the system declined when the docs had nothing to say.

## Why it exists

An assistant that answers from documentation is only useful if it can be trusted. A wrong parameter name, an invented code sample, or a confident reply to something the docs never cover costs more time than reading the docs would have. Three properties have to hold:

1. It retrieves the right source passages.
2. The answer is supported by what was retrieved, not merely plausible.
3. It recognises a question the docs do not cover and says so.

DocTrace treats each of these as something to measure. Every retrieval setting, reranking choice and model swap in this repo is backed by a number from a hand-checked set of 80 questions.

## What is in the box

- Heading-aware splitting of the Tutorial and Advanced User Guide (756 passages, no code fence ever cut)
- Hybrid search: BM25 keyword search plus BGE-M3 vectors in Qdrant, merged with Reciprocal Rank Fusion
- A cross-encoder rescoring stage, tuned on this corpus with mined hard negatives
- Grounded answering with a local Qwen2.5-3B-Instruct model and a fixed refusal sentence
- A claim-level grounding score that can be read line by line
- An injection probe that plants misleading documents in the corpus and checks what happens
- A FastAPI service and a Streamlit dashboard, both runnable from one Docker image

[`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md) is the full write-up, including what did not work and how each problem was traced to its cause.

## Pipeline

```
FastAPI docs (sparse git clone)
        |
        v
  Fetch + split              -- snippet markers expanded into real code, then cut at
                                headings; a '#' inside a code fence is never a heading
        |
        v
  Encode (BGE-M3, GPU)       -- vectors stored in a Qdrant Cloud collection
        |
        v
  Fused search               -- BM25 and vector hits merged with Reciprocal Rank Fusion
        |
        v
  Cross-encoder rescoring    -- blended with the fusion score (weight 0.7); the
                                cross-encoder is tuned on this corpus's hard negatives
        |
        v
  Grounded answer            -- local Qwen2.5-3B-Instruct, told to use only the passages
        |
        v
  Audit                      -- ranking metrics, claim-level grounding, refusal
                                accuracy, planted-document probe
```

Each stage is its own package (`doctrace/corpus`, `vectors`, `search`, `answering`, `audit`, `service`). Search mode, blend weight and answer model can all be changed at run time.

## Headline results

Full detail and caveats are in `ENGINEERING_NOTES.md`.

- With the stock cross-encoder, semantic search alone beat the fused pipeline: 0.831 Hit Rate@5 and 0.692 Recall@5 across the 65 questions that have gold passages. The cause turned out to be the reranker favouring heading vocabulary over the passage that actually states the answer.
- Tuning that cross-encoder on this corpus's own hard negatives moved end-to-end MRR from 0.624 to 0.739 (+18.4%) and Hit Rate@5 from 0.769 to 0.877 (+14.0%), measured through the live pipeline.
- The claim-level grounding score reached 0.971 (single-hop) and 0.914 (multi-hop) against Claude Sonnet 4.5, and parsed every one of the 65 answers. RAGAS's own faithfulness metric failed to parse its judge output on 4 of those 65.
- Hosted API credits ran out during evaluation, so answering moved to a local Qwen2.5-3B-Instruct. That needed its own reliability work, and it produced a real hosted-versus-local comparison instead of a workaround.

## Tech stack

**Search**
- `BAAI/bge-m3` vectors (1024-d) on GPU
- Qdrant Cloud for vector storage
- rank-bm25 for keyword search
- `cross-encoder/ms-marco-MiniLM-L-6-v2`, stock and tuned on this corpus

**Answering**
- Qwen2.5-3B-Instruct, local, bf16, through `transformers` (Claude Sonnet 4.5 in the first evaluation run)

**Serving**
- FastAPI (`/v1/ask`, `/v1/status`, `/v1/scorecard`)
- Streamlit dashboard
- Docker and Compose, with GPU passthrough

**Evaluation**
- Ranking metrics: Hit Rate@k, Recall@k, Precision@k, MRR
- Claim-level grounding score
- RAGAS as a comparison baseline
- 80 hand-written questions: single-hop, multi-hop and unanswerable

## Layout

- `doctrace/`: the package. `corpus/` (fetch and split), `vectors/` (encoder), `search/` (keyword, semantic, fusion), `answering/` (local model and responder), `audit/` (ranking metrics, grounding score, gold set, injection probe, RAGAS adapter), `service/` (the HTTP app).
- `scripts/`: entry points. `build_index`, the `bench_*` runs, `tune_reranker`, and the `smoke_*` checks. `scripts/casefiles/` keeps the one-off investigation scripts behind specific findings; see its README.
- `gold/`: the hand-written question set (`gold_pairs.json`).
- `data/`: `raw/` holds the docs checkout (git-ignored, rebuilt by indexing); `processed/` holds the passage file and the saved reports.
- `models/`: the tuned cross-encoder.
- `tests/`: unit and integration tests.

## Running it

Run everything as `python -m ...` from the project root.

### 1. Build the index

```powershell
# Fetch the docs and write data/processed/passages.json
python -m scripts.build_index

# Also encode the passages and upload them to Qdrant
python -m scripts.build_index --sync-qdrant
```

The Qdrant upload needs `QDRANT_URL` and `QDRANT_API_KEY` in a `.env` file (copy `.env.example`). The default collection name is `doctrace_passages`.

### 2. Tests

```powershell
python -m pytest tests/ -v
```

### 3. Benchmarks

```powershell
# Search configurations compared on the gold set
python -m scripts.bench_retrievers

# Full grid: splitting strategy x encoder x search mode
python -m scripts.bench_matrix

# Two-stage answering and grounding run
python -m scripts.bench_generation

# Planted-document injection probe
python -m scripts.bench_injection

# Tune the cross-encoder on this corpus
python -m scripts.tune_reranker
```

### 4. Start the service and dashboard

```powershell
# Directly
python -m doctrace.service.server
python -m streamlit run dashboard.py

# Or with Docker Compose (HTTP service on 8000, dashboard on 8501)
docker compose up --build
```

The tuned cross-encoder weights are stored with Git LFS. Run `git lfs pull` after cloning; without them the search falls back to the stock cross-encoder.
