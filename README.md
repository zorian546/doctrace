# DocTrace

[![tests](https://github.com/zorian546/doctrace/actions/workflows/tests.yml/badge.svg)](https://github.com/zorian546/doctrace/actions/workflows/tests.yml)

Question answering over the official FastAPI documentation, built so that its claims about quality can be checked. DocTrace retrieves passages, answers only from them, and ships the harness that measures whether retrieval found the right material, whether the answer stayed grounded in it, and whether the system declined when the docs had nothing to say.

## Why it exists

An assistant that answers from documentation is only useful if it can be trusted. A wrong parameter name, an invented code sample, or a confident reply to something the docs never cover costs more time than reading the docs would have. Three properties have to hold:

1. It retrieves the right source passages.
2. The answer is supported by what was retrieved, not merely plausible.
3. It recognises a question the docs do not cover and says so.

DocTrace treats each of these as something to measure. Every retrieval setting, reranking choice and model swap in this repo is backed by a number from a hand-checked set of 80 questions.

## What is in the box

- Heading-aware splitting of the Tutorial and Advanced User Guide (756 passages, no code fence ever cut)
- Hybrid search: BM25 keyword search plus BGE-M3 vectors in Qdrant, merged with Reciprocal Rank Fusion
- A cross-encoder rescoring stage, fine-tuned on this corpus with mined hard negatives
- Grounded answering with a local Qwen2.5-3B-Instruct model and a fixed refusal sentence
- A claim-level grounding score that can be read line by line
- An injection probe that plants misleading documents in the corpus and checks what happens
- A FastAPI service and a Streamlit dashboard, both runnable from one Docker image

[`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md) is the full write-up, including what did not work and how each problem was traced to its cause.

## Pipeline

```
FastAPI docs (sparse checkout, pinned commit)
        |
        v
  Fetch + split              -- snippet markers expanded into real code, then cut at
                                headings; a '#' inside a code fence is never a heading
        |
        v
  Encode (BGE-M3)            -- vectors stored in a Qdrant collection
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

Each stage is its own package (`doctrace/corpus`, `vectors`, `search`, `answering`, `audit`, `service`). Search mode, blend weight and the number of passages are set per request.

## Headline results

Full detail and caveats are in `ENGINEERING_NOTES.md`; the caveats matter, so the short version keeps them.

- With the stock cross-encoder, semantic search alone beat the fused pipeline: 0.831 Hit Rate@5 and 0.692 Recall@5 across the 65 questions that have gold passages. The cause was the cross-encoder preferring heading vocabulary over the passage that states the answer, diagnosed case by case rather than guessed at.
- Fine-tuning that cross-encoder on this corpus's hard negatives moved MRR from 0.624 to 0.739 and Hit Rate@5 from 0.769 to 0.877. **Caveat:** those figures include questions the model trained on, so they overstate the gain. `scripts/tune_reranker.py` now reports held-out questions separately; that re-run is still to do.
- The claim-level grounding score reached 0.971 (single-hop) and 0.914 (multi-hop) in the first run against Claude Sonnet 4.5 (recorded at the time; those raw answers were later overwritten), and parsed every one of the 65 answers, where RAGAS failed to parse its own judge output on 3. The local Qwen2.5-3B run, the one saved in this repo, scores 0.769 and 0.715.
- Both models declined all 15 unanswerable questions. The injection probe is less comfortable: an instruction-override attack was resisted, but insecure advice written as ordinary documentation was repeated faithfully.

## Reproducibility

The docs are edited constantly, so the corpus is pinned to upstream commit `7d210a4a`. Running `python -m scripts.build_index` reproduces `data/processed/passages.json` byte for byte, which a test asserts.

That pin is load-bearing rather than tidiness. A passage's id is its position in the passage list, and the same number is the vector store's point id, which is what lets fusion merge the two result lists. Upstream adding one section shifts every later id by one, and querying an older collection then returns correct-looking ids that point at the wrong text: answers get generated from passages nobody retrieved, which reads as a quality problem rather than a bug. So indexing records a fingerprint of the corpus it wrote and of the corpus it uploaded, and semantic search refuses to run when the two disagree (`doctrace/corpus/manifest.py`).

## Tech stack

**Search:** `BAAI/bge-m3` vectors (1024-d), Qdrant, rank-bm25, and `cross-encoder/ms-marco-MiniLM-L-6-v2` both stock and fine-tuned on this corpus.

**Answering:** Qwen2.5-3B-Instruct, local, bf16, through `transformers` (Claude Sonnet 4.5 in the first evaluation run).

**Serving:** FastAPI (`/v1/ask`, `/v1/status`, `/v1/scorecard`), a Streamlit dashboard, Docker and Compose.

**Evaluation:** ranking metrics (Hit Rate@k, Recall@k, Precision@k, MRR), a claim-level grounding score, RAGAS as a comparison, and 80 hand-written questions (single-hop, multi-hop, unanswerable).

## Layout

- `doctrace/` — the package. `corpus/` (fetch, split, fingerprint), `vectors/` (encoder), `search/` (keyword, semantic, fusion), `answering/` (local model, responder), `audit/` (ranking metrics, grounding score, gold set, injection probe, RAGAS adapter), `service/` (the HTTP app), plus `pipeline.py`, `config.py` and `security.py`.
- `scripts/` — entry points: `build_index`, the `bench_*` runs, `tune_reranker`, and the `smoke_*` checks. `scripts/casefiles/` holds the one-off investigations behind specific findings.
- `gold/` — the hand-written question set.
- `data/processed/` — the passage file, its manifest, and the saved benchmark reports. `data/raw/` holds the docs checkout and is not tracked.
- `models/` — the fine-tuned cross-encoder (Git LFS).
- `tests/` — the test suite.

## Setup

```bash
git clone https://github.com/zorian546/doctrace.git
cd doctrace
git lfs pull          # the fine-tuned cross-encoder; without it search falls back
                      # to the stock model
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then fill in QDRANT_URL and QDRANT_API_KEY
```

`requirements.txt` deliberately leaves out `torch`. This project targets a Blackwell/sm_120 GPU, which needs a PyTorch nightly cu128 build that is not on the default index; install it with the command in the `Dockerfile`, or install any CPU build to try the pipeline slowly.

To run only the tests, `pip install -r requirements-dev.txt` is enough: a few seconds, no ML stack, because those imports are deferred to first use.

## Running it

Run everything as `python -m ...` from the project root.

```bash
# 1. Build the index (fetches the pinned docs, writes data/processed/passages.json)
python -m scripts.build_index
python -m scripts.build_index --sync-qdrant     # also encode and upload the vectors

# 2. Tests
python -m pytest

# 3. Benchmarks (these need the vector store, and most need a GPU)
python -m scripts.bench_retrievers   # search configurations   -> retrieval_report.json
python -m scripts.bench_matrix       # splitting x encoder x mode -> ablation_grid.json
python -m scripts.bench_generation   # answering + grounding   -> generation_report.json
python -m scripts.bench_injection    # planted documents       -> injection_report.json
python -m scripts.tune_reranker      # fine-tune the cross-encoder
python -m scripts.bench_ragas        # RAGAS cross-check       -> ragas_crosscheck.json

# 4. Serve
python -m doctrace.service.server     # http://127.0.0.1:8000/docs
python -m streamlit run dashboard.py  # http://localhost:8501
docker compose up --build             # both; the dashboard calls the service
```

## Configuration and security

Settings come from environment variables (see `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY` | Vector store connection | required for semantic search |
| `DOCTRACE_API_KEY` | If set, `POST /v1/ask` requires a matching `X-API-Key` header | unset (open) |
| `DOCTRACE_HOST`, `DOCTRACE_PORT` | Where the service listens | `127.0.0.1`, `8000` |
| `DOCTRACE_ALLOWED_ORIGINS` | Browser origins allowed by CORS | the local dashboard |
| `DOCTRACE_API_URL` | Dashboard calls this service instead of loading the models itself | unset (in-process) |

The service is built to run on localhost or behind a reverse proxy. It bounds input sizes, does not echo internal errors, and runs as a non-root user in the container. The dashboard strips links, images and HTML from model output before rendering it, because a planted passage can otherwise make the answer carry a markdown image that exfiltrates data as soon as the page loads. What was reviewed and fixed is listed in the "Security review" section of `ENGINEERING_NOTES.md`; rate limiting and TLS are left to a proxy.

## License

MIT, see [LICENSE](LICENSE). The repository also redistributes FastAPI documentation text (MIT) in the passage file and a fine-tuned derivative of an Apache-2.0 cross-encoder; models downloaded at run time keep their own licenses.
