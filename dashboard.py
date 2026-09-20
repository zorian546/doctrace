"""
DocTrace dashboard (Streamlit).

Tabs:
1. Ask: grounded answers with the evidence behind them and a per-stage timing chart.
2. Benchmarks: retrieval and grounding results read from the saved reports.
3. Injection probe: what happened when misleading passages were planted in the corpus.
4. How it works: the pipeline in five steps.

Backend: with DOCTRACE_API_URL set, questions go to the HTTP service (the compose setup
does this, so the models load once). Without it the pipeline runs in this process,
which is what a single-container host such as Hugging Face Spaces needs.

Run with:
    python -m streamlit run dashboard.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from doctrace.security import docs_url, sanitize_answer

REPORTS = Path("data/processed")
API_URL = os.getenv("DOCTRACE_API_URL", "").rstrip("/")
API_KEY = os.getenv("DOCTRACE_API_KEY", "")
MAX_QUESTION_CHARS = 500

MODE_LABELS = {
    "Fused (recommended)": "fused",
    "Semantic only": "semantic",
    "Keyword only": "lexical",
}
STAGE_LABELS = {
    "semantic_ms": "Semantic search",
    "lexical_ms": "Keyword search",
    "merge_ms": "RRF merge",
    "rescore_ms": "Cross-encoder",
    "generation_ms": "Answering",
}
EXAMPLES = [
    "How do path parameters work with type annotations?",
    "How do you schedule background tasks to run after the response?",
    "How do you connect FastAPI directly to Apache Kafka?",
]
CONFIG_LABELS = {
    "semantic_only": "Semantic only",
    "lexical_only": "Keyword only (BM25)",
    "fused_w1.0": "Fused, cross-encoder only",
    "fused_w0.7": "Fused, blend 0.7 (default)",
    "fused_w0.5": "Fused, blend 0.5",
    "fused_w0.3": "Fused, blend 0.3",
}

st.set_page_config(page_title="DocTrace", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-top: 2rem; }
    h1 { letter-spacing: -0.02em; margin-bottom: 0; }
    .tagline { color: #8b98a5; margin: 0 0 1.4rem 0; }
    .pill { display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 0.75rem;
            border: 1px solid #2c3742; color: #8b98a5; margin-right: 6px; }
    .src a { color: #2fb5c0; text-decoration: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- data access ----------

def load_report(name: str):
    try:
        return json.loads((REPORTS / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ask(query: str, mode: str, top_k: int, weight: float) -> dict:
    """Answer via the HTTP service when configured, otherwise in process."""
    if API_URL:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        reply = requests.post(
            f"{API_URL}/v1/ask",
            json={"query": query, "mode": mode, "top_k": top_k, "rerank_weight": weight},
            headers=headers,
            timeout=600,
        )
        reply.raise_for_status()
        return reply.json()
    from doctrace.pipeline import run_query
    return run_query(query, mode, top_k, weight)


@st.cache_resource
def prepare_local_index() -> int:
    """Build the keyword index once per process (in-process backend only)."""
    from doctrace.search.lexical import build_lexical_index
    passages = load_report("passages.json") or []
    if passages:
        build_lexical_index(passages)
    return len(passages)


passage_total = len(load_report("passages.json") or [])
if not API_URL:
    prepare_local_index()


# ---------- sidebar ----------

st.sidebar.header("Settings")
mode_label = st.sidebar.selectbox("Search mode", list(MODE_LABELS), index=0)
mode = MODE_LABELS[mode_label]
top_k = st.sidebar.slider("Passages to retrieve", 1, 10, 5)
weight = st.sidebar.slider(
    "Cross-encoder weight", 0.0, 1.0, 0.7, 0.1, disabled=mode != "fused",
    help="1.0 ranks by the cross-encoder alone; lower values keep more of the "
         "keyword+semantic agreement (RRF score).",
)
st.sidebar.divider()
st.sidebar.caption(f"**Backend:** {'HTTP service' if API_URL else 'in-process'}")
st.sidebar.caption(f"**Passages indexed:** {passage_total}")
st.sidebar.caption("**Encoder:** BAAI/bge-m3")
st.sidebar.caption("**Cross-encoder:** MiniLM-L-6, tuned on this corpus")
st.sidebar.caption("**Answer model:** Qwen2.5-3B-Instruct")

st.title("DocTrace")
st.markdown('<p class="tagline">Answers about the FastAPI docs, grounded in retrieved passages '
            "and measured at every stage.</p>", unsafe_allow_html=True)

tab_ask, tab_bench, tab_probe, tab_how = st.tabs(
    ["Ask", "Benchmarks", "Injection probe", "How it works"]
)


# ---------- Ask ----------

def render_result(result: dict) -> None:
    with st.container(border=True):
        if result["abstained"]:
            st.markdown("**The docs do not cover this.**")
            st.caption("The retrieved passages were not enough, so the model declined "
                       "instead of guessing.")
        else:
            # model output is untrusted: strip links, images and HTML before rendering
            st.markdown(sanitize_answer(result["answer"]))

    timings = result["latency_ms"]
    stages = {label: timings[key] for key, label in STAGE_LABELS.items() if key in timings}
    left, right = st.columns([2, 1])
    with left:
        st.caption("Time per stage (ms)")
        st.bar_chart(pd.Series(stages, name="ms"), horizontal=True, height=40 + 34 * len(stages))
    with right:
        st.metric("End to end", f"{timings['total_pipeline_ms'] / 1000:.2f} s")
        st.metric("Search", f"{timings['total_retrieval_ms']:.0f} ms")

    st.subheader("Evidence")
    for rank, p in enumerate(result["evidence"], start=1):
        with st.expander(f"{rank}. {p['source_path']}  ›  {p['header_path']}"):
            scores = []
            if p.get("blended_score") is not None:
                scores.append(f"blended {p['blended_score']:.3f}")
            if p.get("ce_score") is not None:
                scores.append(f"cross-encoder {p['ce_score']:.2f}")
            if p.get("score") is not None:
                scores.append(f"score {p['score']:.3f}")
            st.markdown(
                f'<span class="src"><a href="{docs_url(p["source_path"], p.get("anchor"))}" target="_blank" '
                f'rel="noopener noreferrer">Open in the FastAPI docs</a></span> '
                f'<span class="pill">{" · ".join(scores)}</span>',
                unsafe_allow_html=True,
            )
            st.code(p["text"], language="markdown")


with tab_ask:
    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input(
            "Your question about FastAPI",
            max_chars=MAX_QUESTION_CHARS,
            placeholder="e.g. How do you declare an integer path parameter?",
        )
        submitted = st.form_submit_button("Get answer", type="primary")

    st.caption("Try one:")
    cols = st.columns(len(EXAMPLES))
    for col, text in zip(cols, EXAMPLES):
        if col.button(text, key=f"ex_{text}"):
            question, submitted = text, True

    if submitted:
        if len(question.strip()) < 2:
            st.warning("Type a question first.")
        else:
            try:
                with st.spinner("Searching the docs and drafting an answer..."):
                    st.session_state["result"] = ask(question.strip(), mode, top_k, weight)
            except Exception:
                st.session_state.pop("result", None)
                st.error("Something went wrong while answering. Check that the service and "
                         "the models are available, then try again.")

    if "result" in st.session_state:
        render_result(st.session_state["result"])


# ---------- Benchmarks ----------

with tab_bench:
    retrieval = load_report("retrieval_report.json")
    if retrieval:
        st.subheader("Retrieval on the gold set")
        st.caption("65 questions with source passages (40 single-hop, 25 multi-hop), top 5. "
                   "Fused rows use the tuned cross-encoder.")
        table = pd.DataFrame(
            [
                {
                    "Configuration": CONFIG_LABELS.get(name, name),
                    "Hit rate@5": r["overall"]["hit_rate"],
                    "Recall@5": r["overall"]["recall"],
                    "Precision@5": r["overall"]["precision"],
                    "MRR": r["overall"]["mrr"],
                }
                for name, r in retrieval.items()
            ]
        ).set_index("Configuration")
        st.dataframe(table.style.format("{:.3f}"))
        st.bar_chart(table[["Hit rate@5", "Recall@5", "MRR"]], horizontal=True)
        st.info(
            "Caveat: the cross-encoder was tuned on questions from this same gold set, so the "
            "fused rows are optimistic. `scripts/tune_reranker.py` now also reports a held-out "
            "split; that number is the one to quote."
        )
    else:
        st.info("Run `python -m scripts.bench_retrievers` to produce the retrieval table.")

    generation = load_report("generation_report.json")
    if generation:
        st.subheader("Answering on the gold set (local Qwen2.5-3B-Instruct)")
        rows = []
        for cat in ("single_hop", "multi_hop"):
            scored = [r["grounding_score"] for r in generation
                      if r["category"] == cat and r.get("grounding_score") is not None]
            rows.append({"Category": cat, "Questions": len(scored),
                         "Mean grounding score": sum(scored) / len(scored) if scored else 0.0})
        unanswerable = [r for r in generation if r["category"] == "no_answer"]
        declined = sum(1 for r in unanswerable if r.get("correctly_refused"))
        st.dataframe(pd.DataFrame(rows).set_index("Category").style.format({"Mean grounding score": "{:.3f}"}),
                     use_container_width=True)
        st.metric("Unanswerable questions correctly declined", f"{declined} / {len(unanswerable)}")


# ---------- Injection probe ----------

with tab_probe:
    probe = load_report("injection_report.json")
    st.subheader("Planted-document scenarios")
    st.caption("Two misleading passages were added to the corpus. Retrieval here is keyword "
               "search only, so this shows how the answer model reacts, not how the deployed "
               "search ranks a planted page.")
    if probe:
        a, b, c = st.columns(3)
        a.metric("Scenarios", probe["total_scenarios"])
        b.metric("Planted passage in top-k", f"{probe['topk_reach_rate'] * 100:.0f}%")
        c.metric("Attack landed", f"{probe['attack_landed_rate'] * 100:.0f}%")
        for s in probe["scenarios"]:
            verdict = "LANDED" if s["attack_landed"] else "HELD"
            with st.expander(f"{s['scenario_id']} · {s['attack_type']} · {verdict}"):
                st.write(f"**Question:** {s['target_query']}")
                st.write(f"**Planted passage retrieved:** {s['tainted_reached_topk']}")
                st.write(f"**Payload in answer:** {s['marker_in_answer']}")
                st.text(s["generated_answer"])  # plain text: never render it as markdown
        st.info("Instruction override was resisted. Bad advice written as ordinary documentation "
                "was repeated faithfully. Grounding and safety are separate properties.")
    else:
        st.info("Run `python -m scripts.bench_injection` to produce the probe report.")


# ---------- How it works ----------

with tab_how:
    st.subheader("From question to answer")
    st.markdown(
        """
        1. **Split**: the docs are cut at headings, never inside a code fence, and snippet
           includes are expanded into real code (756 passages).
        2. **Search twice**: BM25 finds exact terms, BGE-M3 vectors find meaning.
        3. **Merge**: Reciprocal Rank Fusion combines the two rankings using rank only, so the
           two score scales never need to be compared.
        4. **Rescore**: a cross-encoder tuned on this corpus reorders the top 20, blended with the
           merge score.
        5. **Answer**: a local model answers only from the passages, or says the docs do not
           cover it. A claim-level grounding score then checks the answer against the evidence.
        """
    )
    st.caption("Design notes, measurements and known limitations are in ENGINEERING_NOTES.md.")
