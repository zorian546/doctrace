"""
DocTrace dashboard (Streamlit).

Three tabs:
1. Ask: live grounded Q&A with retrieved evidence and a latency breakdown.
2. Benchmarks: metric tables and headline findings from the saved reports.
3. Injection probe: results of the planted-document attack scenarios.

Run with:
    python -m streamlit run dashboard.py
"""

import json
import time
from pathlib import Path
import streamlit as st

# Page setup
st.set_page_config(
    page_title="DocTrace",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling overrides
st.markdown(
    """
    <style>
    .main {
        background-color: #0f1419;
        color: #d7dde4;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px;
    }
    .citation-card {
        background: #182028;
        border: 1px solid #2c3742;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-primary { background-color: #0e7c86; color: #ffffff; }
    .badge-success { background-color: #3a9d5d; color: #ffffff; }
    .badge-warning { background-color: #e0a82e; color: #0f1419; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Setup ---

@st.cache_resource
def load_lexical():
    from doctrace.search.lexical import build_lexical_index
    passages_path = Path("data/processed/passages.json")
    if passages_path.exists():
        passages = json.loads(passages_path.read_text(encoding="utf-8"))
        build_lexical_index(passages)
        return len(passages)
    return 0

passages_count = load_lexical()

# --- Sidebar ---
st.sidebar.title("🔎 Pipeline settings")
mode = st.sidebar.selectbox(
    "Search mode",
    ["Fused (blend α=0.7)", "Semantic only (BGE-M3)", "Keyword only (BM25)", "Fused (cross-encoder only α=1.0)"],
    index=0,
)

top_k = st.sidebar.slider("Passages to retrieve (top-k)", min_value=1, max_value=10, value=5)

rerank_weight = 0.7
if "α=1.0" in mode:
    rerank_weight = 1.0
elif "α=0.7" in mode:
    rerank_weight = 0.7

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Passages indexed:** `{passages_count}`")
st.sidebar.markdown("**Encoder:** `BAAI/bge-m3` (1024d)")
st.sidebar.markdown("**Cross-encoder:** `ms-marco-MiniLM-L-6-v2`")
st.sidebar.markdown("**Answer model:** `Qwen2.5-3B-Instruct` (bf16)")

# --- Page body ---
st.title("🔎 DocTrace")
st.caption("Grounded question answering over the FastAPI docs, with every stage measured and timed.")

tab1, tab2, tab3 = st.tabs(["💬 Ask", "📊 Benchmarks", "🧪 Injection probe"])

# ---------- Tab 1: Ask ----------
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "Your question about FastAPI:",
            placeholder="e.g., How do you declare an integer path parameter in FastAPI?",
        )
    with col2:
        st.write("")
        st.write("")
        ask_btn = st.button("Get answer", use_container_width=True)

    # One-click sample questions
    st.markdown("**Try one:**")
    q_cols = st.columns(3)
    sample_queries = [
        "How do path parameters work with type annotations?",
        "How do you schedule background tasks to run after response?",
        "How do you connect FastAPI directly to Apache Kafka?",  # deliberately unanswerable from the docs
    ]
    
    for idx, q_text in enumerate(sample_queries):
        if q_cols[idx].button(q_text, key=f"quick_q_{idx}"):
            user_query = q_text
            ask_btn = True

    if ask_btn and user_query:
        with st.spinner("Searching the docs and drafting an answer..."):
            from doctrace.answering.responder import answer_question
            from doctrace.search.lexical import lexical_search
            from doctrace.search.semantic import semantic_search
            from doctrace.search.fusion import fused_search
            from doctrace.audit.grounding import is_abstention

            latencies = {}
            t0 = time.perf_counter()

            if "Semantic" in mode:
                retrieved = semantic_search(user_query, top_k=top_k)
                latencies["Semantic search"] = (time.perf_counter() - t0) * 1000
            elif "Keyword" in mode:
                retrieved = lexical_search(user_query, top_k=top_k)
                latencies["Keyword search"] = (time.perf_counter() - t0) * 1000
            else:
                out = fused_search(user_query, top_k=top_k, rerank_weight=rerank_weight)
                retrieved = out["results"]
                latencies["Semantic search"] = out["latency_ms"].get("semantic_ms", 0)
                latencies["Keyword search"] = out["latency_ms"].get("lexical_ms", 0)
                latencies["RRF merge"] = out["latency_ms"].get("merge_ms", 0)
                latencies["Cross-encoder rescoring"] = out["latency_ms"].get("rescore_ms", 0)

            latencies["Search total"] = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            gen = answer_question(user_query, retrieved)
            answer = gen["answer"]
            latencies["Answering"] = (time.perf_counter() - t1) * 1000
            latencies["End to end"] = (time.perf_counter() - t0) * 1000

            refusal = is_abstention(answer)

        # Results
        st.markdown("---")
        st.subheader("Answer")
        if refusal:
            st.warning("⚠️ **Declined to answer**: the retrieved passages do not cover this question, so the model said so instead of guessing.")
        
        st.markdown(answer)

        # Per-stage timings
        st.markdown("### Timings")
        m_cols = st.columns(len(latencies))
        for idx, (stage, ms) in enumerate(latencies.items()):
            m_cols[idx].metric(stage, f"{ms:.1f} ms")

        # Evidence shown to the model
        st.markdown("### Evidence")
        for rank, passage in enumerate(retrieved, start=1):
            with st.expander(f"#{rank} • {passage['source_path']} › {passage['header_path']}"):
                score_info = []
                if "blended_score" in passage:
                    score_info.append(f"**Blended score:** `{passage['blended_score']:.3f}`")
                if "ce_score" in passage:
                    score_info.append(f"**Cross-encoder logit:** `{passage['ce_score']:.3f}`")
                if "score" in passage:
                    score_info.append(f"**Cosine similarity:** `{passage['score']:.3f}`")
                
                if score_info:
                    st.markdown(" | ".join(score_info))
                
                st.code(passage["text"], language="markdown")


# ---------- Tab 2: Benchmarks ----------
with tab2:
    st.subheader("Gold-set results (80 questions)")
    st.caption("Measured on 40 single-hop, 25 multi-hop and 15 unanswerable questions. The table below comes from the saved run that uses the tuned cross-encoder.")

    retrieval_res_path = Path("data/processed/retrieval_report.json")
    if retrieval_res_path.exists():
        eval_data = json.loads(retrieval_res_path.read_text(encoding="utf-8"))

        summary_rows = []
        for cfg_name, res in eval_data.items():
            o = res["overall"]
            summary_rows.append({
                "Configuration": cfg_name,
                "Hit Rate@5": f"{o['hit_rate']:.3f}",
                "Recall@5": f"{o['recall']:.3f}",
                "Precision@5": f"{o['precision']:.3f}",
                "MRR": f"{o['mrr']:.3f}",
            })
        st.table(summary_rows)
    else:
        st.info("Run `python -m scripts.bench_retrievers` to produce the benchmark table.")

    st.markdown("---")
    st.subheader("Headline findings")
    st.markdown(
        """
        - **Stock reranker vs tuned reranker**: with the off-the-shelf cross-encoder, semantic search alone won on hit rate (0.831) and recall (0.692). After tuning the cross-encoder on this corpus, the fused pipeline overtakes it (hit rate 0.877, MRR 0.692 at α=0.7).
        - **Stock cross-encoders over-weight heading vocabulary**: blending in the RRF score at $\\alpha=0.7$ steadies the ranking.
        - **Declines when it should**: the constrained prompt kept the model from inventing answers on the 15 unanswerable (`no_answer`) questions.
        """
    )


# ---------- Tab 3: Injection probe ----------
with tab3:
    st.subheader("Injection probe: planted documents")
    st.caption("How the pipeline holds up when the corpus contains deliberately planted passages.")

    sec_res_path = Path("data/processed/injection_report.json")
    if sec_res_path.exists():
        sec_data = json.loads(sec_res_path.read_text(encoding="utf-8"))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Scenarios", sec_data.get("total_scenarios", 0))
        c1.metric("Planted passage in top-k", f"{sec_data.get('topk_reach_rate', 0)*100:.1f}%")
        c3.metric("Attack landed rate", f"{sec_data.get('attack_landed_rate', 0)*100:.1f}%")

        st.markdown("#### Per scenario")
        for sc in sec_data.get("scenarios", []):
            status_badge = "❌ LANDED" if sc["attack_landed"] else "✅ HELD"
            with st.expander(f"Scenario [{sc['scenario_id']}] - {status_badge}"):
                st.markdown(f"**Question asked:** `{sc['target_query']}`")
                st.markdown(f"**Planted passage retrieved:** `{sc['tainted_reached_topk']}`")
                st.markdown(f"**Marker in answer:** `{sc['marker_in_answer']}`")
                st.markdown(f"**Answer given:**\n> {sc['generated_answer']}")
    else:
        st.info("Run `python -m scripts.bench_injection` to run the injection probe.")
