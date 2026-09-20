"""
Grounded answering.

Asks the local Qwen2.5-3B-Instruct model (doctrace/answering/local_model.py) to answer
strictly from the supplied passages, with a fixed refusal sentence for questions the
passages cannot support.
"""

from doctrace.answering.local_model import complete

_SYSTEM_PROMPT = """You are a documentation assistant answering questions about FastAPI \
using ONLY the provided context chunks.

Rules:
- Answer only using information present in the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, respond with \
exactly: "I don't have enough information in the provided context to answer this." \
Do not guess or fill gaps with general knowledge.
- Be concise and direct. Cite which part of the context supports your answer where natural.
"""


def answer_question(query: str, retrieved: list[dict]) -> dict:
    """Answer `query` from the retrieved passages.

    Returns {"answer": str, "evidence": list[dict]}. The passages are handed back so
    the grounding check does not need to fetch them a second time.
    """
    context_text = "\n\n---\n\n".join(
        f"[Source: {p['source_path']} > {p['header_path']}]\n{p['text']}"
        for p in retrieved
    )

    user_message = f"Context:\n\n{context_text}\n\n---\n\nQuestion: {query}"

    reply = complete(_SYSTEM_PROMPT, user_message, max_new_tokens=500)

    return {"answer": reply, "evidence": retrieved}
