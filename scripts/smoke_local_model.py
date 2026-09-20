"""
Smoke test: Qwen2.5-3B-Instruct loads and answers sensibly.
Run it before relying on the model in responder.py, grounding.py or ragas_bridge.py.

Run with: python -m scripts.smoke_local_model
"""

from doctrace.answering.local_model import generate


def main() -> None:
    print("=== Test 1: basic generation ===")
    answer = generate(
        system_prompt="You are a helpful assistant. Be concise.",
        user_message="What is 2+2? Answer in one short sentence.",
        max_new_tokens=50,
    )
    print(f"Response: {answer}\n")

    print("=== Test 2: grounded generation (RAG-style) ===")
    context = (
        "FastAPI's TestClient is based on HTTPX, which in turn is designed "
        "based on the Requests library."
    )
    answer = generate(
        system_prompt=(
            "Answer ONLY using the provided context. If the context doesn't "
            "have enough information, say exactly: "
            "'I don't have enough information in the provided context to answer this.'"
        ),
        user_message=f"Context:\n{context}\n\nQuestion: What is FastAPI's TestClient built on?",
        max_new_tokens=100,
    )
    print(f"Response: {answer}\n")

    print("=== Test 3: refusal behavior (context doesn't cover the question) ===")
    answer = generate(
        system_prompt=(
            "Answer ONLY using the provided context. If the context doesn't "
            "have enough information, say exactly: "
            "'I don't have enough information in the provided context to answer this.'"
        ),
        user_message=f"Context:\n{context}\n\nQuestion: How do I connect FastAPI to MongoDB?",
        max_new_tokens=100,
    )
    print(f"Response: {answer}\n")

    print("=== Test 4: prefill trick (forcing JSON array output) ===")
    prompt = (
        'Break this into a JSON array of atomic claims, respond with ONLY '
        'the array, no other text:\n\n'
        '"FastAPI is fast and uses Pydantic for validation."'
    )
    answer = generate(system_prompt=None, user_message=prompt, max_new_tokens=200, prefill="[")
    print(f"Response: {answer}")


if __name__ == "__main__":
    main()