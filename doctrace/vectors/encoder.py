"""
Text encoder: a thin wrapper over BAAI/bge-m3 (1024-d, L2-normalised).

One place to turn document text or a user question into a vector.
"""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-m3"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME, device="cuda")
    return _model


def encode_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,  # small batches keep peak VRAM down
    )
    return vectors.tolist()


def encode_query(query: str) -> list[float]:
    """Encode one question. BGE-M3 wants no instruction prefix on queries, so this
    is the same path documents take (unlike e5-style models)."""
    return encode_texts([query])[0]


def release_gpu() -> None:
    """Drop the encoder and hand its VRAM back."""
    global _model
    import torch
    _model = None
    torch.cuda.empty_cache()
