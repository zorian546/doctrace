"""
Text encoder: a thin wrapper over BAAI/bge-m3 (1024-d, L2-normalised).

One place to turn document text or a user question into a vector.
"""

from doctrace.config import get_device

_MODEL_NAME = "BAAI/bge-m3"
_model = None


def _get_model():
    """Load on first use. sentence_transformers pulls in torch, which costs seconds and
    a lot of memory, so importing this module stays cheap for callers that never encode."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME, device=get_device())
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
