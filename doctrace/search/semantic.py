"""
Vector search over passages, backed by a Qdrant Cloud collection of BGE-M3 vectors
and queried by cosine similarity.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from doctrace.vectors.encoder import encode_texts, encode_query

load_dotenv()

COLLECTION = "doctrace_passages"
VECTOR_DIM = 1024  # BGE-M3 output size

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL")
        if not url:
            raise RuntimeError("QDRANT_URL is not set. Copy .env.example to .env and fill it in.")
        _client = QdrantClient(
            url=url,
            api_key=os.getenv("QDRANT_API_KEY"),
            # follow the URL scheme, so a local http:// instance works as well as a cloud one
            timeout=60,  # the free tier can stall on big requests
        )
    return _client


def ensure_collection(recreate: bool = False) -> None:
    """Make sure the collection exists. With recreate=True it is dropped first,
    which is what you want after the splitting logic changes."""
    client = _get_client()
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def index_passages(passages: list[dict], batch_size: int = 64) -> None:
    """Encode passages and upload them in batches (one giant request tends to time
    out on the free tier).

    A passage's position in the list is its point id. lexical.py relies on the same
    positions, which is what lets fusion match hits across the two searches.
    """
    client = _get_client()
    vectors = encode_texts([p["text"] for p in passages])

    points = [
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "source_path": passage["source_path"],
                "header_path": passage["header_path"],
                "text": passage["text"],
            },
        )
        for i, (passage, vector) in enumerate(zip(passages, vectors))
    ]

    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=COLLECTION, points=points[start:start + batch_size])
        print(f"  Uploaded {min(start + batch_size, len(points))}/{len(points)} points")


def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    """Encode the query and return the top_k closest passages, as
    {"id", "score", "source_path", "header_path", "text"}."""
    hits = _get_client().query_points(
        collection_name=COLLECTION,
        query=encode_query(query),
        limit=top_k,
    ).points

    return [
        {
            "id": h.id,
            "score": h.score,
            "source_path": h.payload["source_path"],
            "header_path": h.payload["header_path"],
            "text": h.payload["text"],
        }
        for h in hits
    ]
