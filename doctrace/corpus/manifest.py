"""
Corpus fingerprint, and the check that keeps the two indexes in agreement.

A passage's id is its position in data/processed/passages.json. That one number is
used by both retrievers: the keyword index scores by list position, and the vector
store uses the same number as its point id. Fusion then merges the two result lists by
id. So the list uploaded to the vector store and the list loaded for keyword search
have to be the same list, in the same order.

Nothing enforces that on its own, and the failure is silent rather than loud. Re-running
the indexing script after the upstream docs gain a section renumbers everything after it;
querying an older collection then returns correct-looking ids that point at the wrong
text, and the answer is generated from passages nobody retrieved. It looks like a quality
problem, not a bug.

So the indexing script records a fingerprint of the passage list it wrote, and records it
again when those passages are uploaded. Semantic search compares the two before its first
query and refuses to run on a mismatch, with a message saying which step to re-run.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PATH = Path("data/processed/corpus_manifest.json")


def fingerprint(passages: list[dict]) -> str:
    """Order-sensitive digest of a passage list.

    Covers the identifying fields and the text, so reordering, inserting, or editing a
    passage all change the result.
    """
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(passage["source_path"].encode("utf-8"))
        digest.update(b"\x00")
        digest.update((passage.get("anchor") or "").encode("utf-8"))
        digest.update(b"\x00")
        digest.update(passage["text"].encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def read_manifest(path: Path = MANIFEST_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_manifest(passages: list[dict], docs_commit: str, path: Path = MANIFEST_PATH) -> dict:
    """Record the corpus that was just written, keeping any existing upload record."""
    existing = read_manifest(path) or {}
    manifest = {
        "passage_count": len(passages),
        "fingerprint": fingerprint(passages),
        "docs_commit": docs_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if "vector_index" in existing:
        manifest["vector_index"] = existing["vector_index"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def record_upload(passages: list[dict], collection: str, path: Path = MANIFEST_PATH) -> None:
    """Record which corpus was uploaded to which collection."""
    manifest = read_manifest(path) or {}
    manifest["vector_index"] = {
        "collection": collection,
        "fingerprint": fingerprint(passages),
        "passage_count": len(passages),
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class IndexMismatch(RuntimeError):
    """The local passages and the uploaded vectors are not the same corpus."""


def check_alignment(passages: list[dict], collection: str,
                    path: Path = MANIFEST_PATH) -> str | None:
    """Compare the local corpus against the recorded upload.

    Returns None when they agree. Returns a warning string when there is nothing to
    compare against (no manifest, or vectors uploaded before manifests existed), since
    that is not proof of a problem. Raises IndexMismatch when they demonstrably disagree.
    """
    manifest = read_manifest(path)
    if not manifest or "vector_index" not in manifest:
        return (f"No upload recorded in {path}, so the vector collection cannot be "
                f"checked against the local passages. Re-run "
                f"`python -m scripts.build_index --sync-qdrant` to record one.")

    uploaded = manifest["vector_index"]
    if uploaded.get("collection") != collection:
        return (f"The recorded upload was to collection {uploaded.get('collection')!r}, "
                f"but this process is querying {collection!r}.")

    local = fingerprint(passages)
    if local != uploaded.get("fingerprint"):
        raise IndexMismatch(
            f"data/processed/passages.json ({len(passages)} passages, fingerprint "
            f"{local[:12]}) does not match the vectors uploaded to {collection!r} "
            f"({uploaded.get('passage_count')} passages, fingerprint "
            f"{str(uploaded.get('fingerprint'))[:12]}). Passage ids would not line up, "
            f"so search results would point at the wrong text. Re-run "
            f"`python -m scripts.build_index --sync-qdrant --recreate-collection`."
        )
    return None
