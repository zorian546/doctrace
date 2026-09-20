"""
Corpus acquisition: pull the FastAPI docs and read them into memory.

The tutorial/ and advanced/ sections are fetched as raw markdown through a sparse
git checkout and returned as {"source_path": ..., "text": ...} dicts.

The checkout is pinned to DOCS_COMMIT. The docs are edited often, and an unpinned
build quietly changes the corpus underneath the evaluation: a section added upstream
shifts every later passage id by one, which silently breaks the id agreement between
the keyword index and the vector store (see doctrace/corpus/manifest.py). With the pin,
`python -m scripts.build_index` reproduces data/processed/passages.json exactly, which
is asserted by tests/test_corpus_integrity.py.

To move to newer docs: bump DOCS_COMMIT, re-run the indexing script, re-upload the
vectors, and re-run the benchmarks. All three, or the numbers stop matching the corpus.

Worth knowing: these docs do not keep code inline. They reference it with a
snippet-include marker such as `{* ../../docs_src/first_steps/tutorial001_py310.py *}`,
and the code is stitched in at doc-build time. 80 of the 85 files in scope use it, so
`read_docs()` expands every marker into a real fenced block before any splitting happens.

A git checkout is used rather than the GitHub contents API because the unauthenticated
API allows only 60 requests an hour, and the upstream repo has changed orgs
(tiangolo/fastapi -> fastapi/fastapi).
"""

import re
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
# Pinned upstream commit (2026-07-21). Produces exactly 756 passages.
DOCS_COMMIT = "7d210a4a9f54f2744e50ce55c65eb852958478c5"
REPO_ROOT = Path("data/raw/fastapi-repo")
SPARSE_PATHS = ["docs/en/docs", "docs_src"]
SUBSET_DIRS = ["tutorial", "advanced"]

# Matches `{* path/to/file.py *}` and the highlighted form `{* path/to/file.py hl[3:16] *}`.
# 381 of the 394 markers in scope carry the hl[] suffix. It is irrelevant for search
# text, but the path still has to be captured cleanly or the marker would survive
# untouched inside the passage.
SNIPPET_RE = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def pull_docs(force: bool = False) -> None:
    """Fetch docs/ and docs_src/ at DOCS_COMMIT into data/raw/fastapi-repo.

    Only the pinned commit is downloaded, and only the two directories needed, so this
    stays small. Safe to call repeatedly: an existing checkout at the right commit is
    left alone unless force=True.
    """
    if REPO_ROOT.exists() and not force:
        if _checked_out_commit() == DOCS_COMMIT:
            return
        # a checkout from a previous pin would silently produce a different corpus
        shutil.rmtree(REPO_ROOT)
    elif REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)

    REPO_ROOT.mkdir(parents=True, exist_ok=True)
    _git("init", "--quiet", cwd=REPO_ROOT)
    _git("remote", "add", "origin", REPO_URL, cwd=REPO_ROOT)
    _git("sparse-checkout", "set", *SPARSE_PATHS, cwd=REPO_ROOT)
    # fetching a single commit by sha keeps this to one revision instead of all history
    _git("fetch", "--depth", "1", "--filter=blob:none", "origin", DOCS_COMMIT, cwd=REPO_ROOT)
    _git("checkout", "--quiet", DOCS_COMMIT, cwd=REPO_ROOT)


def _checked_out_commit() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            check=True, capture_output=True, text=True,
        )
        return done.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _inline_snippets(raw_text: str, snippet_base: Path) -> str:
    """Swap each `{* path *}` marker for the referenced code in a fenced block.

    The '../../docs_src/...' paths are NOT relative to the markdown file that holds
    them. A doc one directory deeper (tutorial/security/first-steps.md) uses the very
    same '../../' prefix as a top-level one (tutorial/first-steps.md), so they resolve
    against one fixed base: docs/en/, two levels above docs/en/docs/.
    """

    def _substitute(match: "re.Match[str]") -> str:
        rel_path = match.group(1)
        target = (snippet_base / rel_path).resolve()
        # the marker text comes from a third-party repo: never read outside the checkout
        if not target.is_relative_to(REPO_ROOT.resolve()):
            return f"[MISSING SNIPPET: {rel_path}]"
        try:
            code = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"[MISSING SNIPPET: {rel_path}]"
        lang = target.suffix.lstrip(".") or "text"
        return f"```{lang}\n{code}\n```"

    return SNIPPET_RE.sub(_substitute, raw_text)


def read_docs() -> list[dict]:
    """Read every .md file under tutorial/ and advanced/ with snippets expanded.

    Expects pull_docs() to have run. Each item is {"source_path": str, "text": str}.
    """
    docs_root = REPO_ROOT / "docs" / "en" / "docs"
    snippet_base = docs_root.parent  # docs/en/, see _inline_snippets
    found = []
    for section in SUBSET_DIRS:
        for md_file in sorted((docs_root / section).rglob("*.md")):
            raw = md_file.read_text(encoding="utf-8")
            expanded = _inline_snippets(raw, snippet_base)
            found.append({
                "source_path": md_file.relative_to(docs_root).as_posix(),
                "text": expanded,
            })
    return found
