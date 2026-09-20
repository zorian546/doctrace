"""
Corpus acquisition: pull the FastAPI docs and read them into memory.

The tutorial/ and advanced/ sections are fetched as raw markdown through a sparse
git clone and returned as {"source_path": ..., "text": ...} dicts.

Worth knowing: these docs do not keep code inline. They reference it with a
snippet-include marker such as `{* ../../docs_src/first_steps/tutorial001_py310.py *}`,
and the code is stitched in at doc-build time. 80 of the 85 files in scope use it, so
`read_docs()` expands every marker into a real fenced block before any splitting happens.

A git clone is used rather than the GitHub contents API because the unauthenticated
API allows only 60 requests an hour, and the upstream repo has changed orgs
(tiangolo/fastapi -> fastapi/fastapi).
"""

import re
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
REPO_ROOT = Path("data/raw/fastapi-repo")
SPARSE_PATHS = ["docs/en/docs", "docs_src"]
SUBSET_DIRS = ["tutorial", "advanced"]

# Matches `{* path/to/file.py *}` and the highlighted form `{* path/to/file.py hl[3:16] *}`.
# 381 of the 394 markers in scope carry the hl[] suffix. It is irrelevant for search
# text, but the path still has to be captured cleanly or the marker would survive
# untouched inside the passage.
SNIPPET_RE = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")


def pull_docs(force: bool = False) -> None:
    """Sparse-clone docs/ and docs_src/ into data/raw/fastapi-repo.

    Safe to call repeatedly: an existing checkout is left alone unless force=True.
    """
    if REPO_ROOT.exists():
        if not force:
            return
        shutil.rmtree(REPO_ROOT)

    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "--filter=blob:none", "--sparse",
            REPO_URL, str(REPO_ROOT),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", *SPARSE_PATHS],
        cwd=REPO_ROOT,
        check=True,
    )


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
