from graphrag.ingest.chunk import (
    chunk_changelog,
    chunk_python,
    chunk_rst,
    iter_docstrings_for_llm,
    make_chunk_id,
)


def test_chunk_id_is_stable_across_calls():
    first = make_chunk_id("requests", "src/requests/sessions.py", 10, 42)
    second = make_chunk_id("requests", "src/requests/sessions.py", 10, 42)
    assert first == second


RST = """\
Advanced Usage
==============

This document covers advanced features.

Session Objects
---------------

The Session object persists parameters across requests.
"""


def test_chunk_rst_records_section_breadcrumbs():
    chunks = chunk_rst(RST, repo="requests", path="docs/user/advanced.rst")

    assert [c.section for c in chunks] == [
        "Advanced Usage",
        "Advanced Usage > Session Objects",
    ]


def _oversize_rst(paragraphs: int) -> str:
    body = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(paragraphs))
    return f"Big Section\n===========\n\n{body}\n"


def test_oversize_section_is_split_into_several_chunks():
    chunks = chunk_rst(
        _oversize_rst(10), repo="requests", path="docs/big.rst", max_chars=600
    )

    assert len(chunks) > 1
    assert all(len(c.text) <= 600 for c in chunks)


def test_split_chunks_keep_their_section_breadcrumb():
    chunks = chunk_rst(
        _oversize_rst(10), repo="requests", path="docs/big.rst", max_chars=600
    )

    assert {c.section for c in chunks} == {"Big Section"}


def test_split_chunks_get_distinct_ids():
    chunks = chunk_rst(
        _oversize_rst(10), repo="requests", path="docs/big.rst", max_chars=600
    )

    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_doc_chunks_embed_their_own_text():
    chunks = chunk_rst(RST, repo="requests", path="docs/user/advanced.rst")

    assert all(c.embed_text == c.text for c in chunks)


PY_SOURCE = '''import os


def helper():
    return 1


class Adapter:
    """An adapter."""

    def send(self, request):
        """Send a request."""
        return None

    def close(self):
        pass
'''


def _by_section(chunks):
    return {c.section: c for c in chunks}


def test_chunk_python_emits_one_chunk_per_symbol():
    chunks = chunk_python(PY_SOURCE, repo="requests", path="src/requests/adapters.py")

    assert [c.section for c in chunks] == [
        "helper",
        "Adapter",
        "Adapter.send",
        "Adapter.close",
    ]


def test_documented_symbol_embeds_signature_and_docstring_not_body():
    chunks = _by_section(
        chunk_python(PY_SOURCE, repo="requests", path="src/requests/adapters.py")
    )
    send = chunks["Adapter.send"]

    assert send.embed_text == "def send(self, request):\nSend a request."
    assert "return None" in send.text


def test_undocumented_symbol_falls_back_to_embedding_full_text():
    chunks = _by_section(
        chunk_python(PY_SOURCE, repo="requests", path="src/requests/adapters.py")
    )
    close = chunks["Adapter.close"]

    assert close.embed_text == close.text


CHANGELOG = """\
Release History
===============

2.34.2 (2026-05-14)
-------------------
- Moved headers input type back to Mapping.

2.34.1 (2026-05-13)
-------------------
- Widened json input type.
"""


def test_chunk_changelog_sections_by_release_version():
    chunks = chunk_changelog(CHANGELOG, repo="requests", path="HISTORY.md")

    assert [c.section for c in chunks] == ["2.34.2", "2.34.1"]


def test_iter_docstrings_for_llm_pairs_docstring_with_its_code_chunk_id():
    docs = iter_docstrings_for_llm(
        PY_SOURCE, repo="requests", path="src/requests/adapters.py"
    )
    code_chunks = _by_section(
        chunk_python(PY_SOURCE, repo="requests", path="src/requests/adapters.py")
    )

    by_symbol = {d.symbol: d for d in docs}
    assert by_symbol["Adapter"].chunk_id == code_chunks["Adapter"].chunk_id
    assert by_symbol["Adapter"].docstring == "An adapter."
    assert by_symbol["Adapter.send"].docstring == "Send a request."


def test_iter_docstrings_for_llm_skips_undocumented_symbols():
    docs = iter_docstrings_for_llm(
        PY_SOURCE, repo="requests", path="src/requests/adapters.py"
    )
    by_symbol = {d.symbol: d for d in docs}

    assert "Adapter.close" not in by_symbol
    assert "helper" not in by_symbol
