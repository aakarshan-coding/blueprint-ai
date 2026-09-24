from graphrag.ingest.chunk import Chunk
from graphrag.ingest.doc_release_index import build_doc_release_index


def _chunk(chunk_id, kind, section, repo="requests"):
    return Chunk(
        chunk_id=chunk_id, repo=repo, path="p", kind=kind, section=section,
        start_line=1, end_line=2, text="t", embed_text="t",
    )


def test_doc_section_is_findable_by_its_leaf_heading():
    docs, _rel = build_doc_release_index([
        _chunk("abc123", "doc", "Advanced Usage > Session Objects"),
    ])

    assert docs["session objects"] == "doc:abc123"


def test_doc_section_is_findable_by_its_full_breadcrumb():
    docs, _rel = build_doc_release_index([
        _chunk("abc123", "doc", "Advanced Usage > Session Objects"),
    ])

    assert docs["advanced usage > session objects"] == "doc:abc123"


def test_release_is_findable_by_version_and_scoped_by_repo():
    _docs, rel = build_doc_release_index([
        _chunk("c1", "changelog", "2.34.2", repo="requests"),
    ])

    assert rel["2.34.2"] == "requests:2.34.2"


def test_a_heading_appearing_in_two_repos_is_dropped_not_guessed():
    # "Timeouts" exists in both requests and urllib3 docs. Picking one would
    # attach a fact to the wrong library.
    docs, _rel = build_doc_release_index([
        _chunk("r1", "doc", "Advanced Usage > Timeouts", repo="requests"),
        _chunk("u1", "doc", "User Guide > Timeouts", repo="urllib3"),
    ])

    assert "timeouts" not in docs
    # the unambiguous full breadcrumbs still resolve
    assert docs["advanced usage > timeouts"] == "doc:r1"
    assert docs["user guide > timeouts"] == "doc:u1"


def test_code_chunks_are_ignored():
    docs, rel = build_doc_release_index([_chunk("x", "code", "Session.request")])

    assert docs == {} and rel == {}
