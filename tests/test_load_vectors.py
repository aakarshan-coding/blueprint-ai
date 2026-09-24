from graphrag.ingest.chunk import Chunk
from graphrag.ingest.load_vectors import upsert_chunks


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def executemany(self, query, rows):
        self.calls.append((query, rows))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def _make_chunk(chunk_id="c1", embed_text="hello") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, repo="requests", path="src/requests/adapters.py",
        kind="code", section="HTTPAdapter.send", start_line=1, end_line=5,
        text="def send(): pass", embed_text=embed_text,
    )


def test_upsert_chunks_uses_on_conflict_so_reruns_are_idempotent():
    conn = _FakeConn()
    upsert_chunks(conn, [_make_chunk()], embeddings=[[0.1] * 384])

    query, rows = conn.cursor_obj.calls[0]
    assert "ON CONFLICT" in query
    assert "chunk_id" in query


def test_upsert_chunks_passes_one_row_per_chunk_in_order():
    conn = _FakeConn()
    chunks = [_make_chunk("c1"), _make_chunk("c2")]
    upsert_chunks(conn, chunks, embeddings=[[0.1] * 384, [0.2] * 384])

    _query, rows = conn.cursor_obj.calls[0]
    assert len(rows) == 2
    assert rows[0]["chunk_id"] == "c1"
    assert rows[1]["chunk_id"] == "c2"


def test_upsert_chunks_commits():
    conn = _FakeConn()
    upsert_chunks(conn, [_make_chunk()], embeddings=[[0.1] * 384])

    assert conn.committed is True


def test_upsert_chunks_rejects_mismatched_lengths():
    import pytest

    conn = _FakeConn()
    with pytest.raises(ValueError):
        upsert_chunks(conn, [_make_chunk(), _make_chunk("c2")], embeddings=[[0.1] * 384])
