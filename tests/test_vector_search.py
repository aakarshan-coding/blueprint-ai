from graphrag.retrieval.vector_search import search_chunks


class _FakeModel:
    def encode(self, texts, **kwargs):
        return [[0.1] * 384 for _ in texts]


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def execute(self, query, params):
        self.calls.append((query, params))

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, rows):
        self.cursor_obj = _FakeCursor(rows)

    def cursor(self):
        return self.cursor_obj


def test_search_chunks_embeds_the_question_with_the_injected_model():
    conn = _FakeConn(rows=[])
    search_chunks("What does verify do?", conn=conn, model=_FakeModel())

    _query, params = conn.cursor_obj.calls[0]
    assert len(params["qvec"]) == 384


def test_search_chunks_defaults_to_k_five():
    conn = _FakeConn(rows=[])
    search_chunks("What does verify do?", conn=conn, model=_FakeModel())

    query, params = conn.cursor_obj.calls[0]
    assert params["k"] == 5


def test_search_chunks_respects_a_custom_k():
    conn = _FakeConn(rows=[])
    search_chunks("q", conn=conn, model=_FakeModel(), k=10)

    _query, params = conn.cursor_obj.calls[0]
    assert params["k"] == 10


def test_search_chunks_returns_rows_as_dicts():
    conn = _FakeConn(
        rows=[("c1", "requests", "path.py", "sec", "code", "text", 0.1)]
    )
    results = search_chunks("q", conn=conn, model=_FakeModel())

    assert results[0]["chunk_id"] == "c1"
    assert results[0]["text"] == "text"
