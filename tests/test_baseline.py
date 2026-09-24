from graphrag.eval.baseline import answer_vector_only


class _FakeEmbedModel:
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


class _FakeResponse:
    def __init__(self, text):
        self.output_text = text


class _FakeResponses:
    def __init__(self, text):
        self._text = text
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return _FakeResponse(self._text)


class _FakeOpenAI:
    def __init__(self, text):
        self.responses = _FakeResponses(text)


ROWS = [("c1", "requests", "p.py", "sec", "doc", "Some passage text.", 0.1)]


def test_baseline_never_includes_graph_facts():
    conn = _FakeConn(ROWS)
    client = _FakeOpenAI("An answer [c1].")

    result = answer_vector_only(
        "q", conn=conn, openai_client=client, embedding_model=_FakeEmbedModel()
    )

    assert result["graph_facts_used"] == 0
    assert "GRAPH FACTS" not in client.responses.last_call["input"]


def test_baseline_still_validates_citations_the_same_way():
    conn = _FakeConn(ROWS)
    client = _FakeOpenAI("An answer [c1].")

    result = answer_vector_only(
        "q", conn=conn, openai_client=client, embedding_model=_FakeEmbedModel()
    )

    assert result["citations_valid"] is True


def test_baseline_flags_an_invented_citation_just_like_the_hybrid_path():
    conn = _FakeConn(ROWS)
    client = _FakeOpenAI("An answer [not_retrieved].")

    result = answer_vector_only(
        "q", conn=conn, openai_client=client, embedding_model=_FakeEmbedModel()
    )

    assert result["citations_valid"] is False
    assert result["invalid_citations"] == ["not_retrieved"]


def test_baseline_reports_how_many_passages_it_used():
    conn = _FakeConn(ROWS)
    client = _FakeOpenAI("answer [c1]")

    result = answer_vector_only(
        "q", conn=conn, openai_client=client, embedding_model=_FakeEmbedModel()
    )

    assert result["vector_passages_used"] == 1
