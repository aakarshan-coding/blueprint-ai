from graphrag.answer.pipeline import answer_hybrid
from graphrag.retrieval.router import RouterDecision


class _FakeEmbed:
    def encode(self, texts, **kwargs):
        return [[0.1] * 384 for _ in texts]


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
    def execute(self, q, p): pass
    def fetchall(self): return self._rows
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeConn:
    def __init__(self, rows): self._rows = rows
    def cursor(self): return _FakeCursor(self._rows)


class _FakeSession:
    """Graph returns nothing — the case that broke 25% of the benchmark."""
    def run(self, query, **params):
        class R:
            def data(self): return []
        return R()


class _Parsed:
    def __init__(self, v): self.output_parsed = v
class _Text:
    def __init__(self, t): self.output_text = t


class _FakeResponses:
    def __init__(self): self.parse_calls = 0
    def parse(self, **kw):
        self.parse_calls += 1
        if kw["text_format"] is RouterDecision:
            return _Parsed(RouterDecision(route="GRAPH", confidence=0.95))
        from graphrag.retrieval.graph_query import GraphQueryPlan
        return _Parsed(GraphQueryPlan(template_id="T1_NEIGHBORS", entity_surface="Session"))
    def create(self, **kw):
        self.last_input = kw["input"]
        return _Text("An answer [c1].")


class _FakeOpenAI:
    def __init__(self): self.responses = _FakeResponses()


class _FakeResolver:
    def resolve(self, surface, **kw):
        class R: canonical_id = "requests.sessions.Session"; candidates = ()
        return R()


ROWS = [("c1", "requests", "p.py", "sec", "doc", "A useful passage.", 0.1)]


def test_empty_graph_result_falls_back_to_vector_instead_of_answering_with_nothing():
    # Routing GRAPH and getting nothing back must not mean answering with an
    # empty context — the baseline would have had passages, so this loses
    # outright. 15 of 60 benchmark questions hit exactly this.
    result = answer_hybrid(
        "q", conn=_FakeConn(ROWS), neo4j_session=_FakeSession(),
        openai_client=_FakeOpenAI(), resolver=_FakeResolver(),
        embedding_model=_FakeEmbed(),
    )

    assert result["vector_passages_used"] == 1
    assert "No supporting information" not in result["answer"]
    assert result["fell_back_to_vector"] is True
