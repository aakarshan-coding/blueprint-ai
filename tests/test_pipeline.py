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
        text_format = kw["text_format"]
        if text_format is RouterDecision:
            return _Parsed(RouterDecision(route="GRAPH", confidence=0.95))
        # The planner's two calls (D59): the mention list, then a plan in a
        # schema built per question. Both are constructed through the class
        # the pipeline passed, so the fake never has to know its shape.
        if text_format.__name__ == "Mentions":
            return _Parsed(text_format(mentions=[{"surface": "Session", "package": "unknown"}]))
        return _Parsed(text_format(
            plan={"template_id": "T1_NEIGHBORS", "entity_id": "requests.sessions.Session"}
        ))
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


class _FakeSessionWithFacts:
    """Graph returns one T1_NEIGHBORS row — the common case, not the empty one."""
    def run(self, query, **params):
        class R:
            def data(self):
                return [{"relationship": "CALLS", "neighbor": "requests.adapters.HTTPAdapter.send",
                         "chunk_id": "g1", "outgoing": True}]
        return R()


def test_empty_graph_result_still_gets_vector_passages():
    # Routing GRAPH and getting nothing back must not mean answering with an
    # empty context — the baseline would have had passages, so this loses
    # outright. 15 of 60 benchmark questions hit exactly this.
    result = answer_hybrid(
        "q", conn=_FakeConn(ROWS), neo4j_session=_FakeSession(),
        openai_client=_FakeOpenAI(), resolver=_FakeResolver(),
        embedding_model=_FakeEmbed(),
    )

    assert result["vector_passages_used"] == 1
    assert result["graph_facts_used"] == 0
    assert "No supporting information" not in result["answer"]


def test_graph_route_with_facts_still_gets_vector_passages():
    """Graph facts are additive, never a substitute for passages.

    A GRAPH route that found facts used to withhold passages entirely. On 16
    of 60 benchmark questions the model then answered from a handful of
    triples the planner had chosen — usually for the wrong entity — while
    the baseline had five full passages. Hybrid scored 5 there; baseline 7.
    Hybrid must be baseline plus facts, so the benchmark measures whether
    facts help on top of passages rather than whether they can replace them.
    """
    result = answer_hybrid(
        "q", conn=_FakeConn(ROWS), neo4j_session=_FakeSessionWithFacts(),
        openai_client=_FakeOpenAI(), resolver=_FakeResolver(),
        embedding_model=_FakeEmbed(),
    )

    assert result["route"] == "GRAPH"
    assert result["graph_facts_used"] == 1
    assert result["vector_passages_used"] == 1
    assert set(result["retrieved_ids"]) == {"g1", "c1"}
