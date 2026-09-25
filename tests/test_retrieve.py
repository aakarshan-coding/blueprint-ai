"""retrieve(): the one retrieval sequence, used by the pipeline, the stability
probe and any audit (fix 12). It was written out three times by hand and
drifted twice. Now the measurement tool cannot measure a different pipeline
than the one that answers.

Also fix 3: when the question names nothing the resolver knows, the graph is
seeded from the passages -- each code chunk *is* a node, so the entities the
question is about are already in hand once search has run.
"""

from graphrag.retrieval.retrieve import RetrievalResult, retrieve
from tests.fakes import (
    DOC_ROW, PASSAGE_ROW, T1_ROW, FakeConn, FakeEmbed, FakeOpenAI, FakeResolver, FakeSession,
)


def _retrieve(**overrides):
    kwargs = dict(
        conn=FakeConn([PASSAGE_ROW]), neo4j_session=FakeSession([T1_ROW]),
        openai_client=FakeOpenAI(), resolver=FakeResolver(), embedding_model=FakeEmbed(),
    )
    kwargs.update(overrides)
    return retrieve("q", **kwargs)


def test_refusal_returns_no_context_and_no_calls_beyond_the_router():
    client = FakeOpenAI(route="REFUSE")

    result = _retrieve(openai_client=client)

    assert result.refused is True
    assert result.context == ""
    assert result.retrieved_ids == set()
    assert client.responses.parse_calls == 1


def test_graph_facts_and_passages_are_both_in_the_context():
    result = _retrieve()

    assert isinstance(result, RetrievalResult)
    assert result.route == "GRAPH"
    assert result.plan == "T1_NEIGHBORS({'entity_id': 'requests.sessions.Session'})"
    assert len(result.graph_facts) == 1
    assert len(result.passages) == 1
    assert result.retrieved_ids == {"g1", "c1"}
    assert "=== GRAPH RELATIONSHIPS ===" in result.context
    assert "=== RETRIEVED PASSAGES ===" in result.context


def test_a_vector_route_skips_the_graph_entirely():
    client = FakeOpenAI(route="VECTOR")

    result = _retrieve(openai_client=client)

    assert result.plan is None
    assert result.graph_facts == []
    assert len(result.passages) == 1
    assert client.responses.parse_calls == 1


def test_a_graph_failure_is_recorded_not_raised():
    class Boom(FakeSession):
        def run(self, query, **params):
            raise RuntimeError("neo4j down")

    result = _retrieve(neo4j_session=Boom())

    assert result.graph_error == "RuntimeError: neo4j down"
    assert result.graph_facts == []
    assert len(result.passages) == 1


# --- fix 3: seed the graph from the passages when the question names nothing -----

def test_when_nothing_resolves_the_passages_seed_the_graph():
    """"Which requests exception surfaces when a proxy fails?" names no
    class. The top passage is the send() handler chunk, whose section is
    "HTTPAdapter.send" -- a node. That becomes the candidate."""
    resolver = FakeResolver({"HTTPAdapter.send": "requests.adapters.HTTPAdapter.send"})
    client = FakeOpenAI(
        mentions=[{"surface": "proxy failure", "package": "unknown"}],
        plan={"template_id": "T1_NEIGHBORS", "entity_id": "requests.adapters.HTTPAdapter.send"},
    )
    row = ("c7", "requests", "src/requests/adapters.py", "HTTPAdapter.send", "code", "...", 0.1)

    result = _retrieve(conn=FakeConn([row]), resolver=resolver, openai_client=client)

    assert result.seeded_from_passages is True
    assert [c.canonical_id for c in result.candidates] == ["requests.adapters.HTTPAdapter.send"]
    assert result.plan == "T1_NEIGHBORS({'entity_id': 'requests.adapters.HTTPAdapter.send'})"


def test_doc_passages_do_not_seed_the_graph():
    # A doc section heading is not a code node; only code chunks seed.
    resolver = FakeResolver({})
    client = FakeOpenAI(mentions=[{"surface": "nothing", "package": "unknown"}])

    result = _retrieve(conn=FakeConn([DOC_ROW]), resolver=resolver, openai_client=client)

    assert result.seeded_from_passages is False
    assert result.candidates == []
    assert result.plan is None


def test_seeding_is_not_used_when_the_question_itself_resolves():
    result = _retrieve()

    assert result.seeded_from_passages is False
