from graphrag.answer.pipeline import answer_hybrid
from tests.fakes import (
    PASSAGE_ROW, T1_ROW, FakeConn, FakeEmbed, FakeOpenAI, FakeResolver, FakeSession,
)


def _answer(**overrides):
    kwargs = dict(
        conn=FakeConn([PASSAGE_ROW]), neo4j_session=FakeSession([T1_ROW]),
        openai_client=FakeOpenAI(), resolver=FakeResolver(), embedding_model=FakeEmbed(),
    )
    kwargs.update(overrides)
    return answer_hybrid("q", **kwargs)


def test_empty_graph_result_still_gets_vector_passages():
    # Routing GRAPH and getting nothing back must not mean answering with an
    # empty context — the baseline would have had passages, so this loses
    # outright. 15 of 60 benchmark questions hit exactly this.
    result = _answer(neo4j_session=FakeSession([]))

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
    result = _answer()

    assert result["route"] == "GRAPH"
    assert result["graph_facts_used"] == 1
    assert result["vector_passages_used"] == 1
    assert set(result["retrieved_ids"]) == {"g1", "c1"}


def test_refusal_makes_no_claims_and_cites_nothing():
    result = _answer(openai_client=FakeOpenAI(route="REFUSE"))

    assert result["refused"] is True
    assert result["citations_valid"] is True
    assert result["retrieved_ids"] == []


def test_the_answer_is_written_from_the_assembled_context():
    client = FakeOpenAI(answer="Session calls HTTPAdapter.send [g1].")

    result = _answer(openai_client=client)

    assert result["answer"] == "Session calls HTTPAdapter.send [g1]."
    assert result["citations_valid"] is True
    assert "=== GRAPH RELATIONSHIPS ===" in client.responses.last_input
    assert "=== RETRIEVED PASSAGES ===" in client.responses.last_input
