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


def test_package_root_candidates_do_not_block_seeding():
    """"requests" and "urllib3" appear as mentions in most questions and
    resolve to the package Module node, so `candidates` was never empty and
    seeding fired on zero of 90 questions (D66). A candidate list that is
    only Modules is no anchor; the passages' symbols are."""
    resolver = FakeResolver({
        "requests": "requests",
        "HTTPAdapter.send": "requests.adapters.HTTPAdapter.send",
    })
    session = FakeSession([T1_ROW], descriptions={"requests": (["Module"], 65)})
    client = FakeOpenAI(
        mentions=[{"surface": "requests", "package": "requests"}],
        plan={"template_id": "T1_NEIGHBORS", "entity_id": "requests.adapters.HTTPAdapter.send"},
    )
    row = ("c7", "requests", "src/requests/adapters.py", "HTTPAdapter.send", "code", "...", 0.1)

    result = _retrieve(conn=FakeConn([row]), neo4j_session=session, resolver=resolver, openai_client=client)

    assert result.seeded_from_passages is True
    assert "requests.adapters.HTTPAdapter.send" in [c.canonical_id for c in result.candidates]


# --- D66 fix: the cap belongs on the unfiltered neighbourhood, not on T8 -----------

def test_a_related_by_plan_is_never_capped():
    """T8 is already filtered to one relationship; its rows *are* the answer.
    Capping at eight cut "which exceptions derive from RequestException"
    from fifteen to eight and dropped the ValueError ones (D66, 3h-04)."""
    rows = [
        {"relationship": "INHERITS_FROM", "neighbor": f"requests.exceptions.E{i}",
         "chunk_id": f"g{i}", "outgoing": False}
        for i in range(12)
    ]
    client = FakeOpenAI(plan={
        "template_id": "T8_RELATED_BY", "entity_id": "requests.sessions.Session",
        "relationship": "INHERITS_FROM",
    })

    result = _retrieve(neo4j_session=FakeSession(rows), openai_client=client)

    cited = [f for f in result.graph_facts if f.chunk_id is not None]
    assert len(cited) == 12


def test_a_neighbours_plan_is_still_capped():
    rows = [
        {"relationship": "CALLS", "neighbor": f"m.f{i}", "chunk_id": f"g{i}", "outgoing": True}
        for i in range(12)
    ]

    result = _retrieve(neo4j_session=FakeSession(rows))

    assert len(result.graph_facts) == 8


class _ChainThenParents(FakeSession):
    """The wrap-chain query returns `chain_rows`; the parents query returns
    the parents of whichever ids it was asked about."""

    def __init__(self, chain_rows, parents):
        super().__init__(chain_rows)
        self._parents = parents
        self.parent_queries = []

    def run(self, query, **params):
        if "$entity_ids" in query:
            self.parent_queries.append(params["entity_ids"])
            rows = [
                {"entity": e, "parent": p, "chunk_id": f"p-{e}", "source": "ast"}
                for e in params["entity_ids"] for p in self._parents.get(e, [])
            ]

            class R:
                def data(self_inner):
                    return rows

            return R()
        return super().run(query, **params)


def _wrap_plan():
    return FakeOpenAI(plan={
        "template_id": "T3_EXCEPTION_WRAP_CHAIN",
        "entity_id": "requests.sessions.Session", "max_hops": 2,
    })


def test_a_wrap_chain_is_followed_by_the_parents_of_the_nodes_it_kept():
    """"...and what does that inherit from?" ends eleven three-hop questions.
    The chain walks WRAPS_EXCEPTION only, so the answer model said "the
    parent class is not specified" in five of five runs (D70). The parents
    of the kept chain nodes are fetched afterwards, parser-tagged (D71)."""
    chain_rows = [{
        "chain": ["urllib3.exceptions.ProtocolError", "requests.exceptions.ConnectionError"],
        "chunk_ids": ["c1"], "starts": ["requests.exceptions.ConnectionError"],
        "sources": ["ast"],
    }]
    session = _ChainThenParents(chain_rows, parents={
        "requests.exceptions.ConnectionError": ["requests.exceptions.RequestException"],
    })

    result = _retrieve(neo4j_session=session, openai_client=_wrap_plan())

    statements = [f.statement for f in result.graph_facts]
    assert "requests.exceptions.ConnectionError inherits from requests.exceptions.RequestException." in statements
    assert result.expanded == 1
    assert "(code)" in result.context
    assert set(session.parent_queries[0]) == {
        "urllib3.exceptions.ProtocolError", "requests.exceptions.ConnectionError"}


def test_parents_are_fetched_only_for_nodes_that_survived_the_cap():
    """Expansion runs after the cap, not before: twelve chain facts are cut
    to eight first, and only those eight's nodes are asked about. Otherwise
    the parents would compete with the chain they explain for the same
    eight slots."""
    chain_rows = [{
        "chain": [f"urllib3.exceptions.E{i}", f"requests.exceptions.R{i}"],
        "chunk_ids": [f"c{i}"], "starts": [f"requests.exceptions.R{i}"],
    } for i in range(12)]
    parents = {f"requests.exceptions.R{i}": ["requests.exceptions.RequestException"] for i in range(12)}
    session = _ChainThenParents(chain_rows, parents=parents)

    result = _retrieve(neo4j_session=session, openai_client=_wrap_plan())

    asked = session.parent_queries[0]
    assert len(asked) == 16, "8 kept facts x 2 nodes"
    chain_facts = [f for f in result.graph_facts if f.relationship == "WRAPS_EXCEPTION"]
    assert len(chain_facts) == 8
    assert result.expanded == 8


def test_a_neighbours_plan_does_not_expand():
    session = _ChainThenParents([T1_ROW], parents={"m.f": ["m.base"]})

    result = _retrieve(neo4j_session=session)

    assert session.parent_queries == []
    assert result.expanded == 0
