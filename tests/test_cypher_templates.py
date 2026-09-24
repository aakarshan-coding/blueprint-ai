import pytest

from graphrag.retrieval.cypher_templates import run_template


class _FakeResult:
    def __init__(self, records):
        self._records = records

    def data(self):
        return self._records


class _FakeSession:
    def __init__(self, records=()):
        self.calls = []
        self._records = list(records)

    def run(self, query, **params):
        self.calls.append((query, params))
        return _FakeResult(self._records)


KNOWN_IDS = {"requests.sessions.Session", "urllib3.poolmanager.PoolManager"}


def test_neighbors_template_runs_with_valid_entity_id():
    session = _FakeSession()
    run_template(
        session,
        "T1_NEIGHBORS",
        {"entity_id": "requests.sessions.Session"},
        known_entity_ids=KNOWN_IDS,
    )

    query, params = session.calls[0]
    assert params["entity_id"] == "requests.sessions.Session"


def test_neighbors_template_rejects_an_entity_id_the_question_never_resolved():
    session = _FakeSession()
    with pytest.raises(ValueError):
        run_template(
            session,
            "T1_NEIGHBORS",
            {"entity_id": "made.up.Thing"},
            known_entity_ids=KNOWN_IDS,
        )

    assert session.calls == []


def test_unknown_template_id_is_rejected():
    session = _FakeSession()
    with pytest.raises(ValueError):
        run_template(session, "T99_NOT_REAL", {}, known_entity_ids=KNOWN_IDS)

    assert session.calls == []


def test_missing_required_param_is_rejected():
    session = _FakeSession()
    with pytest.raises(ValueError):
        run_template(session, "T1_NEIGHBORS", {}, known_entity_ids=KNOWN_IDS)

    assert session.calls == []


def test_exception_wrap_chain_rejects_an_out_of_range_hop_limit():
    session = _FakeSession()
    with pytest.raises(ValueError):
        run_template(
            session,
            "T3_EXCEPTION_WRAP_CHAIN",
            {"entity_id": "requests.sessions.Session", "max_hops": 10},
            known_entity_ids=KNOWN_IDS,
        )

    assert session.calls == []


def test_exception_wrap_chain_accepts_a_valid_hop_limit():
    session = _FakeSession()
    run_template(
        session,
        "T3_EXCEPTION_WRAP_CHAIN",
        {"entity_id": "requests.sessions.Session", "max_hops": 3},
        known_entity_ids=KNOWN_IDS,
    )

    query, params = session.calls[0]
    assert "*1..3" in query
    assert "max_hops" not in params


def test_count_by_rel_rejects_a_relationship_type_outside_the_ontology():
    session = _FakeSession()
    with pytest.raises(ValueError):
        run_template(
            session,
            "T6_COUNT_BY_REL",
            {"relationship": "FRIENDS_WITH"},
            known_entity_ids=KNOWN_IDS,
        )

    assert session.calls == []


def test_count_by_rel_accepts_a_real_ontology_relationship():
    session = _FakeSession()
    run_template(
        session, "T6_COUNT_BY_REL", {"relationship": "WRAPS_EXCEPTION"},
        known_entity_ids=KNOWN_IDS,
    )

    query, params = session.calls[0]
    assert ":WRAPS_EXCEPTION" in query


def test_run_template_returns_the_query_results():
    session = _FakeSession(records=[{"neighbor": "urllib3.poolmanager.PoolManager"}])
    result = run_template(
        session, "T1_NEIGHBORS", {"entity_id": "requests.sessions.Session"},
        known_entity_ids=KNOWN_IDS,
    )

    assert result == [{"neighbor": "urllib3.poolmanager.PoolManager"}]


def test_exception_wrap_chain_never_puts_max_hops_in_the_param_dict():
    # Neo4j rejects a parameter as a variable-length relationship bound
    # ("*1..$max_hops" is a real CypherSyntaxError, confirmed against a live
    # instance) — the validated integer must be interpolated into the query
    # text instead, so it must never be sent to session.run as a parameter.
    session = _FakeSession()
    run_template(
        session,
        "T3_EXCEPTION_WRAP_CHAIN",
        {"entity_id": "requests.sessions.Session", "max_hops": 3},
        known_entity_ids=KNOWN_IDS,
    )

    query, params = session.calls[0]
    assert "max_hops" not in params
    assert "*1..3" in query


def test_exception_wrap_chain_returns_chunk_id_per_hop():
    # Without this, a multi-hop statement built from the chain has nothing
    # to cite — the whole point of the chunk_id join key.
    session = _FakeSession()
    run_template(
        session,
        "T3_EXCEPTION_WRAP_CHAIN",
        {"entity_id": "requests.sessions.Session", "max_hops": 2},
        known_entity_ids=KNOWN_IDS,
    )
    query, _params = session.calls[0]
    assert "chunk_id" in query.lower()


def test_delegation_chain_returns_chunk_id_per_hop():
    session = _FakeSession()
    run_template(
        session,
        "T5_DELEGATION_CHAIN",
        {"entity_id": "requests.sessions.Session", "max_hops": 2},
        known_entity_ids=KNOWN_IDS,
    )
    query, _params = session.calls[0]
    assert "chunk_id" in query.lower()


def test_related_by_filters_to_one_relationship_and_reports_direction():
    session = _FakeSession()
    run_template(
        session, "T8_RELATED_BY",
        {"entity_id": "requests.sessions.Session", "relationship": "INHERITS_FROM"},
        known_entity_ids=KNOWN_IDS,
    )

    query, _params = session.calls[0]
    assert ":INHERITS_FROM" in query
    assert "outgoing" in query


def test_related_by_rejects_a_relationship_outside_the_ontology():
    session = _FakeSession()
    with pytest.raises(ValueError):
        run_template(
            session, "T8_RELATED_BY",
            {"entity_id": "requests.sessions.Session", "relationship": "MADE_UP"},
            known_entity_ids=KNOWN_IDS,
        )
    assert session.calls == []


def test_wrap_chain_traverses_both_directions():
    # "what does requests raise when urllib3 raises ProtocolError" asks what
    # WRAPS ProtocolError — the incoming direction. Edges are stored as
    # "requests exception WRAPS urllib3 exception", so an outgoing-only walk
    # misses the flagship question shape entirely.
    session = _FakeSession()
    run_template(
        session, "T3_EXCEPTION_WRAP_CHAIN",
        {"entity_id": "requests.sessions.Session", "max_hops": 2},
        known_entity_ids=KNOWN_IDS,
    )
    query, _ = session.calls[0]

    assert "]->(" not in query, "must not be direction-restricted"
    assert "WRAPS_EXCEPTION" in query


def test_delegation_chain_walks_calls_as_well_as_delegates_to():
    """"What does requests.get ultimately hand off to" is a chain of CALLS:
    get -> request -> Session.request -> Session.send -> adapter.send ->
    conn.urlopen. DELEGATES_TO is the LLM's prose-derived version of the
    same idea and is sparse; walking it alone returned 0 facts for that
    question (planner audit, 3h-02). Now that CALLS edges are inferred by
    jedi (D58) the chain exists in the graph, and the template must walk it.
    Each hop reports its own type so the verbalizer can say which it was."""
    session = _FakeSession()
    run_template(
        session, "T5_DELEGATION_CHAIN",
        {"entity_id": "requests.sessions.Session", "max_hops": 2},
        known_entity_ids=KNOWN_IDS,
    )
    query, _ = session.calls[0]

    assert "DELEGATES_TO|CALLS" in query
    assert "AS rels" in query
