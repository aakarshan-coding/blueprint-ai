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


_QUESTION = ("What do Session, requests, requests.exceptions and ProxyError do "
             "on a proxy failure, or nothing?")


def _retrieve(**overrides):
    kwargs = dict(
        conn=FakeConn([PASSAGE_ROW]), neo4j_session=FakeSession([T1_ROW]),
        openai_client=FakeOpenAI(), resolver=FakeResolver(), embedding_model=FakeEmbed(),
    )
    kwargs.update(overrides)
    # Mentions the question does not contain are dropped (D76), so the fake
    # question names every surface the fakes use.
    return retrieve(_QUESTION, **kwargs)


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
    """The plan's query returns `chain_rows`; an expansion query returns the
    outgoing edges of that relationship for whichever ids it was asked
    about: `parents` for INHERITS_FROM, `wraps` for WRAPS_EXCEPTION,
    `raises` for RAISES."""

    def __init__(self, chain_rows, parents=None, wraps=None, raises=None):
        super().__init__(chain_rows)
        self._edges = {
            ":INHERITS_FROM": parents or {}, ":WRAPS_EXCEPTION": wraps or {}, ":RAISES": raises or {},
        }
        self.parent_queries = []
        self.expansion_queries = []

    def run(self, query, **params):
        if "$entity_ids" in query:
            rel = next(r for r in self._edges if r in query)
            self.expansion_queries.append((rel, params["entity_ids"]))
            if rel == ":INHERITS_FROM":
                self.parent_queries.append(params["entity_ids"])
            rows = [
                {"entity": e, "neighbor": p, "chunk_id": f"p-{e}", "source": "ast"}
                for e in params["entity_ids"] for p in self._edges[rel].get(e, [])
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


def test_a_modules_members_are_followed_by_their_parents():
    """"Which classes in requests.exceptions are warnings?" needs each
    member's INHERITS_FROM edge, which the members query does not fetch
    (D73). Same post-cap expansion as the wrap chain (D71)."""
    rows = [
        {"relationship": "DEFINED_IN", "neighbor": "requests.exceptions.RequestsWarning",
         "chunk_id": "c1", "outgoing": False, "entity_labels": ["Module"], "neighbor_labels": ["Class"]},
    ]
    session = _ChainThenParents(rows, parents={
        "requests.exceptions.RequestsWarning": ["builtins.Warning"],
    })
    client = FakeOpenAI(plan={
        "template_id": "T8_RELATED_BY", "entity_id": "requests.sessions.Session",
        "relationship": "DEFINED_IN",
    })

    result = _retrieve(neo4j_session=session, openai_client=client)

    statements = [f.statement for f in result.graph_facts]
    assert "requests.exceptions.RequestsWarning inherits from builtins.Warning." in statements
    assert result.expanded == 1


def test_a_related_by_plan_on_another_relationship_does_not_expand():
    """CALLS has no expansion registered; only the shapes in _EXPANSIONS do."""
    rows = [{"relationship": "CALLS", "neighbor": "requests.adapters.HTTPAdapter.send",
             "chunk_id": "c1", "outgoing": True}]
    session = _ChainThenParents(rows, parents={"requests.adapters.HTTPAdapter.send": ["x"]})
    client = FakeOpenAI(plan={
        "template_id": "T8_RELATED_BY", "entity_id": "requests.sessions.Session",
        "relationship": "CALLS",
    })

    result = _retrieve(neo4j_session=session, openai_client=client)

    assert session.parent_queries == []


def test_a_wrong_relationship_on_a_module_anchor_is_repaired_before_running():
    """The planner chose INHERITS_FROM on the module requests.exceptions
    (D73, live). A Module has no such edges; the repair runs DEFINED_IN and
    records why, and the plan string shows what actually ran."""
    resolver = FakeResolver({"requests.exceptions": "requests.exceptions"})
    session = FakeSession([], descriptions={"requests.exceptions": (["Module"], 40)})
    client = FakeOpenAI(
        mentions=[{"surface": "requests.exceptions", "package": "requests"}],
        plan={"template_id": "T8_RELATED_BY", "entity_id": "requests.exceptions",
              "relationship": "INHERITS_FROM"},
    )

    result = _retrieve(neo4j_session=session, resolver=resolver, openai_client=client)

    assert result.plan_repair
    assert "'DEFINED_IN'" in result.plan
    ran = [q for q, _ in session.queries if ":DEFINED_IN" in q]
    assert ran, "the repaired plan is what ran"


def test_a_wrap_chain_fetches_grandparents_from_the_parents_it_added():
    """"...what is its parent class, and what is that parent's parent?"
    (3h-24). The second hop starts from what the first added, not from
    every node again."""
    chain_rows = [{
        "chain": ["urllib3.exceptions.ReadTimeoutError", "requests.exceptions.ReadTimeout"],
        "chunk_ids": ["c1"], "starts": ["requests.exceptions.ReadTimeout"],
    }]
    session = _ChainThenParents(chain_rows, parents={
        "requests.exceptions.ReadTimeout": ["requests.exceptions.Timeout"],
        "requests.exceptions.Timeout": ["requests.exceptions.RequestException"],
    })

    result = _retrieve(neo4j_session=session, openai_client=_wrap_plan())

    statements = [f.statement for f in result.graph_facts]
    assert "requests.exceptions.Timeout inherits from requests.exceptions.RequestException." in statements
    assert result.expanded == 2
    assert session.parent_queries[1] == ["requests.exceptions.ReadTimeout", "requests.exceptions.Timeout"]


def test_a_raises_plan_fetches_what_each_exception_wraps_and_inherits():
    """"Which urllib3 error is caught in iter_content, which requests
    exception replaces it, and what does that inherit from" is RAISES from
    the function, then WRAPS_EXCEPTION and INHERITS_FROM from each raised
    exception (D74)."""
    rows = [{"relationship": "RAISES", "neighbor": "requests.exceptions.ChunkedEncodingError",
             "chunk_id": "c1", "outgoing": True}]
    session = _ChainThenParents(
        rows,
        wraps={"requests.exceptions.ChunkedEncodingError": ["urllib3.exceptions.ProtocolError"]},
        parents={"requests.exceptions.ChunkedEncodingError": ["requests.exceptions.RequestException"]},
    )
    client = FakeOpenAI(plan={
        "template_id": "T8_RELATED_BY", "entity_id": "requests.sessions.Session",
        "relationship": "RAISES",
    })

    result = _retrieve(neo4j_session=session, openai_client=client)

    statements = [f.statement for f in result.graph_facts]
    assert "requests.exceptions.ChunkedEncodingError wraps urllib3.exceptions.ProtocolError." in statements
    assert "requests.exceptions.ChunkedEncodingError inherits from requests.exceptions.RequestException." in statements
    # wraps, parents, then the parents' parents (D78)
    assert [rel for rel, _ in session.expansion_queries] == [":WRAPS_EXCEPTION", ":INHERITS_FROM", ":INHERITS_FROM"]


def test_a_call_chain_fetches_what_the_callees_raise():
    """"...which classmethod turns max_retries into it, and what does that
    class raise when attempts run out" (3h-23): RAISES from the chain's
    nodes (D74)."""
    chain_rows = [{
        "chain": ["requests.adapters.HTTPAdapter.__init__", "urllib3.util.retry.Retry.from_int"],
        "chunk_ids": ["c1"], "starts": ["requests.adapters.HTTPAdapter.__init__"], "rels": ["CALLS"],
    }]
    session = _ChainThenParents(chain_rows, raises={
        "urllib3.util.retry.Retry.from_int": ["urllib3.exceptions.MaxRetryError"],
    })
    client = FakeOpenAI(plan={
        "template_id": "T5_DELEGATION_CHAIN", "entity_id": "requests.sessions.Session", "max_hops": 2,
    })

    result = _retrieve(neo4j_session=session, openai_client=client)

    statements = [f.statement for f in result.graph_facts]
    assert "urllib3.util.retry.Retry.from_int raises urllib3.exceptions.MaxRetryError." in statements


def test_a_raised_by_methods_plan_is_uncapped_and_expands_wraps_and_parents():
    rows = [
        {"entity": f"requests.sessions.Session.m{i}", "neighbor": f"requests.exceptions.E{i}",
         "chunk_id": f"c{i}", "source": "ast"}
        for i in range(10)
    ]
    session = _ChainThenParents(rows, wraps={"requests.exceptions.E0": ["urllib3.exceptions.X"]},
                                parents={"requests.exceptions.E0": ["requests.exceptions.RequestException"]})
    client = FakeOpenAI(
        mentions=[{"surface": "Session", "package": "requests"}],
        plan={"template_id": "T8_RELATED_BY", "entity_id": "requests.sessions.Session",
              "relationship": "RAISES"},
    )
    session_desc = FakeSession(rows, descriptions={"requests.sessions.Session": (["Class"], 40)})
    session._descriptions = session_desc._descriptions

    result = _retrieve(neo4j_session=session, openai_client=client)

    assert result.plan.startswith("T10_RAISED_BY_METHODS_OF")
    raised = [f for f in result.graph_facts if f.relationship == "RAISES"]
    assert len(raised) == 10, "rows are the answer: not capped"
    statements = [f.statement for f in result.graph_facts]
    assert "requests.exceptions.E0 wraps urllib3.exceptions.X." in statements
    assert "requests.exceptions.E0 inherits from requests.exceptions.RequestException." in statements


def test_subclasses_are_followed_by_what_they_wrap():
    """From the anchor Timeout: ConnectTimeout and ReadTimeout inherit from
    it, and each wraps a urllib3 error; that is "what does requests convert
    a socket timeout to" (3h-03, D76)."""
    rows = [{"relationship": "INHERITS_FROM", "neighbor": "requests.exceptions.ConnectTimeout",
             "chunk_id": "c1", "outgoing": False}]
    session = _ChainThenParents(rows, wraps={
        "requests.exceptions.ConnectTimeout": ["urllib3.exceptions.ConnectTimeoutError"]})
    client = FakeOpenAI(plan={
        "template_id": "T8_RELATED_BY", "entity_id": "requests.sessions.Session",
        "relationship": "INHERITS_FROM",
    })

    result = _retrieve(neo4j_session=session, openai_client=client)

    statements = [f.statement for f in result.graph_facts]
    assert "requests.exceptions.ConnectTimeout wraps urllib3.exceptions.ConnectTimeoutError." in statements


def test_repair_sees_the_kind_of_a_candidate_seeded_from_passages():
    """When nothing in the question resolves, the anchor comes from the
    passages. Its kind must reach repair_plan too: a seeded HTTPAdapter
    anchoring a wrap chain went unrepaired (3h-03, D76)."""
    resolver = FakeResolver({"HTTPAdapter": "requests.adapters.HTTPAdapter"})
    session = _ChainThenParents([], raises={})
    session._descriptions = {"requests.adapters.HTTPAdapter": (["Class"], 30)}
    client = FakeOpenAI(
        mentions=[{"surface": "nothing", "package": "unknown"}],
        plan={"template_id": "T3_EXCEPTION_WRAP_CHAIN",
              "entity_id": "requests.adapters.HTTPAdapter", "max_hops": 2},
    )
    row = ("c7", "requests", "src/requests/adapters.py", "HTTPAdapter", "code", "...", 0.1)

    result = _retrieve(conn=FakeConn([row]), neo4j_session=session, resolver=resolver, openai_client=client)

    assert result.seeded_from_passages is True
    # The chain ran (and returned nothing), then what the class's methods
    # raise was tried; in this fake that is empty too, so the members
    # fallback (D82) is what finally ran. The point here is the first
    # fallback fired at all: it needs the seeded candidate's kind.
    queries = [q for q, _ in session.queries]
    assert any("WRAPS_EXCEPTION" in q for q in queries)
    assert any("[r:RAISES]" in q for q in queries), "T10 was attempted"
    assert "returned nothing" in result.plan_repair


def test_a_wrap_chain_that_returns_rows_is_never_replaced():
    """The flagship question "if urllib3 raises ProtocolError, what does
    requests raise" anchors on a class labelled Class alone. The chain
    returns rows, so nothing falls back (D77)."""
    chain_rows = [{
        "chain": ["urllib3.exceptions.ProtocolError", "requests.exceptions.ConnectionError"],
        "chunk_ids": ["c1"], "starts": ["requests.exceptions.ConnectionError"], "sources": ["ast"],
    }]
    resolver = FakeResolver({"ProxyError": "urllib3.exceptions.ProtocolError"})
    session = _ChainThenParents(chain_rows, parents={})
    session._descriptions = {"urllib3.exceptions.ProtocolError": (["Class"], 6)}
    client = FakeOpenAI(
        mentions=[{"surface": "ProxyError", "package": "urllib3"}],
        plan={"template_id": "T3_EXCEPTION_WRAP_CHAIN",
              "entity_id": "urllib3.exceptions.ProtocolError", "max_hops": 2},
    )

    result = _retrieve(neo4j_session=session, resolver=resolver, openai_client=client)

    assert result.plan.startswith("T3_EXCEPTION_WRAP_CHAIN")
    assert result.plan_repair is None
    assert "requests.exceptions.ConnectionError wraps urllib3.exceptions.ProtocolError." in [
        f.statement for f in result.graph_facts]


def test_a_raises_plan_fetches_grandparents_too():
    """"...what does it inherit from directly, and what is that parent's own
    base class" (3h-13): JSONDecodeError -> InvalidJSONError -> RequestException,
    the second hop from what the first added (D78)."""
    rows = [{"relationship": "RAISES", "neighbor": "requests.exceptions.JSONDecodeError",
             "chunk_id": "c1", "outgoing": True}]
    session = _ChainThenParents(rows, parents={
        "requests.exceptions.JSONDecodeError": ["requests.exceptions.InvalidJSONError"],
        "requests.exceptions.InvalidJSONError": ["requests.exceptions.RequestException"],
    })
    client = FakeOpenAI(plan={
        "template_id": "T8_RELATED_BY", "entity_id": "requests.sessions.Session", "relationship": "RAISES",
    })

    result = _retrieve(neo4j_session=session, openai_client=client)

    statements = [f.statement for f in result.graph_facts]
    assert "requests.exceptions.InvalidJSONError inherits from requests.exceptions.RequestException." in statements


def test_a_package_listing_plan_is_never_capped():
    rows = [
        {"entity": f"requests.exceptions.E{i}", "neighbor": f"urllib3.exceptions.U{i}", "chunk_id": f"c{i}", "source": "ast"}
        for i in range(12)
    ]
    client = FakeOpenAI(plan={
        "template_id": "T11_EDGES_IN_PACKAGE", "package": "requests", "relationship": "WRAPS_EXCEPTION",
    })

    result = _retrieve(neo4j_session=FakeSession(rows), openai_client=client)

    assert len([f for f in result.graph_facts if f.chunk_id]) == 12


def test_an_empty_plan_falls_back_to_the_named_modules_members():
    """"How many warning classes does requests.exceptions define" planned
    DEFINED_IN on a seeded class and got nothing; the module the question
    named is the other reasonable plan (D82). Run first, fall back on empty."""
    resolver = FakeResolver({"requests.exceptions": "requests.exceptions",
                             "ProxyError": "requests.exceptions.RequestsWarning"})
    member_rows = [{"relationship": "DEFINED_IN", "neighbor": "requests.exceptions.RequestsWarning",
                    "chunk_id": "c1", "outgoing": False, "entity_labels": ["Module"], "neighbor_labels": ["Class"]}]

    class _EmptyThenMembers(FakeSession):
        def run(self, query, **params):
            if "labels(n)" in query or "$entity_ids" in query:
                return super().run(query, **params)
            rows = member_rows if params.get("entity_id") == "requests.exceptions" else []

            class R:
                def data(self_inner):
                    return rows

            self.queries.append((query, params))
            return R()

    session = _EmptyThenMembers([], descriptions={
        "requests.exceptions": (["Module"], 40),
        "requests.exceptions.RequestsWarning": (["Class"], 3),
    })
    client = FakeOpenAI(
        mentions=[{"surface": "requests.exceptions", "package": "requests"},
                  {"surface": "ProxyError", "package": "requests"}],
        plan={"template_id": "T8_RELATED_BY", "entity_id": "requests.exceptions.RequestsWarning",
              "relationship": "DEFINED_IN"},
    )

    result = _retrieve(neo4j_session=session, resolver=resolver, openai_client=client)

    assert result.plan == "T8_RELATED_BY({'entity_id': 'requests.exceptions', 'relationship': 'DEFINED_IN'})"
    assert "returned nothing" in result.plan_repair
    assert any("RequestsWarning is a class defined in requests.exceptions" in f.statement for f in result.graph_facts)


def test_members_plus_parents_get_summary_lines_by_base_and_multi_base():
    """"How many warning classes" and "how many have more than one base"
    over sixty lines: two derived lines state the grouping (D82)."""
    from graphrag.retrieval.retrieve import summarize_members
    from graphrag.retrieval.merge import GraphFact
    facts = [
        GraphFact("requests.exceptions.A is a class defined in requests.exceptions.", "c1",
                  relationship="DEFINED_IN", subject="requests.exceptions.A", object="requests.exceptions"),
        GraphFact("requests.exceptions.B is a class defined in requests.exceptions.", "c2",
                  relationship="DEFINED_IN", subject="requests.exceptions.B", object="requests.exceptions"),
        GraphFact("requests.exceptions.A inherits from builtins.Warning.", "p1",
                  relationship="INHERITS_FROM", subject="requests.exceptions.A", object="builtins.Warning"),
        GraphFact("requests.exceptions.B inherits from requests.exceptions.RequestException.", "p2",
                  relationship="INHERITS_FROM", subject="requests.exceptions.B", object="requests.exceptions.RequestException"),
        GraphFact("requests.exceptions.B inherits from builtins.ValueError.", "p3",
                  relationship="INHERITS_FROM", subject="requests.exceptions.B", object="builtins.ValueError"),
    ]
    lines = [f.statement for f in summarize_members(facts)]
    assert lines[0] == "Members by base class: RequestException: B; ValueError: B; Warning: A."
    assert lines[1] == "1 members have more than one base class: B."


def test_an_empty_plan_on_a_class_falls_back_to_its_members():
    """IMPLEMENTS on BaseAdapter returned nothing; "which methods must a
    subclass implement" is the class's members (D82)."""
    resolver = FakeResolver({"Session": "requests.adapters.BaseAdapter"})
    member_rows = [{"relationship": "DEFINED_IN", "neighbor": "requests.adapters.BaseAdapter.send",
                    "chunk_id": "c1", "outgoing": False, "entity_labels": ["Class"], "neighbor_labels": ["Function"]}]

    class _EmptyThenMembers(FakeSession):
        def run(self, query, **params):
            if "labels(n)" in query or "$entity_ids" in query:
                return super().run(query, **params)
            rows = member_rows if ":DEFINED_IN" in query else []

            class R:
                def data(self_inner):
                    return rows

            self.queries.append((query, params))
            return R()

    session = _EmptyThenMembers([], descriptions={"requests.adapters.BaseAdapter": (["Class"], 20)})
    client = FakeOpenAI(plan={"template_id": "T8_RELATED_BY", "entity_id": "requests.adapters.BaseAdapter",
                              "relationship": "IMPLEMENTS"})

    result = _retrieve(neo4j_session=session, resolver=resolver, openai_client=client)

    assert "'DEFINED_IN'" in result.plan and "returned nothing" in result.plan_repair
    assert any("BaseAdapter.send is a method of requests.adapters.BaseAdapter" in f.statement
               for f in result.graph_facts)


def test_a_parameters_neighbourhood_is_followed_by_where_every_same_named_parameter_goes():
    """"Which function applies verify": the anchor resolves to the documented
    `Session.request.verify`, and cert_verify is reached from
    `HTTPAdapter.send.verify`. The flow is followed from every parameter
    named verify, then one hop further (D83)."""
    rows = [{"relationship": "DEFINED_IN", "neighbor": "requests.sessions.Session.request",
             "chunk_id": "c1", "outgoing": True, "entity_labels": ["Parameter"], "neighbor_labels": ["Function"]}]
    flow_rows = [
        {"entity": "requests.adapters.HTTPAdapter.send.verify",
         "neighbor": "requests.adapters.HTTPAdapter.cert_verify.verify", "chunk_id": "f1", "source": "jedi"},
        {"entity": "requests.sessions.Session.request.verify",
         "neighbor": "requests.sessions.Session.merge_environment_settings.verify", "chunk_id": "f2", "source": "jedi"},
    ]

    class _WithFlow(_ChainThenParents):
        def run(self, query, **params):
            if "$name" in query:
                self.expansion_queries.append((":T12", params["name"]))

                class R:
                    def data(self_inner):
                        return flow_rows

                return R()
            return super().run(query, **params)

    session = _WithFlow(rows)
    session._edges[":PASSES_TO"] = {
        "requests.adapters.HTTPAdapter.cert_verify.verify": ["urllib3.util.ssl_.resolve_cert_reqs.candidate"]}
    session._descriptions = {"requests.sessions.Session.request.verify": (["Parameter"], 12)}
    resolver = FakeResolver({"Session": "requests.sessions.Session.request.verify"})
    client = FakeOpenAI(plan={"template_id": "T1_NEIGHBORS", "entity_id": "requests.sessions.Session.request.verify"})

    result = _retrieve(neo4j_session=session, resolver=resolver, openai_client=client)

    statements = [f.statement for f in result.graph_facts]
    assert (":T12", "verify") in session.expansion_queries
    assert "requests.adapters.HTTPAdapter.send.verify is passed to requests.adapters.HTTPAdapter.cert_verify.verify." in statements
    assert "requests.adapters.HTTPAdapter.cert_verify.verify is passed to urllib3.util.ssl_.resolve_cert_reqs.candidate." in statements


def test_seeding_looks_deeper_when_the_top_passages_are_all_prose():
    """A question naming nothing, whose top passages are docs, still gets a
    code anchor from further down the ranking (D84)."""
    from tests.fakes import _Cursor
    resolver = FakeResolver({"HTTPAdapter.send": "requests.adapters.HTTPAdapter.send"})
    code_row = ("c9", "requests", "src/requests/adapters.py", "HTTPAdapter.send", "code", "...", 0.3)

    class _DeeperConn(FakeConn):
        """Docs only at the default depth; a code chunk further down."""

        def cursor(self):
            class C(_Cursor):
                def execute(s, q, p=None):
                    s._rows = [DOC_ROW, code_row] if (p or {}).get("k", 0) >= 20 else [DOC_ROW]

            return C([DOC_ROW])

    session = FakeSession([T1_ROW], descriptions={"requests.adapters.HTTPAdapter.send": (["Function"], 30)})
    client = FakeOpenAI(mentions=[{"surface": "nothing", "package": "unknown"}],
                        plan={"template_id": "T1_NEIGHBORS", "entity_id": "requests.adapters.HTTPAdapter.send"})

    result = _retrieve(conn=_DeeperConn(), neo4j_session=session, resolver=resolver, openai_client=client)

    assert result.seeded_from_passages is True
    assert "requests.adapters.HTTPAdapter.send" in [c.canonical_id for c in result.candidates]


def test_a_wrap_chain_also_fetches_the_anchors_subclasses():
    """"Which exceptions ultimately derive from RequestException" planned a
    wrap chain from RequestException five runs of five (D85). The chain
    says nothing about subclasses; an incoming INHERITS_FROM hop from the
    anchor alone does."""
    chain_rows = [{
        "chain": ["requests.exceptions.RequestException", "requests.exceptions.ConnectionError"],
        "chunk_ids": ["c1"], "starts": ["requests.exceptions.ConnectionError"],
    }]
    session = _ChainThenParents(chain_rows, parents={})
    incoming = {"requests.exceptions.RequestException": ["requests.exceptions.MissingSchema"]}
    original_run = session.run

    def run(query, **params):
        if "WHERE b.id IN $entity_ids" in query:
            session.expansion_queries.append((":<-INHERITS_FROM", params["entity_ids"]))
            rows = [{"entity": sub, "neighbor": e, "chunk_id": "s1", "source": "ast"}
                    for e in params["entity_ids"] for sub in incoming.get(e, [])]

            class R:
                def data(self_inner):
                    return rows

            return R()
        return original_run(query, **params)

    session.run = run
    resolver = FakeResolver({"ProxyError": "requests.exceptions.RequestException"})
    client = FakeOpenAI(mentions=[{"surface": "ProxyError", "package": "requests"}],
                        plan={"template_id": "T3_EXCEPTION_WRAP_CHAIN",
                              "entity_id": "requests.exceptions.RequestException", "max_hops": 3})

    result = _retrieve(neo4j_session=session, resolver=resolver, openai_client=client)

    assert (":<-INHERITS_FROM", ["requests.exceptions.RequestException"]) in session.expansion_queries
    assert "requests.exceptions.MissingSchema inherits from requests.exceptions.RequestException." in [
        f.statement for f in result.graph_facts]
