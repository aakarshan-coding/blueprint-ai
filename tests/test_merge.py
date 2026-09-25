from graphrag.ontology import RELATIONSHIP_TYPES
from graphrag.retrieval.merge import REL_PHRASES, GraphFact, assemble_context, verbalize


def test_every_ontology_relationship_has_a_phrase():
    # A missing entry would silently produce an ungrammatical or blank
    # sentence the first time that relationship type shows up in an answer.
    assert set(REL_PHRASES) == RELATIONSHIP_TYPES


def test_neighbors_produces_one_statement_per_row_citing_its_chunk():
    rows = [{"relationship": "DELEGATES_TO", "neighbor": "urllib3.poolmanager.PoolManager", "chunk_id": "e1f3"}]

    facts = verbalize("T1_NEIGHBORS", rows, entity_id="requests.sessions.Session")

    assert len(facts) == 1
    assert facts[0].chunk_id == "e1f3"
    assert "requests.sessions.Session" in facts[0].statement
    assert "delegates to" in facts[0].statement
    assert "urllib3.poolmanager.PoolManager" in facts[0].statement


def test_exception_wrap_chain_collapses_overlapping_paths_to_unique_edges():
    # Neo4j returns every path up to max_hops, so a 2-hop query returns both
    # the 1-hop and 2-hop paths — verbalizing each row directly would repeat
    # the first edge's statement.
    rows = [
        {"chain": ["A", "B"], "chunk_ids": ["c1"]},
        {"chain": ["A", "B", "C"], "chunk_ids": ["c1", "c2"]},
    ]

    facts = verbalize("T3_EXCEPTION_WRAP_CHAIN", rows)

    statements = {f.statement for f in facts}
    assert len(facts) == 2  # A-wraps-B and B-wraps-C, not three
    assert any("A" in s and "B" in s for s in statements)
    assert any("B" in s and "C" in s for s in statements)


def test_exception_wrap_chain_facts_cite_the_right_chunk_per_edge():
    rows = [{"chain": ["A", "B", "C"], "chunk_ids": ["c1", "c2"]}]

    facts = verbalize("T3_EXCEPTION_WRAP_CHAIN", rows)
    by_statement = {f.statement: f.chunk_id for f in facts}

    a_to_b = next(s for s in by_statement if "A" in s and "B" in s and "C" not in s)
    b_to_c = next(s for s in by_statement if "B" in s and "C" in s)
    assert by_statement[a_to_b] == "c1"
    assert by_statement[b_to_c] == "c2"


def test_count_by_rel_has_no_chunk_id_its_an_aggregate():
    rows = [{"count": 2}]

    facts = verbalize("T6_COUNT_BY_REL", rows, relationship="WRAPS_EXCEPTION")

    assert facts[0].chunk_id is None
    assert "2" in facts[0].statement
    assert "WRAPS_EXCEPTION" in facts[0].statement


def test_docs_for_symbol_cites_the_doc_sections_chunk():
    rows = [{"doc_section": "requests.docs.SSLVerification", "chunk_id": "ffe9"}]

    facts = verbalize("T7_DOCS_FOR_SYMBOL", rows, entity_id="requests.sessions.Session")

    assert facts[0].chunk_id == "ffe9"
    assert "documented" in facts[0].statement


from graphrag.retrieval.merge import GraphFact, assemble_context


def test_assemble_context_labels_graph_and_vector_sections_separately():
    facts = [GraphFact("Session delegates to PoolManager.", "e1f3")]
    passages = [{"chunk_id": "abc1", "text": "Some retrieved passage text."}]

    context, _retrieved_ids = assemble_context(graph_facts=facts, vector_passages=passages)

    assert "GRAPH RELATIONSHIPS" in context
    assert "RETRIEVED PASSAGES" in context
    assert "Session delegates to PoolManager." in context
    assert "Some retrieved passage text." in context


def test_assemble_context_dedupes_a_chunk_appearing_in_both_sources():
    # The same chunk_id can justify a graph edge and also come back as a top
    # vector hit — it should only appear once in the assembled context.
    facts = [GraphFact("Session delegates to PoolManager.", "e1f3")]
    passages = [
        {"chunk_id": "e1f3", "text": "The Session object persists parameters..."},
        {"chunk_id": "other", "text": "A different passage."},
    ]

    context, _retrieved_ids = assemble_context(graph_facts=facts, vector_passages=passages)

    assert context.count("e1f3") == 1
    assert "A different passage." in context


def test_assemble_context_includes_every_chunk_id_for_validation():
    facts = [GraphFact("A wraps B.", "c1")]
    passages = [{"chunk_id": "c2", "text": "text"}]

    _context, retrieved_ids = assemble_context(graph_facts=facts, vector_passages=passages)

    assert retrieved_ids == {"c1", "c2"}


def test_assemble_context_omits_a_section_that_has_nothing_in_it():
    # The vector-only baseline has no graph facts at all; printing an empty
    # "GRAPH RELATIONSHIPS" header would show it a heading with nothing under it.
    # The hybrid system hits the same case when one path returns nothing.
    context, _ = assemble_context(graph_facts=[], vector_passages=[
        {"chunk_id": "c1", "text": "some passage"},
    ])

    assert "GRAPH RELATIONSHIPS" not in context
    assert "RETRIEVED PASSAGES" in context


def test_assemble_context_omits_the_passages_section_when_empty():
    context, _ = assemble_context(
        graph_facts=[GraphFact("A wraps B.", "c1")], vector_passages=[]
    )

    assert "GRAPH RELATIONSHIPS" in context
    assert "RETRIEVED PASSAGES" not in context


def test_incoming_edges_are_verbalized_in_the_correct_direction():
    # T1_NEIGHBORS matches edges in BOTH directions. The graph says
    # "HTTPAdapter INHERITS_FROM BaseAdapter"; asking about BaseAdapter must
    # not render that as "BaseAdapter inherits from HTTPAdapter" — a false
    # statement handed to the answer model as fact.
    rows = [{
        "relationship": "INHERITS_FROM",
        "neighbor": "requests.adapters.HTTPAdapter",
        "chunk_id": "c1",
        "outgoing": False,
    }]

    facts = verbalize("T1_NEIGHBORS", rows, entity_id="requests.adapters.BaseAdapter")

    assert facts[0].statement == (
        "requests.adapters.HTTPAdapter inherits from requests.adapters.BaseAdapter."
    )


def test_outgoing_edges_keep_the_queried_entity_as_subject():
    rows = [{
        "relationship": "DELEGATES_TO",
        "neighbor": "urllib3.poolmanager.PoolManager",
        "chunk_id": "c1",
        "outgoing": True,
    }]

    facts = verbalize("T1_NEIGHBORS", rows, entity_id="requests.sessions.Session")

    assert facts[0].statement == (
        "requests.sessions.Session delegates to urllib3.poolmanager.PoolManager."
    )


def test_chain_hops_are_verbalized_in_their_true_direction():
    # Undirected traversal means chain order no longer implies edge
    # direction. The graph says "ConnectionError WRAPS ProtocolError"; a walk
    # starting at ProtocolError must not render that as "ProtocolError wraps
    # ConnectionError" — the same false-statement bug as D41.
    rows = [{
        "chain": ["urllib3.ProtocolError", "requests.ConnectionError"],
        "chunk_ids": ["c1"],
        "starts": ["requests.ConnectionError"],   # the edge's real source
    }]

    facts = verbalize("T3_EXCEPTION_WRAP_CHAIN", rows)

    assert facts[0].statement == (
        "requests.ConnectionError wraps urllib3.ProtocolError."
    )


def test_chain_hop_already_pointing_forward_is_left_alone():
    rows = [{
        "chain": ["A", "B"],
        "chunk_ids": ["c1"],
        "starts": ["A"],
    }]

    facts = verbalize("T3_EXCEPTION_WRAP_CHAIN", rows)

    assert facts[0].statement == "A wraps B."


def test_delegation_chain_verbalizes_each_hop_with_its_own_relationship():
    # T5 walks DELEGATES_TO and CALLS together. A hop that is a CALLS edge
    # must read "calls", not "delegates to" — the template says which each
    # hop was, and the verbalizer has to use it rather than assume.
    rows = [{
        "chain": ["requests.api.get", "requests.api.request", "requests.sessions.Session.request"],
        "chunk_ids": ["c1", "c2"],
        "starts": ["requests.api.get", "requests.api.request"],
        "rels": ["CALLS", "DELEGATES_TO"],
    }]

    facts = verbalize("T5_DELEGATION_CHAIN", rows)
    statements = [f.statement for f in facts]

    assert "requests.api.get calls requests.api.request." in statements
    assert "requests.api.request delegates to requests.sessions.Session.request." in statements


def test_delegation_chain_without_rels_column_still_uses_the_template_verb():
    rows = [{"chain": ["A", "B"], "chunk_ids": ["c1"], "starts": ["A"]}]

    facts = verbalize("T5_DELEGATION_CHAIN", rows)

    assert facts[0].statement == "A delegates to B."


def test_assemble_context_states_each_relationship_once():
    """The LLM pass records the same relationship from several chunks, so
    T1 on `verify` handed the model "verify controls certificate
    verification" four times among ten lines (D60, 3h-01). Deduping by chunk
    id keeps all four; the statement is the fact, so it is stated once, with
    the first chunk that supports it."""
    facts = [
        GraphFact("verify controls concept:certificateverification.", "c1"),
        GraphFact("verify controls concept:certificateverification.", "c2"),
        GraphFact("verify controls concept:certificateverification.", "c3"),
        GraphFact("verify is defined in Session.request.", "c4"),
    ]

    context, retrieved_ids = assemble_context(graph_facts=facts, vector_passages=[])

    assert context.count("verify controls concept:certificateverification.") == 1
    assert "[c1]" in context and "[c2]" not in context
    assert retrieved_ids == {"c1", "c4"}


# --- D63: say what an edge means, so it cannot be misread ---------------------

def test_a_parameters_defined_in_edge_is_phrased_as_a_parameter_of():
    """"verify is defined in Session.request" was read by the answer model as
    "Session.request is the function that applies verify" (D60, five runs of
    five). The sentence itself was the problem: "defined in" reads like
    behaviour. With the subject's label the phrasing can be exact."""
    rows = [{
        "relationship": "DEFINED_IN", "neighbor": "requests.sessions.Session.request",
        "chunk_id": "c1", "outgoing": True,
        "entity_labels": ["Parameter"], "neighbor_labels": ["Function"],
    }]

    facts = verbalize("T1_NEIGHBORS", rows, entity_id="requests.sessions.Session.request.verify")

    assert facts[0].statement == (
        "requests.sessions.Session.request.verify is a parameter of requests.sessions.Session.request."
    )


def test_a_methods_defined_in_edge_is_phrased_as_a_method_of():
    rows = [{
        "relationship": "DEFINED_IN", "neighbor": "requests.sessions.Session",
        "chunk_id": "c1", "outgoing": True,
        "entity_labels": ["Function"], "neighbor_labels": ["Class"],
    }]

    facts = verbalize("T1_NEIGHBORS", rows, entity_id="requests.sessions.Session.send")

    assert facts[0].statement == "requests.sessions.Session.send is a method of requests.sessions.Session."


def test_defined_in_phrasing_uses_the_subjects_label_on_an_incoming_edge():
    # Incoming edge: the neighbour is the subject, so its labels decide.
    rows = [{
        "relationship": "DEFINED_IN", "neighbor": "requests.sessions.Session.request.verify",
        "chunk_id": "c1", "outgoing": False,
        "entity_labels": ["Function"], "neighbor_labels": ["Parameter"],
    }]

    facts = verbalize("T1_NEIGHBORS", rows, entity_id="requests.sessions.Session.request")

    assert facts[0].statement.startswith("requests.sessions.Session.request.verify is a parameter of")


def test_rows_without_labels_keep_the_generic_phrase():
    rows = [{"relationship": "DEFINED_IN", "neighbor": "m", "chunk_id": "c1", "outgoing": True}]

    facts = verbalize("T1_NEIGHBORS", rows, entity_id="x")

    assert facts[0].statement == "x is defined in m."


def test_each_fact_records_its_relationship_type():
    rows = [{"relationship": "CALLS", "neighbor": "b", "chunk_id": "c1", "outgoing": True}]

    facts = verbalize("T1_NEIGHBORS", rows, entity_id="a")

    assert facts[0].relationship == "CALLS"


def test_context_carries_a_legend_for_the_relationship_types_it_uses():
    """A generic "these are relationships, not facts" paragraph changed
    nothing (D62). The legend is specific: one line per type that actually
    appears, from the ontology, placed next to the lines it explains."""
    facts = [
        GraphFact("a is a parameter of b.", "c1", relationship="DEFINED_IN"),
        GraphFact("a controls c.", "c2", relationship="CONTROLS"),
    ]

    context, _ = assemble_context(graph_facts=facts, vector_passages=[])

    assert "Relationship meanings:" in context
    assert "DEFINED_IN:" in context and "CONTROLS:" in context
    assert "WRAPS_EXCEPTION:" not in context
    assert context.index("Relationship meanings:") < context.index("=== GRAPH RELATIONSHIPS ===")


def test_no_graph_lines_means_no_legend():
    context, _ = assemble_context(
        graph_facts=[], vector_passages=[{"chunk_id": "p1", "text": "some passage"}]
    )

    assert "Relationship meanings:" not in context
