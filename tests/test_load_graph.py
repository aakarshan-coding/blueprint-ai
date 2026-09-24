import pytest

from graphrag.ingest.load_graph import build_defined_in_edges, merge_edge, merge_node


class _FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))


def test_merge_node_uses_merge_not_create():
    session = _FakeSession()
    merge_node(session, "Class", "requests.sessions.Session")

    query, params = session.calls[0]
    assert "MERGE" in query
    assert "CREATE" not in query
    assert params["id"] == "requests.sessions.Session"


def test_merge_node_puts_the_validated_label_in_the_query_not_the_params():
    session = _FakeSession()
    merge_node(session, "Class", "requests.sessions.Session")

    query, params = session.calls[0]
    assert ":Class" in query
    assert "label" not in params


def test_merge_node_rejects_a_label_outside_the_ontology():
    session = _FakeSession()
    with pytest.raises(ValueError):
        merge_node(session, "Widget", "some.id")

    assert session.calls == []


def test_merge_edge_uses_merge_and_carries_chunk_id():
    session = _FakeSession()
    merge_edge(
        session,
        source_id="requests.adapters.HTTPAdapter.send",
        relationship="DELEGATES_TO",
        target_id="urllib3.poolmanager.PoolManager",
        chunk_id="abc123",
        confidence=0.9,
    )

    query, params = session.calls[0]
    assert "MERGE" in query
    assert "CREATE" not in query
    assert ":DELEGATES_TO" in query
    assert params["source_id"] == "requests.adapters.HTTPAdapter.send"
    assert params["target_id"] == "urllib3.poolmanager.PoolManager"
    assert params["chunk_id"] == "abc123"
    assert params["confidence"] == 0.9


def test_merge_edge_rejects_a_relationship_outside_the_ontology():
    session = _FakeSession()
    with pytest.raises(ValueError):
        merge_edge(
            session,
            source_id="a",
            relationship="FRIENDS_WITH",
            target_id="b",
            chunk_id="x",
        )

    assert session.calls == []


def test_build_defined_in_edges_links_a_method_to_its_class_and_module():
    universe = {
        "requests.adapters",
        "requests.adapters.HTTPAdapter",
        "requests.adapters.HTTPAdapter.send",
    }

    edges = build_defined_in_edges(universe)

    assert ("requests.adapters.HTTPAdapter.send", "requests.adapters.HTTPAdapter") in edges
    assert ("requests.adapters.HTTPAdapter", "requests.adapters") in edges


def test_build_defined_in_edges_skips_a_parent_that_isnt_a_known_node():
    # A top-level module has no parent module node to link to (the package
    # root isn't tracked as a node here) — it should be skipped, not guessed.
    universe = {"requests.adapters"}

    edges = build_defined_in_edges(universe)

    assert edges == []


def test_merge_node_matches_on_id_alone_so_it_never_duplicates_an_existing_node():
    # MERGE (n:Class {id: x}) will NOT match an existing unlabeled node with
    # that id — it creates a second node sharing the id, breaking id as a
    # unique key. Matching on id alone and then SETting the label is the fix.
    session = _FakeSession()
    merge_node(session, "Class", "http.cookiejar.CookieJar")

    query, params = session.calls[0]
    assert "MERGE (n {id: $id})" in query
    assert "MERGE (n:Class" not in query
    assert "SET n:Class" in query
