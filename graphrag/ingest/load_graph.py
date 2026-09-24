"""Write extracted, resolved facts into Neo4j.

Every write uses MERGE, never CREATE, so re-running ingestion converges
instead of duplicating (design §5.2). Cypher can't parameterize a label or
relationship type — only property values — so labels and relationship names
are validated against the ontology first and interpolated only after that
check passes. This is the same discipline as the Phase 3 Cypher template
library: a fixed, validated vocabulary going into the query text, real values
only ever going in as parameters.
"""

from graphrag.ontology import ENTITY_TYPES, RELATIONSHIP_TYPES


_ROLE_IMPLIES_TYPE = {
    "INHERITS_FROM": "Class",      # you can only inherit from a class
    "RAISES": "Exception",         # you can only raise an exception
    "CALLS": "Function",           # see the caveat below
}


def infer_external_type(relationship: str) -> str | None:
    """Infer an entity type for a node outside our corpus, from the role it
    played in an edge — the only type information available for something
    whose source we never parsed (Python's stdlib, third-party deps).

    Returns None when the relationship implies nothing definite; leaving a
    node unlabeled beats asserting a type the edge doesn't actually support.

    Known weakness: a class constructor call (`CookieJar()`) is
    syntactically identical to a function call, so "CALLS" can mis-type a
    class as a Function. Callers record `type_source="inferred"` so this
    stays visible in the graph rather than passing as verified fact.
    """
    return _ROLE_IMPLIES_TYPE.get(relationship)


def merge_node(session, entity_type: str, canonical_id: str, **properties) -> None:
    """Create or update a node, keyed by its canonical id.

    Matches on `id` alone, then adds the label — never `MERGE (n:Type {id})`.
    That form matches on label *and* id, so it silently creates a second node
    when one with the same id already exists unlabeled (as `merge_edge` leaves
    behind when it anchors an edge to a node not yet written). id is the join
    key this whole system rests on; two nodes sharing one is never acceptable.
    """
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"{entity_type!r} is not in the ontology's entity types")

    query = f"MERGE (n {{id: $id}}) SET n:{entity_type}, n += $properties"
    session.run(query, id=canonical_id, properties=properties)


def label_if_unlabeled(session, entity_type: str, canonical_id: str, **properties) -> None:
    """Label a node only if it has no label yet — for inferred external types.

    An external symbol can be reached by several relationships (`CALLS` and
    `RAISES` both hit `http.client.ResponseNotReady`), and their implied types
    disagree. Callers apply the reliable inferences first (INHERITS_FROM,
    RAISES) and the unreliable one (CALLS) last, so first-write-wins lands on
    the better guess rather than the last one processed.
    """
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"{entity_type!r} is not in the ontology's entity types")

    query = (
        "MERGE (n {id: $id}) "
        "SET n += $properties "
        "WITH n WHERE size(labels(n)) = 0 "
        f"SET n:{entity_type}"
    )
    session.run(query, id=canonical_id, properties=properties)


def merge_edge(
    session,
    *,
    source_id: str,
    relationship: str,
    target_id: str,
    chunk_id: str,
    confidence: float | None = None,
) -> None:
    """Create or update an edge, anchored to its two endpoint ids.

    Endpoints are matched by id only (no label) — the edge can be written
    before both endpoint nodes have necessarily been merged with their full
    type, and MERGE on a bare id still finds or creates the right anchor.
    """
    if relationship not in RELATIONSHIP_TYPES:
        raise ValueError(f"{relationship!r} is not in the ontology's relationship types")

    query = (
        "MERGE (a {id: $source_id}) "
        "MERGE (b {id: $target_id}) "
        f"MERGE (a)-[r:{relationship} {{chunk_id: $chunk_id}}]->(b) "
        "SET r.confidence = $confidence"
    )
    session.run(
        query,
        source_id=source_id,
        target_id=target_id,
        chunk_id=chunk_id,
        confidence=confidence,
    )


def build_defined_in_edges(node_universe: set[str]) -> list[tuple[str, str]]:
    """Derive (child, parent) DEFINED_IN pairs from dotted-id containment.

    No extractor emits DEFINED_IN directly — a symbol's parent is already
    encoded in its own dotted id ("requests.adapters.HTTPAdapter.send" is
    defined in "requests.adapters.HTTPAdapter"). This turns that implicit
    containment into real edges, so Cypher can traverse "what's defined in
    this module" without string-splitting ids at query time.
    """
    edges = []
    for canonical_id in node_universe:
        if "." not in canonical_id:
            continue
        parent = canonical_id.rsplit(".", 1)[0]
        if parent in node_universe:
            edges.append((canonical_id, parent))
    return edges
