"""Turns raw run_template() results into readable, cited statements
(design §9). Raw triples make awkward prompt text — this converts each one
to a sentence before it ever reaches the model.
"""

from dataclasses import dataclass

REL_PHRASES = {
    "DEFINED_IN": "is defined in",
    "INHERITS_FROM": "inherits from",
    "CALLS": "calls",
    "IMPORTS": "imports",
    "RAISES": "raises",
    "HAS_PARAMETER": "has parameter",
    "DOCUMENTED_IN": "is documented in",
    "EXPLAINS": "explains",
    "IMPLEMENTS": "implements",
    "CONTROLS": "controls",
    "DELEGATES_TO": "delegates to",
    "WRAPS_EXCEPTION": "wraps",
    "CHANGED_IN": "changed in",
    "CONSTRAINS": "constrains",
}


@dataclass(frozen=True)
class GraphFact:
    statement: str
    chunk_id: str | None


def _verbalize_neighbors(rows: list[dict], *, entity_id: str) -> list[GraphFact]:
    """Render each neighbour edge in the direction it actually runs.

    T1_NEIGHBORS matches edges pointing both ways. An incoming edge has to be
    stated with the *neighbour* as subject — otherwise the graph's
    "HTTPAdapter INHERITS_FROM BaseAdapter" comes back as "BaseAdapter
    inherits from HTTPAdapter", which is false, and gets handed to the answer
    model as though it were fact.
    """
    facts = []
    for row in rows:
        verb = REL_PHRASES[row["relationship"]]
        # Default to outgoing when the column is absent, so older callers and
        # templates that only ever emit outgoing edges keep working.
        if row.get("outgoing", True):
            subject, obj = entity_id, row["neighbor"]
        else:
            subject, obj = row["neighbor"], entity_id
        facts.append(GraphFact(f"{subject} {verb} {obj}.", row["chunk_id"]))
    return facts


def _verbalize_chain(rows: list[dict], *, verb: str) -> list[GraphFact]:
    """Collapse every returned path into its unique edges.

    Neo4j returns one row per path length up to max_hops, so a 2-hop query
    returns both the 1-hop and 2-hop paths — they share their first edge.
    Deduplicating by (source, target, chunk_id) means that shared edge is
    verbalized once, not once per path that happens to contain it.
    """
    edges: dict[tuple[str, str, str], GraphFact] = {}
    for row in rows:
        chain, chunk_ids = row["chain"], row["chunk_ids"]
        # Traversal is undirected, so position in the chain does NOT imply
        # edge direction. `starts` carries each hop's real source node; without
        # it a walk that crosses an edge backwards states the reverse of what
        # the graph says.
        starts = row.get("starts") or list(chain)
        # A chain template that walks more than one relationship type (T5
        # walks DELEGATES_TO and CALLS) reports each hop's type in `rels`;
        # a hop is verbalized with its own verb, falling back to the
        # template's default only when the column is absent.
        rels = row.get("rels") or [None] * len(chunk_ids)
        for i, (a, b, chunk_id) in enumerate(zip(chain, chain[1:], chunk_ids)):
            source, target = (a, b) if starts[i] == a else (b, a)
            hop_verb = REL_PHRASES.get(rels[i], verb) if rels[i] else verb
            edges[(source, target, chunk_id)] = GraphFact(
                f"{source} {hop_verb} {target}.", chunk_id
            )
    return list(edges.values())


def _verbalize_count(rows: list[dict], *, relationship: str) -> list[GraphFact]:
    count = rows[0]["count"] if rows else 0
    return [GraphFact(f"There are {count} {relationship} relationships.", None)]


def _verbalize_docs(rows: list[dict], *, entity_id: str) -> list[GraphFact]:
    return [
        GraphFact(
            f"{entity_id} is documented in {row['doc_section']}.", row["chunk_id"]
        )
        for row in rows
    ]


def verbalize(
    template_id: str,
    rows: list[dict],
    *,
    entity_id: str | None = None,
    relationship: str | None = None,
) -> list[GraphFact]:
    """Convert one run_template() result into readable, cited statements."""
    if template_id in ("T1_NEIGHBORS", "T8_RELATED_BY"):
        # T8 rows carry no `relationship` column — the type is fixed by the
        # query — so supply it from the parameter the caller used.
        rows = [{**row, "relationship": row.get("relationship", relationship)} for row in rows]
        return _verbalize_neighbors(rows, entity_id=entity_id)
    if template_id == "T3_EXCEPTION_WRAP_CHAIN":
        return _verbalize_chain(rows, verb=REL_PHRASES["WRAPS_EXCEPTION"])
    if template_id == "T5_DELEGATION_CHAIN":
        return _verbalize_chain(rows, verb=REL_PHRASES["DELEGATES_TO"])
    if template_id == "T6_COUNT_BY_REL":
        return _verbalize_count(rows, relationship=relationship)
    if template_id == "T7_DOCS_FOR_SYMBOL":
        return _verbalize_docs(rows, entity_id=entity_id)
    raise ValueError(f"no verbalizer for template {template_id!r}")


def assemble_context(
    *, graph_facts: list[GraphFact], vector_passages: list[dict]
) -> tuple[str, set[str]]:
    """Build the labeled prompt context, and the set of chunk_ids actually
    included in it — the latter is what citation validation checks answers
    against, per design §9 ("require a citation per claim... resolves to a
    chunk id that was actually retrieved").

    Deduplicated by chunk_id: the same chunk can justify a graph edge and
    also surface as a top vector hit, and should appear once, not twice.
    """
    retrieved_ids: set[str] = set()

    fact_lines = []
    for fact in graph_facts:
        if fact.chunk_id is not None and fact.chunk_id in retrieved_ids:
            continue
        if fact.chunk_id is not None:
            retrieved_ids.add(fact.chunk_id)
        fact_lines.append(
            f"[{fact.chunk_id}] {fact.statement}" if fact.chunk_id else fact.statement
        )

    passage_lines = []
    for passage in vector_passages:
        chunk_id = passage["chunk_id"]
        if chunk_id in retrieved_ids:
            continue
        retrieved_ids.add(chunk_id)
        passage_lines.append(f"[{chunk_id}] {passage['text']}")

    # An empty section is omitted entirely rather than left as a bare header.
    # The vector-only baseline never has graph facts, and either path can come
    # back empty in the hybrid system — a heading with nothing under it just
    # tells the model a source exists and is silent, which it shouldn't have
    # to interpret.
    sections = []
    if fact_lines:
        sections.append("=== GRAPH FACTS ===\n" + "\n".join(fact_lines))
    if passage_lines:
        sections.append("=== RETRIEVED PASSAGES ===\n" + "\n".join(passage_lines))

    return "\n\n".join(sections), retrieved_ids
