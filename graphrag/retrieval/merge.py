"""Turns raw run_template() results into readable, cited statements
(design §9). Raw triples make awkward prompt text — this converts each one
to a sentence before it ever reaches the model.
"""

from dataclasses import dataclass

from graphrag.ontology import RELATIONSHIP_MEANINGS

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
    # Which relationship type the statement expresses, so assemble_context can
    # put that type's meaning in the legend above it (D63).
    relationship: str | None = None


def _phrase(relationship: str, subject_labels: list[str], object_labels: list[str]) -> str:
    """The verb for one edge, made exact by what kind of nodes it joins.

    "verify is defined in Session.request" was read by the answer model as
    "Session.request is the function that applies verify" (D60, five runs of
    five): "defined in" reads like behaviour. A Parameter's DEFINED_IN edge is
    "is a parameter of"; a method's is "is a method of". Without labels the
    generic phrase stands.
    """
    if relationship == "DEFINED_IN":
        if "Parameter" in subject_labels:
            return "is a parameter of"
        if "Function" in subject_labels and "Class" in object_labels:
            return "is a method of"
    return REL_PHRASES[relationship]


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
        relationship = row["relationship"]
        # Default to outgoing when the column is absent, so older callers and
        # templates that only ever emit outgoing edges keep working.
        outgoing = row.get("outgoing", True)
        entity_labels = row.get("entity_labels") or []
        neighbor_labels = row.get("neighbor_labels") or []
        if outgoing:
            subject, obj = entity_id, row["neighbor"]
            subject_labels, object_labels = entity_labels, neighbor_labels
        else:
            subject, obj = row["neighbor"], entity_id
            subject_labels, object_labels = neighbor_labels, entity_labels
        verb = _phrase(relationship, subject_labels, object_labels)
        facts.append(
            GraphFact(f"{subject} {verb} {obj}.", row["chunk_id"], relationship=relationship)
        )
    return facts


def _verbalize_chain(rows: list[dict], *, relationship: str) -> list[GraphFact]:
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
            hop_relationship = rels[i] if rels[i] in REL_PHRASES else relationship
            edges[(source, target, chunk_id)] = GraphFact(
                f"{source} {REL_PHRASES[hop_relationship]} {target}.", chunk_id,
                relationship=hop_relationship,
            )
    return list(edges.values())


def _verbalize_count(rows: list[dict], *, relationship: str) -> list[GraphFact]:
    count = rows[0]["count"] if rows else 0
    return [GraphFact(
        f"There are {count} {relationship} relationships.", None, relationship=relationship
    )]


def _verbalize_docs(rows: list[dict], *, entity_id: str) -> list[GraphFact]:
    return [
        GraphFact(
            f"{entity_id} is documented in {row['doc_section']}.", row["chunk_id"],
            relationship="DOCUMENTED_IN",
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
        return _verbalize_chain(rows, relationship="WRAPS_EXCEPTION")
    if template_id == "T5_DELEGATION_CHAIN":
        return _verbalize_chain(rows, relationship="DELEGATES_TO")
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
    # Deduplicated by statement as well as by chunk: the LLM pass records the
    # same relationship from several chunks, so T1 on `verify` handed the
    # model "verify controls certificate verification" four times among ten
    # lines (D60, 3h-01). The statement is the fact; it is stated once, with
    # the first chunk that supports it.
    stated: set[str] = set()
    for fact in graph_facts:
        if fact.statement in stated:
            continue
        if fact.chunk_id is not None and fact.chunk_id in retrieved_ids:
            continue
        stated.add(fact.statement)
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
        # A legend for the relationship types actually present, from the
        # ontology, placed next to the lines it explains (D63). A generic
        # "these are relationships, not facts" paragraph in the shared prompt
        # changed nothing (D62); "DEFINED_IN: location only, not what applies
        # the thing" beside the line is a definition, not a disclaimer. It is
        # built here, with the facts, so the vector-only baseline never sees
        # it and the comparison stays graph-side.
        used = []
        for fact in graph_facts:
            if fact.relationship and fact.relationship not in used:
                used.append(fact.relationship)
        legend = "Relationship meanings:\n" + "\n".join(
            f"- {rel}: {RELATIONSHIP_MEANINGS[rel]}" for rel in used
        )
        # "RELATIONSHIPS", not "FACTS": the heading is the first authority cue
        # the model sees, and D60 showed a true structural link presented as
        # fact being taken as the answer to a question it doesn't answer.
        sections.append(
            (legend + "\n\n" if used else "")
            + "=== GRAPH RELATIONSHIPS ===\n" + "\n".join(fact_lines)
        )
    if passage_lines:
        sections.append("=== RETRIEVED PASSAGES ===\n" + "\n".join(passage_lines))

    return "\n\n".join(sections), retrieved_ids
