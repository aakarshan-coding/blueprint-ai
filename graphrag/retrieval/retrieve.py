"""The one retrieval sequence: route, plan, run, verbalize, rank, search,
assemble. Everything up to -- and not including -- writing the answer.

This sequence used to be written out three times by hand: the answer
pipeline, the stability probe, and an audit script (architecture review, #1;
fix 12). It drifted twice -- the probe kept the old passages rule after the
pipeline dropped it, so a measurement tool was measuring a different
pipeline than the one that answers. Now there is one, and both call it.

Fix 3 lives here too. When the question names nothing the resolver knows
("which requests exception surfaces when a proxy fails?" names no class),
the graph has no starting point. But the passages have already been fetched,
and every code chunk *is* a node -- its section is the symbol it defines. So
the top passages' symbols become the candidates, and the planner proceeds as
it would have for a named entity. The graph and the text were always joined
on the chunk id; this uses that join in the other direction. Doc chunks do
not seed: a heading is not a node.
"""

import re
from dataclasses import dataclass, field

from graphrag.retrieval.cypher_templates import run_template
from graphrag.retrieval.graph_query import (
    Candidate,
    build_template_values,
    describe_entities,
    repair_plan,
    extract_mentions,
    plan_graph_query,
    resolve_mentions,
)
from graphrag.retrieval.merge import GraphFact, assemble_context, rank_facts, verbalize
from graphrag.retrieval.router import classify_question, effective_route
from graphrag.retrieval.vector_search import DEFAULT_K, search_chunks


@dataclass
class RetrievalResult:
    route: str
    confidence: float
    refused: bool
    plan: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    seeded_from_passages: bool = False
    graph_facts: list[GraphFact] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)
    context: str = ""
    retrieved_ids: set[str] = field(default_factory=set)
    graph_error: str | None = None
    # Why the plan was changed after the model wrote it, or None (D73).
    plan_repair: str | None = None
    # How many parent-class facts were added after the cap (D71); 0 when the
    # plan was not a wrap chain or the kept chain nodes have no parents.
    expanded: int = 0


def _seed_from_passages(passages: list[dict], *, resolver) -> list[str]:
    """Canonical ids of the code symbols the top passages define."""
    ids: list[str] = []
    for passage in passages:
        if passage.get("kind") != "code" or not passage.get("section"):
            continue
        resolution = resolver.resolve(passage["section"])
        found = [resolution.canonical_id] if resolution.canonical_id else list(resolution.candidates)
        for canonical_id in found:
            if canonical_id not in ids:
                ids.append(canonical_id)
    return ids


def _nodes_of(facts: list[GraphFact]) -> list[str]:
    ids: list[str] = []
    for fact in facts:
        for node in (fact.subject, fact.object):
            if node and node not in ids:
                ids.append(node)
    return ids


def _expand(
    facts: list[GraphFact], *, relationship: str, neo4j_session, from_facts=None
) -> tuple[list[GraphFact], list[GraphFact]]:
    """Append one outgoing hop of `relationship` from every node the kept
    facts name (or only the nodes `from_facts` name). Returns the new list
    and the facts it added, so a caller can hop again from just those."""
    ids = _nodes_of(facts if from_facts is None else from_facts)
    if not ids:
        return facts, []
    rows = run_template(
        neo4j_session, "T9_EDGES_FROM",
        {"entity_ids": ids, "relationship": relationship}, known_entity_ids=set(ids),
    )
    stated = {f.statement for f in facts}
    added = [
        f for f in verbalize("T9_EDGES_FROM", rows, relationship=relationship)
        if f.statement not in stated
    ]
    return facts + added, added


# What to fetch after the cap, by the shape of the plan that ran (D71, D73,
# D74). Each entry is a list of hops; a hop is (relationship, follow), where
# follow=True means hop again from what the previous hop added, so parents
# become grandparents ("what is that parent's parent", 3h-24) without
# walking every ancestor of every node.
_EXPANSIONS: dict[tuple[str, str | None], list[tuple[str, bool]]] = {
    # wrap chain: "...and what does that inherit from" -> parents, then theirs
    ("T3_EXCEPTION_WRAP_CHAIN", None): [("INHERITS_FROM", False), ("INHERITS_FROM", True)],
    # a module's members: which are warnings, which have two bases
    ("T8_RELATED_BY", "DEFINED_IN"): [("INHERITS_FROM", False)],
    # what a function raises: what each of those wraps, and inherits from
    # ...and the parents' parents: "what does it inherit from directly, and
    # what is that parent's own base class" (3h-13, D78)
    ("T8_RELATED_BY", "RAISES"): [("WRAPS_EXCEPTION", False), ("INHERITS_FROM", False), ("INHERITS_FROM", True)],
    # a call chain: what the callees raise ("what does Retry raise when
    # attempts run out", 3h-23)
    ("T5_DELEGATION_CHAIN", None): [("RAISES", False)],
    # what a class's methods raise: same as a function's raises (D76)
    ("T10_RAISED_BY_METHODS_OF", None): [("WRAPS_EXCEPTION", False), ("INHERITS_FROM", False), ("INHERITS_FROM", True)],
    # subclasses of X: what each of them wraps ("what does requests convert
    # a socket timeout to" from the anchor Timeout, 3h-03, D76)
    ("T8_RELATED_BY", "INHERITS_FROM"): [("WRAPS_EXCEPTION", False)],
}


def retrieve(
    question: str,
    *,
    conn,
    neo4j_session,
    openai_client,
    resolver,
    embedding_model=None,
    k: int = DEFAULT_K,
) -> RetrievalResult:
    decision = classify_question(question, client=openai_client)
    route = effective_route(decision)
    result = RetrievalResult(route=route, confidence=decision.confidence, refused=route == "REFUSE")
    if result.refused:
        return result

    # Passages first, on every non-refused route. Graph facts are additive,
    # never a substitute (D56) -- and fetching them first is what lets them
    # seed the graph when the question itself names nothing.
    result.passages = search_chunks(question, conn=conn, model=embedding_model, k=k)

    template_id = None
    if route in ("GRAPH", "BOTH"):
        try:
            describe = lambda ids: describe_entities(neo4j_session, ids)  # noqa: E731
            # Resolve first, then plan (D59): the model chooses its entity
            # from ids our code resolved, never from a surface it wrote.
            mentions = extract_mentions(question, client=openai_client)
            candidates = resolve_mentions(mentions, resolver=resolver, describe=describe)
            own_candidates = list(candidates)
            # A candidate list that is only package/module nodes is no
            # anchor: "requests" and "urllib3" appear as mentions in most
            # questions, so this fired on zero of 90 before (D66).
            only_modules = all("Module" in c.kind for c in candidates)
            if not candidates or only_modules:
                have = {c.canonical_id for c in candidates}
                seeds = [s for s in _seed_from_passages(result.passages, resolver=resolver) if s not in have]
                if seeds:
                    info = describe(seeds)
                    candidates = candidates + [
                        Candidate(i, kind=info.get(i, ("unknown", 0))[0],
                                  degree=info.get(i, ("unknown", 0))[1])
                        for i in seeds
                    ]
                    result.seeded_from_passages = True
            result.candidates = candidates

            plan = plan_graph_query(question, candidates=candidates, client=openai_client)
            template_id = plan.template_id if plan is not None else None
            if plan is not None:
                values, known_ids = build_template_values(plan)
                # The repair needs every candidate's kind, including the
                # ones seeded from passages: a seeded HTTPAdapter anchoring
                # a wrap chain was not repaired because its kind was not
                # among the question's own candidates (3h-03, D76).
                template_id, values, result.plan_repair = repair_plan(
                    template_id, values, candidates
                )
                if "entity_id" in values:
                    known_ids = {values["entity_id"]}
                result.plan = f"{template_id}({values})"
                rows = run_template(
                    neo4j_session, template_id, values, known_entity_ids=known_ids
                )
                # A wrap chain from a class that is not an exception walks
                # nothing ("which PreparedRequest method raises it"). The
                # labels cannot tell such a class from an exception (D77),
                # so the chain runs first and the fallback is what the
                # class's methods raise, only when the chain came back empty.
                anchor_kind = {c.canonical_id: c.kind for c in candidates}.get(values.get("entity_id"), "")
                if template_id == "T3_EXCEPTION_WRAP_CHAIN" and not rows and "Class" in anchor_kind:
                    template_id = "T10_RAISED_BY_METHODS_OF"
                    values = {"entity_id": values["entity_id"]}
                    result.plan = f"{template_id}({values})"
                    result.plan_repair = "the wrap chain returned nothing; what this class's methods raise instead"
                    rows = run_template(
                        neo4j_session, template_id, values, known_entity_ids=known_ids
                    )
                result.graph_facts = verbalize(
                    template_id, rows,
                    entity_id=values.get("entity_id"),
                    relationship=values.get("relationship"),
                )
        except Exception as e:
            # A graph path that can't plan or run is a real outcome worth
            # recording, not a crash -- the benchmark needs to see it rather
            # than lose the whole question.
            result.graph_error = f"{type(e).__name__}: {e}"

    # Keep the facts most related to the question (fix 2, D64); eight is the
    # cap -- on the unfiltered neighbourhood and the chains. T8 is already
    # filtered to one relationship and its rows *are* the answer: capping it
    # cut "which exceptions derive from RequestException" from fifteen to
    # eight and dropped the ValueError ones (D66, 3h-04).
    if template_id not in ("T8_RELATED_BY", "T10_RAISED_BY_METHODS_OF"):
        result.graph_facts = rank_facts(question, result.graph_facts, model=embedding_model)

    # A wrap chain answers "what does requests raise when urllib3 raises X",
    # but eleven three-hop questions go one step further: "...and what does
    # that inherit from?" The chain walks WRAPS_EXCEPTION only, so that last
    # edge -- which the graph holds, parser-extracted -- was never fetched,
    # and the model answered "the parent class is not specified" (D70, five
    # of five runs). So: after the cap, fetch the parents of exactly the
    # nodes the kept facts name. After the cap, not before, so the parents
    # cannot crowd out the chain they explain; bounded by the cap, so at
    # most a few lines. Walking 4 hops instead does not help: it returns 72
    # wrap facts against a cap of 8, and still no inheritance edge.
    # The same last hop for a module's members (D73): "which classes in
    # requests.exceptions are warnings" and "how many have more than one
    # base" are answered by the members' INHERITS_FROM edges, which the
    # members query does not fetch. Bounded by the member count.
    relationship = None
    if template_id == "T8_RELATED_BY" and result.plan:
        found = re.search(r"'relationship': '(\w+)'", result.plan)
        relationship = found.group(1) if found else None
    hops = _EXPANSIONS.get((template_id, relationship if template_id == "T8_RELATED_BY" else None), [])
    if hops and result.graph_facts:
        try:
            last_added: list[GraphFact] | None = None
            for rel, follow in hops:
                result.graph_facts, added = _expand(
                    result.graph_facts, relationship=rel, neo4j_session=neo4j_session,
                    from_facts=last_added if follow else None,
                )
                result.expanded += len(added)
                last_added = added
        except Exception as e:
            result.graph_error = f"{type(e).__name__}: {e}"

    result.context, result.retrieved_ids = assemble_context(
        graph_facts=result.graph_facts, vector_passages=result.passages
    )
    return result
