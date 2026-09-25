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

from dataclasses import dataclass, field

from graphrag.retrieval.cypher_templates import run_template
from graphrag.retrieval.graph_query import (
    Candidate,
    build_template_values,
    describe_entities,
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

    if route in ("GRAPH", "BOTH"):
        try:
            describe = lambda ids: describe_entities(neo4j_session, ids)  # noqa: E731
            # Resolve first, then plan (D59): the model chooses its entity
            # from ids our code resolved, never from a surface it wrote.
            mentions = extract_mentions(question, client=openai_client)
            candidates = resolve_mentions(mentions, resolver=resolver, describe=describe)
            if not candidates:
                seeds = _seed_from_passages(result.passages, resolver=resolver)
                if seeds:
                    info = describe(seeds)
                    candidates = [
                        Candidate(i, kind=info.get(i, ("unknown", 0))[0],
                                  degree=info.get(i, ("unknown", 0))[1])
                        for i in seeds
                    ]
                    result.seeded_from_passages = True
            result.candidates = candidates

            plan = plan_graph_query(question, candidates=candidates, client=openai_client)
            if plan is not None:
                values, known_ids = build_template_values(plan)
                result.plan = f"{plan.template_id}({values})"
                rows = run_template(
                    neo4j_session, plan.template_id, values, known_entity_ids=known_ids
                )
                result.graph_facts = verbalize(
                    plan.template_id, rows,
                    entity_id=values.get("entity_id"),
                    relationship=values.get("relationship"),
                )
        except Exception as e:
            # A graph path that can't plan or run is a real outcome worth
            # recording, not a crash -- the benchmark needs to see it rather
            # than lose the whole question.
            result.graph_error = f"{type(e).__name__}: {e}"

    # Keep the facts most related to the question (fix 2, D64); eight is the cap.
    result.graph_facts = rank_facts(question, result.graph_facts, model=embedding_model)

    result.context, result.retrieved_ids = assemble_context(
        graph_facts=result.graph_facts, vector_passages=result.passages
    )
    return result
