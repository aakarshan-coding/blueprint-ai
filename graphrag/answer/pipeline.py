"""The full hybrid answer path, as one callable.

Router decides which retrieval paths run; the graph path plans a template and
resolves entities; both paths' results are merged, labeled, and answered; every
citation is validated. Returns the same shape as eval/baseline.py so the two
can be compared question-for-question in the benchmark.
"""

from graphrag.answer.citations import validate_citations
from graphrag.answer.synthesize import synthesize_answer
from graphrag.retrieval.cypher_templates import run_template
from graphrag.retrieval.graph_query import build_template_values, plan_graph_query
from graphrag.retrieval.merge import assemble_context, verbalize
from graphrag.retrieval.router import classify_question, effective_route
from graphrag.retrieval.vector_search import DEFAULT_K, search_chunks

REFUSAL_TEXT = "This question is outside the scope of the requests/urllib3 corpus."


def answer_hybrid(
    question: str,
    *,
    conn,
    neo4j_session,
    openai_client,
    resolver,
    embedding_model=None,
    k: int = DEFAULT_K,
) -> dict:
    """Route, retrieve from whichever paths apply, merge, answer, validate."""
    decision = classify_question(question, client=openai_client)
    route = effective_route(decision)

    if route == "REFUSE":
        return {
            "answer": REFUSAL_TEXT,
            "route": route,
            "confidence": decision.confidence,
            "retrieved_ids": [],
            "citations_valid": True,  # a refusal makes no claims to cite
            "invalid_citations": [],
            "graph_facts_used": 0,
            "vector_passages_used": 0,
            "refused": True,
            "graph_error": None,
            "fell_back_to_vector": False,
        }

    graph_facts = []
    graph_error = None
    if route in ("GRAPH", "BOTH"):
        try:
            plan = plan_graph_query(question, client=openai_client)
            values, known_ids = build_template_values(plan, resolver=resolver)
            rows = run_template(
                neo4j_session, plan.template_id, values, known_entity_ids=known_ids
            )
            graph_facts = verbalize(
                plan.template_id, rows,
                entity_id=values.get("entity_id"),
                relationship=values.get("relationship"),
            )
        except Exception as e:
            # A graph path that can't resolve an entity or plan a template is
            # a real outcome worth recording, not a crash — the benchmark
            # needs to see it rather than lose the whole question.
            graph_error = f"{type(e).__name__}: {e}"

    # A GRAPH-only route that comes back with nothing must not answer from an
    # empty context — the baseline would have had passages for the same
    # question, so answering with nothing loses outright and tells us nothing
    # about whether the graph helps. This hit 15 of 60 benchmark questions.
    fell_back_to_vector = bool(
        route == "GRAPH" and not graph_facts
    )

    vector_passages = []
    if route in ("VECTOR", "BOTH") or fell_back_to_vector:
        vector_passages = search_chunks(
            question, conn=conn, model=embedding_model, k=k
        )

    context, retrieved_ids = assemble_context(
        graph_facts=graph_facts, vector_passages=vector_passages
    )

    if not context.strip():
        return {
            "answer": "No supporting information was retrieved for this question.",
            "route": route,
            "confidence": decision.confidence,
            "retrieved_ids": [],
            "citations_valid": True,
            "invalid_citations": [],
            "graph_facts_used": 0,
            "vector_passages_used": 0,
            "refused": False,
            "graph_error": graph_error,
            "fell_back_to_vector": fell_back_to_vector,
        }

    answer = synthesize_answer(question, context=context, client=openai_client)
    check = validate_citations(answer, retrieved_ids=retrieved_ids)

    return {
        "answer": answer,
        "route": route,
        "confidence": decision.confidence,
        "retrieved_ids": sorted(retrieved_ids),
        "citations_valid": check.valid,
        "invalid_citations": check.invalid_citations,
        "graph_facts_used": len(graph_facts),
        "vector_passages_used": len(vector_passages),
        "refused": False,
        "graph_error": graph_error,
        "fell_back_to_vector": fell_back_to_vector,
    }
