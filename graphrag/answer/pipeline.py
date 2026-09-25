"""The full hybrid answer path, as one callable.

retrieve() does everything up to the answer -- route, plan, run, verbalize,
rank, search, assemble -- and is the same function the stability probe
measures (fix 12). This module adds the two steps that only the real answer
needs: writing it, and checking that every citation points at something
retrieved. Returns the same shape as eval/baseline.py so the two can be
compared question-for-question in the benchmark.
"""

from graphrag.answer.citations import validate_citations
from graphrag.answer.synthesize import synthesize_answer
from graphrag.retrieval.retrieve import RetrievalResult, retrieve
from graphrag.retrieval.vector_search import DEFAULT_K

REFUSAL_TEXT = "This question is outside the scope of the requests/urllib3 corpus."
NO_CONTEXT_TEXT = "No supporting information was retrieved for this question."


def _report(r: RetrievalResult, *, answer: str, citations_valid: bool, invalid: list[str]) -> dict:
    return {
        "answer": answer,
        "route": r.route,
        "confidence": r.confidence,
        "retrieved_ids": sorted(r.retrieved_ids),
        "citations_valid": citations_valid,
        "invalid_citations": invalid,
        "graph_facts_used": len(r.graph_facts),
        "vector_passages_used": len(r.passages),
        "refused": r.refused,
        "graph_error": r.graph_error,
        "graph_plan": r.plan,
        "seeded_from_passages": r.seeded_from_passages,
    }


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
    """Retrieve, answer, validate."""
    r = retrieve(
        question, conn=conn, neo4j_session=neo4j_session, openai_client=openai_client,
        resolver=resolver, embedding_model=embedding_model, k=k,
    )

    if r.refused:
        # A refusal makes no claims to cite.
        return _report(r, answer=REFUSAL_TEXT, citations_valid=True, invalid=[])

    if not r.context.strip():
        return _report(r, answer=NO_CONTEXT_TEXT, citations_valid=True, invalid=[])

    answer = synthesize_answer(question, context=r.context, client=openai_client)
    check = validate_citations(answer, retrieved_ids=r.retrieved_ids)
    return _report(r, answer=answer, citations_valid=check.valid, invalid=check.invalid_citations)
