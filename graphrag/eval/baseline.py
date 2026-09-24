"""The vector-only baseline the hybrid system is measured against (design §10).

Deliberately built from the *same* pieces as the real system — same chunks,
same embedding model, same k, same prompt assembly, same answer model, same
citation validation. The single difference is that it never consults the
graph: no router, no Cypher, no graph facts in the context.

That's the whole point. If the baseline differed in prompt or model too, a
win couldn't be attributed to retrieval, and the benchmark would prove
nothing.
"""

from graphrag.answer.citations import validate_citations
from graphrag.answer.synthesize import synthesize_answer
from graphrag.retrieval.merge import assemble_context
from graphrag.retrieval.vector_search import DEFAULT_K, search_chunks


def answer_vector_only(
    question: str, *, conn, openai_client, embedding_model=None, k: int = DEFAULT_K
) -> dict:
    """Answer using vector retrieval alone. Returns the answer plus the
    same metadata the hybrid path reports, so the two are comparable."""
    passages = search_chunks(question, conn=conn, model=embedding_model, k=k)

    context, retrieved_ids = assemble_context(
        graph_facts=[], vector_passages=passages
    )

    answer = synthesize_answer(question, context=context, client=openai_client)
    check = validate_citations(answer, retrieved_ids=retrieved_ids)

    return {
        "answer": answer,
        "retrieved_ids": sorted(retrieved_ids),
        "citations_valid": check.valid,
        "invalid_citations": check.invalid_citations,
        "graph_facts_used": 0,
        "vector_passages_used": len(passages),
    }
