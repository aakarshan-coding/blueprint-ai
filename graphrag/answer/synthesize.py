"""Generates the final answer from assembled, labeled context (design §9).

Requires a citation per claim — enforced by instruction here, and checked
for real afterward by citations.py, which is what actually catches a claim
that skipped this rule rather than trusting the model followed it.
"""

MODEL = "gpt-4o"

_SYSTEM_PROMPT = """\
Answer the question using only the facts and passages in the context below.

The context has two labeled sections: GRAPH FACTS (relationships proven by
the knowledge graph) and RETRIEVED PASSAGES (text found by search). Each
line in both sections starts with a [chunk_id] tag.

Every claim in your answer must end with the [chunk_id] tag of the context
line that supports it. If the context doesn't contain enough to answer the
question, say so plainly instead of guessing — do not state anything without
a [chunk_id] immediately after it.
"""


def synthesize_answer(question: str, *, context: str, client, model: str = MODEL) -> str:
    """Generate a cited answer from a question and its assembled context."""
    response = client.responses.create(
        model=model,
        instructions=_SYSTEM_PROMPT,
        input=f"Question: {question}\n\nContext:\n{context}",
        # Deterministic on purpose: when the router picks VECTOR, the hybrid
        # path and the baseline assemble identical context, so any difference
        # in their answers would be pure sampling noise showing up in the
        # benchmark as a delta that isn't real.
        temperature=0,
    )
    return response.output_text
