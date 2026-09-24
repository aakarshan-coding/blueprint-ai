"""recall@k measurement: did the vector index return the right chunk for a
question, within the top k results? This is the Phase 2 gate — the router
(Phase 3) isn't allowed to be built on retrieval this number hasn't checked.
"""


def recall_at_k(retrieved: list[str], correct: set[str]) -> bool:
    """True if any correct chunk_id appears anywhere in the retrieved list.

    A question can have more than one correct id when its source section was
    split into multiple chunks (design §5) — either half counts as a hit.
    """
    return any(chunk_id in correct for chunk_id in retrieved)
