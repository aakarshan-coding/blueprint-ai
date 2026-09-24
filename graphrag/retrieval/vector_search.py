"""The vector retrieval path: embed a question, find its nearest chunks.

k defaults to 5, not 1 or 3 — D21's real recall@k measurement showed 100%
recall at k=5 versus 67% at k=1 on this corpus. That number, not a guess,
is why 5 is the default.
"""

from graphrag.ingest.embed import embed_texts

DEFAULT_K = 5

_SEARCH_QUERY = """
SELECT chunk_id, repo, path, section, kind, text, embedding <=> %(qvec)s::vector AS dist
FROM chunks
ORDER BY dist
LIMIT %(k)s
"""

_COLUMNS = ["chunk_id", "repo", "path", "section", "kind", "text", "dist"]


def search_chunks(
    question: str, *, conn, model=None, k: int = DEFAULT_K
) -> list[dict]:
    """Return the k chunks whose embedding is nearest to the question."""
    qvec = embed_texts([question], model=model)[0]

    with conn.cursor() as cur:
        cur.execute(_SEARCH_QUERY, {"qvec": qvec, "k": k})
        rows = cur.fetchall()

    return [dict(zip(_COLUMNS, row)) for row in rows]
