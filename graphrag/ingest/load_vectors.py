"""Write chunks and their embeddings into Postgres/pgvector.

ON CONFLICT (chunk_id) DO UPDATE makes reruns idempotent — same principle as
MERGE in load_graph.py, applied to a SQL upsert instead of Cypher.
"""

from graphrag.ingest.chunk import Chunk

_UPSERT_QUERY = """
INSERT INTO chunks
    (chunk_id, repo, path, section, kind, start_line, end_line, text, embed_text, embedding)
VALUES
    (%(chunk_id)s, %(repo)s, %(path)s, %(section)s, %(kind)s,
     %(start_line)s, %(end_line)s, %(text)s, %(embed_text)s, %(embedding)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    repo = EXCLUDED.repo, path = EXCLUDED.path, section = EXCLUDED.section,
    kind = EXCLUDED.kind, start_line = EXCLUDED.start_line, end_line = EXCLUDED.end_line,
    text = EXCLUDED.text, embed_text = EXCLUDED.embed_text, embedding = EXCLUDED.embedding
"""


def upsert_chunks(conn, chunks: list[Chunk], *, embeddings: list[list[float]]) -> None:
    """Write chunks and their embeddings, keyed by chunk_id."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings — "
            "they must be paired one-to-one, in the same order"
        )

    rows = [
        {
            "chunk_id": c.chunk_id, "repo": c.repo, "path": c.path,
            "section": c.section, "kind": c.kind, "start_line": c.start_line,
            "end_line": c.end_line, "text": c.text, "embed_text": c.embed_text,
            "embedding": vec,
        }
        for c, vec in zip(chunks, embeddings)
    ]

    with conn.cursor() as cur:
        cur.executemany(_UPSERT_QUERY, rows)
    conn.commit()
