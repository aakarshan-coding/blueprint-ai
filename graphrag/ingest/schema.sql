-- Postgres/pgvector schema for the vector store (design §3, §7).
-- chunk_id is the join key back to Neo4j — nothing here duplicates graph data.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    repo        TEXT NOT NULL,
    path        TEXT NOT NULL,
    section     TEXT,
    kind        TEXT NOT NULL,
    start_line  INT NOT NULL,
    end_line    INT NOT NULL,
    text        TEXT NOT NULL,
    embed_text  TEXT NOT NULL,
    embedding   VECTOR(384) NOT NULL
);

-- HNSW over cosine distance, matching the sentence-transformers convention
-- of comparing normalized embeddings by cosine similarity.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Every routing decision (design §8) — needed for the Phase 5 benchmark's
-- per-route breakdown, and can't be reconstructed after the fact.
CREATE TABLE IF NOT EXISTS routing_log (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    question    TEXT NOT NULL,
    route       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    latency_ms  INT NOT NULL,
    template_id TEXT,
    outcome     TEXT
);
