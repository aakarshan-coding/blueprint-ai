"""Where the stores are (D93). Defaults match docker-compose.yml; override with
environment variables for anything else."""

import os

NEO4J_URI = os.environ.get("GRAPHRAG_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("GRAPHRAG_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("GRAPHRAG_NEO4J_PASSWORD", "graphragpassword")
POSTGRES_URL = os.environ.get(
    "GRAPHRAG_POSTGRES_URL", "postgresql://graphrag:graphragpassword@localhost:5432/graphrag"
)


def neo4j_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def postgres_conn():
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(POSTGRES_URL)
    register_vector(conn)
    return conn
