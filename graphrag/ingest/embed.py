"""Turn chunk text into vectors for pgvector.

Embeds `embed_text`, never `text` — see D6/D17 in DECISIONS.md for why the
two differ. The model is injected rather than hardcoded so tests never load
real weights; production code passes a real SentenceTransformer.
"""

from functools import lru_cache

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def load_model():
    """Load the real embedding model once and reuse it.

    Loading is slow (weights from disk/network) and the model is stateless
    across calls, so every caller in one process shares a single instance.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], *, model=None) -> list[list[float]]:
    """Embed a batch of strings, preserving input order."""
    if not texts:
        return []
    if model is None:
        model = load_model()

    vectors = model.encode(texts)
    return [list(v) for v in vectors]
