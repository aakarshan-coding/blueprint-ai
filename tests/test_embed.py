from graphrag.ingest.embed import EMBEDDING_DIM, embed_texts


class _FakeModel:
    """Mimics SentenceTransformer.encode's shape without loading real weights."""

    def encode(self, texts, **kwargs):
        return [[float(len(t))] * EMBEDDING_DIM for t in texts]


def test_embed_texts_returns_one_vector_per_input_in_order():
    vectors = embed_texts(["short", "a longer string"], model=_FakeModel())

    assert len(vectors) == 2
    assert vectors[0][0] == float(len("short"))
    assert vectors[1][0] == float(len("a longer string"))


def test_embed_texts_returns_vectors_of_the_declared_dimension():
    vectors = embed_texts(["x"], model=_FakeModel())

    assert len(vectors[0]) == EMBEDDING_DIM


def test_embed_texts_handles_an_empty_list():
    assert embed_texts([], model=_FakeModel()) == []
