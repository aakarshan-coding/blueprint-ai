import pytest

from graphrag.corpus import DEFAULT_FILE, Corpus, set_corpus


@pytest.fixture(autouse=True)
def _pin_the_measured_corpus():
    """Tests run against the requests + urllib3 corpus definition, never against
    whatever `graphrag ingest` last wrote to data/corpus.json on this machine."""
    set_corpus(Corpus.from_file(DEFAULT_FILE))
    yield
    set_corpus(None)
