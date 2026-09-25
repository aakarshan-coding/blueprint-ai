"""The ingestion runner's write helpers (fixes 10, 11, 14).

run_ingestion.py had no tests at all (architecture review, #8). These cover
the three small things changed here; the orchestration itself is still
exercised only by running it.
"""

from graphrag.ingest.ast_extract import Parameter, ParametersEdge
from graphrag.ingest.chunk import Chunk
from graphrag.ingest.run_ingestion import (
    _write_parameters,
    plausible_exception_endpoint,
    write_vectors,
)


class _FakeSession:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))


# --- fix 11: HAS_PARAMETER is written, not just promised -----------------------

def test_write_parameters_writes_has_parameter_as_well_as_defined_in():
    """HAS_PARAMETER was in the ontology and offered to the planner, and no
    edge of that type existed: ingestion wrote Parameter -DEFINED_IN->
    Function only. A T8 plan on it returned nothing (D59, 3h-07)."""
    session = _FakeSession()
    edges = [ParametersEdge(
        source_id="requests.sessions.Session.request",
        parameters=[Parameter(name="verify", has_default=True, default="None")],
    )]

    _write_parameters(session, edges)

    relationships = [
        (q.split("[r:")[1].split(" ")[0], p["source_id"], p["target_id"])
        for q, p in session.calls if "[r:" in q
    ]
    assert ("DEFINED_IN", "requests.sessions.Session.request.verify",
            "requests.sessions.Session.request") in relationships
    assert ("HAS_PARAMETER", "requests.sessions.Session.request",
            "requests.sessions.Session.request.verify") in relationships


# --- fix 10: an exception edge needs exception-shaped endpoints -------------------

def test_a_module_or_function_cannot_be_the_endpoint_of_a_wrap_edge():
    """The LLM pass produced "ReadTimeout wraps urllib3.util.timeout" (a
    module) and "ReadTimeout wraps Timeout" via a docstring that meant
    inheritance. An endpoint the AST pass typed as anything but a class is
    not an exception, and the edge is dropped rather than written."""
    node_types = {
        "urllib3.util.timeout": "Module",
        "requests.adapters.HTTPAdapter.send": "Function",
        "requests.sessions.Session.request.verify": "Parameter",
        "requests.exceptions.ReadTimeout": "Class",
    }

    assert not plausible_exception_endpoint("urllib3.util.timeout", node_types)
    assert not plausible_exception_endpoint("requests.adapters.HTTPAdapter.send", node_types)
    assert not plausible_exception_endpoint("requests.sessions.Session.request.verify", node_types)
    assert plausible_exception_endpoint("requests.exceptions.ReadTimeout", node_types)


def test_an_endpoint_outside_the_corpus_is_allowed_through():
    # builtins.OSError, socket.timeout, a Concept: not in node_types, and the
    # edge's role types them as Exception on write. Nothing to reject on.
    assert plausible_exception_endpoint("builtins.OSError", {})
    assert plausible_exception_endpoint("concept:networkproblem", {})


# --- fix 14: the vector store is rebuilt by the same script -------------------------

class _FakeModel:
    def encode(self, texts, **kwargs):
        return [[0.5] * 384 for _ in texts]


def test_write_vectors_embeds_every_chunk_and_upserts_it(monkeypatch):
    """upsert_chunks was tested and never called: no script in the repo
    populated pgvector, so a fresh clone could not reproduce the benchmark."""
    seen = {}

    def fake_upsert(conn, chunks, *, embeddings):
        seen["n_chunks"] = len(chunks)
        seen["n_embeddings"] = len(embeddings)
        seen["dim"] = len(embeddings[0])

    monkeypatch.setattr("graphrag.ingest.run_ingestion.upsert_chunks", fake_upsert)
    chunks = [
        Chunk(chunk_id=f"c{i}", repo="requests", path="p.py", kind="code", section=f"s{i}",
              start_line=1, end_line=1, text=f"def f{i}(): pass", embed_text=f"def f{i}(): pass")
        for i in range(3)
    ]

    written = write_vectors(conn=object(), chunks=chunks, model=_FakeModel())

    assert written == 3
    assert seen == {"n_chunks": 3, "n_embeddings": 3, "dim": 384}
