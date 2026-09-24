from graphrag.ingest.llm_extract import ExtractionResult, Relationship, extract_relationships


class _FakeResponse:
    def __init__(self, output_parsed):
        self.output_parsed = output_parsed


class _FakeResponses:
    def __init__(self, result: ExtractionResult):
        self._result = result
        self.last_call = None

    def parse(self, **kwargs):
        self.last_call = kwargs
        return _FakeResponse(self._result)


class _FakeClient:
    def __init__(self, result: ExtractionResult):
        self.responses = _FakeResponses(result)


def test_extract_relationships_attaches_the_chunk_id_we_pass_in():
    canned = ExtractionResult(
        relationships=[
            Relationship(
                source_surface="Session",
                source_type="Class",
                relationship="DELEGATES_TO",
                target_surface="PoolManager",
                target_type="Class",
                confidence=0.9,
            )
        ]
    )
    client = _FakeClient(canned)

    edges = extract_relationships(
        "The Session object uses urllib3's PoolManager for connection pooling.",
        chunk_id="abc123",
        client=client,
    )

    assert len(edges) == 1
    assert edges[0].chunk_id == "abc123"
    assert edges[0].relationship == "DELEGATES_TO"
    assert edges[0].source_surface == "Session"


def test_extract_relationships_passes_the_chunk_text_to_the_model():
    canned = ExtractionResult(relationships=[])
    client = _FakeClient(canned)

    extract_relationships("some prose", chunk_id="x", client=client)

    assert "some prose" in client.responses.last_call["input"]


def test_extract_relationships_uses_text_format_for_structured_output():
    canned = ExtractionResult(relationships=[])
    client = _FakeClient(canned)

    extract_relationships("some prose", chunk_id="x", client=client)

    assert client.responses.last_call["text_format"] is ExtractionResult
