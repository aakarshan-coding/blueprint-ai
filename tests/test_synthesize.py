from graphrag.answer.synthesize import synthesize_answer


class _FakeResponse:
    def __init__(self, text):
        self.output_text = text


class _FakeResponses:
    def __init__(self, text):
        self._text = text
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text):
        self.responses = _FakeResponses(text)


def test_synthesize_answer_returns_the_models_text():
    client = _FakeClient("Session delegates to PoolManager [e1f3].")

    answer = synthesize_answer("What does Session use?", context="...", client=client)

    assert answer == "Session delegates to PoolManager [e1f3]."


def test_synthesize_answer_sends_the_question_and_context():
    client = _FakeClient("answer")

    synthesize_answer("my question", context="my context", client=client)

    sent = client.responses.last_call["input"]
    assert "my question" in sent
    assert "my context" in sent


def test_synthesize_is_deterministic_so_two_systems_with_identical_context_agree():
    # When the router picks VECTOR, the hybrid path and the baseline build the
    # exact same context. Any difference in their answers would then be pure
    # sampling noise, which would show up in the benchmark as a fake delta.
    client = _FakeClient("answer")
    synthesize_answer("q", context="c", client=client)

    assert client.responses.last_call["temperature"] == 0
