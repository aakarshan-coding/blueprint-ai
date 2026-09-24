from graphrag.retrieval.router import RouterDecision, effective_route


def test_high_confidence_route_is_used_as_is():
    decision = RouterDecision(route="GRAPH", confidence=0.9)
    assert effective_route(decision) == "GRAPH"


def test_low_confidence_route_falls_back_to_both():
    decision = RouterDecision(route="VECTOR", confidence=0.4)
    assert effective_route(decision) == "BOTH"


def test_low_confidence_refuse_still_falls_back_to_both():
    # A REFUSE the model wasn't sure about shouldn't silently refuse —
    # low confidence means "run both and let the merge step sort it out",
    # not "trust an uncertain refusal."
    decision = RouterDecision(route="REFUSE", confidence=0.3)
    assert effective_route(decision) == "BOTH"


def test_confident_both_stays_both():
    decision = RouterDecision(route="BOTH", confidence=0.9)
    assert effective_route(decision) == "BOTH"


def test_confident_refuse_is_respected():
    decision = RouterDecision(route="REFUSE", confidence=0.95)
    assert effective_route(decision) == "REFUSE"


def test_threshold_is_configurable():
    decision = RouterDecision(route="GRAPH", confidence=0.6)
    assert effective_route(decision, threshold=0.5) == "GRAPH"
    assert effective_route(decision, threshold=0.7) == "BOTH"


from graphrag.retrieval.router import classify_question


class _FakeResponse:
    def __init__(self, output_parsed):
        self.output_parsed = output_parsed


class _FakeResponses:
    def __init__(self, result: RouterDecision):
        self._result = result
        self.last_call = None

    def parse(self, **kwargs):
        self.last_call = kwargs
        return _FakeResponse(self._result)


class _FakeClient:
    def __init__(self, result: RouterDecision):
        self.responses = _FakeResponses(result)


def test_classify_question_returns_the_models_decision():
    client = _FakeClient(RouterDecision(route="GRAPH", confidence=0.85))

    decision = classify_question(
        "Which requests exception wraps urllib3's MaxRetryError?", client=client
    )

    assert decision.route == "GRAPH"
    assert decision.confidence == 0.85


def test_classify_question_sends_the_question_to_the_model():
    client = _FakeClient(RouterDecision(route="VECTOR", confidence=0.9))

    classify_question("What does verify=False do?", client=client)

    assert "What does verify=False do?" in client.responses.last_call["input"]


def test_classify_question_uses_structured_output():
    client = _FakeClient(RouterDecision(route="VECTOR", confidence=0.9))

    classify_question("What does verify=False do?", client=client)

    assert client.responses.last_call["text_format"] is RouterDecision


def test_router_is_deterministic():
    client = _FakeClient(RouterDecision(route="GRAPH", confidence=0.9))
    classify_question("q", client=client)

    assert client.responses.last_call["temperature"] == 0


class _SequenceResponses:
    """Returns each canned decision in turn, so a vote can be observed."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.last_call = None

    def parse(self, **kwargs):
        self.last_call = kwargs
        result = self._results[self.calls % len(self._results)]
        self.calls += 1
        return _FakeResponse(result)


class _SequenceClient:
    def __init__(self, results):
        self.responses = _SequenceResponses(results)


def test_classify_question_takes_the_majority_route():
    """temperature=0 is not determinism (D53): the router returned a different
    route on 6.7% of benchmark questions. Voting collapses most of that."""
    client = _SequenceClient([
        RouterDecision(route="GRAPH", confidence=0.8),
        RouterDecision(route="VECTOR", confidence=0.55),
        RouterDecision(route="GRAPH", confidence=0.7),
    ])

    decision = classify_question("q", client=client, votes=3)

    assert decision.route == "GRAPH"
    assert client.responses.calls == 3


def test_classify_question_votes_on_route_ignoring_confidence_jitter():
    """Confidence is a float that differs every call. Voting on the whole
    decision would find three distinct results and no majority at all."""
    client = _SequenceClient([
        RouterDecision(route="BOTH", confidence=0.81),
        RouterDecision(route="BOTH", confidence=0.74),
        RouterDecision(route="BOTH", confidence=0.92),
    ])

    decision = classify_question("q", client=client, votes=3)

    assert decision.route == "BOTH"
    assert decision.confidence == 0.81


def test_classify_question_can_be_called_without_voting():
    client = _SequenceClient([RouterDecision(route="GRAPH", confidence=0.9)])

    classify_question("q", client=client, votes=1)

    assert client.responses.calls == 1
