from graphrag.retrieval.routing_log import log_decision


class _FakeCursor:
    def __init__(self):
        self.calls = []

    def execute(self, query, params):
        self.calls.append((query, params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def test_log_decision_inserts_the_question_and_route():
    conn = _FakeConn()
    log_decision(
        conn, question="What does verify do?", route="VECTOR",
        confidence=0.9, latency_ms=120,
    )

    query, params = conn.cursor_obj.calls[0]
    assert params["question"] == "What does verify do?"
    assert params["route"] == "VECTOR"
    assert params["confidence"] == 0.9
    assert params["latency_ms"] == 120


def test_log_decision_defaults_template_and_outcome_to_none():
    conn = _FakeConn()
    log_decision(conn, question="q", route="VECTOR", confidence=0.9, latency_ms=1)

    _query, params = conn.cursor_obj.calls[0]
    assert params["template_id"] is None
    assert params["outcome"] is None


def test_log_decision_commits():
    conn = _FakeConn()
    log_decision(conn, question="q", route="GRAPH", confidence=0.9, latency_ms=1)

    assert conn.committed is True
