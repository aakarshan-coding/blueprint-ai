"""Logs every routing decision (design §8). Phase 5's benchmark needs this
data broken out by route, and it can't be reconstructed after the fact —
so every decision is written as it happens, not batched or sampled.
"""

_INSERT_QUERY = """
INSERT INTO routing_log (question, route, confidence, latency_ms, template_id, outcome)
VALUES (%(question)s, %(route)s, %(confidence)s, %(latency_ms)s, %(template_id)s, %(outcome)s)
"""


def log_decision(
    conn,
    *,
    question: str,
    route: str,
    confidence: float,
    latency_ms: int,
    template_id: str | None = None,
    outcome: str | None = None,
) -> None:
    params = {
        "question": question, "route": route, "confidence": confidence,
        "latency_ms": latency_ms, "template_id": template_id, "outcome": outcome,
    }
    with conn.cursor() as cur:
        cur.execute(_INSERT_QUERY, params)
    conn.commit()
