"""Routes a question to the graph, the vector index, both, or a refusal.

A cheap model call classifies the question (§8 of the design). Low confidence
never trusts a single-path answer — it falls back to running both paths and
merging, since a wrong single-path guess costs an answer, while running both
just costs one extra retrieval. The one exception REFUSE gets no exception:
an uncertain refusal still falls back to BOTH, because refusing to answer on
low confidence is exactly the failure mode this fallback exists to avoid.
"""

from typing import Literal

from pydantic import BaseModel

from graphrag.retrieval.consistency import VOTES, majority_vote

Route = Literal["GRAPH", "VECTOR", "BOTH", "REFUSE"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class RouterDecision(BaseModel):
    route: Route
    confidence: float


MODEL = "gpt-4o-mini"  # cheap/fast — this is the "small model call" role (§8)

_SYSTEM_PROMPT = """\
You route questions about the requests and urllib3 Python libraries to one \
of two retrieval systems, or both.

Route to GRAPH for:
- connection questions ("what does X depend on")
- multi-hop chains ("if X happens, what does Y raise")
- comparisons across entities ("how do X and Y differ")
- aggregations over relationships ("how many exceptions wrap Z")

Route to VECTOR for:
- definitions ("what is X")
- policy or usage lookups ("how do I configure X")
- single-fact questions ("what does parameter X do")

Route to BOTH only when the question genuinely needs both a fact and its \
relationships.

Route to REFUSE when the question cannot be answered from the requests/urllib3 \
documentation and source code themselves — even if it mentions "requests" or \
"urllib3" by name. A question about integrating requests with another \
framework (Django, Flask, Celery, ...), or about a library requests/urllib3 \
don't depend on, is out of scope and should be REFUSE, not VECTOR or GRAPH, \
regardless of which library names appear in it. Never REFUSE merely because \
you're unsure how to route a genuinely in-scope question — use a lower \
confidence score for that instead.

Always include a confidence score between 0 and 1 for your own classification.
"""


def classify_question(
    question: str, *, client, model: str = MODEL, votes: int = VOTES
) -> RouterDecision:
    """Classify a question into a retrieval route, by majority of cheap calls.

    Voted rather than asked once because temperature=0 is not determinism: the
    router picked a different route for the same question on 6.7% of benchmark
    questions (D53), which changed what was retrieved and so changed the
    answer. The vote is over `.route` alone — confidence is a float that
    differs on every call, so including it would leave no two results equal.
    """
    def ask() -> RouterDecision:
        response = client.responses.parse(
            model=model,
            instructions=_SYSTEM_PROMPT,
            input=question,
            text_format=RouterDecision,
            temperature=0,
        )
        return response.output_parsed

    return majority_vote(ask, trials=votes, key=lambda d: d.route)


def effective_route(
    decision: RouterDecision, *, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Route:
    """The route to actually use, after applying the low-confidence fallback."""
    if decision.confidence < threshold:
        return "BOTH"
    return decision.route
