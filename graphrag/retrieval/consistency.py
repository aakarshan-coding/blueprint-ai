"""Self-consistency for the small classification calls.

`temperature=0` does not make OpenAI deterministic — there is no seed and
routing varies — and measurement showed the cost of assuming otherwise (D53):
the router returned a different route on 6.7% of benchmark questions and the
planner a different plan on 11.7%, so one question in ten reached synthesis
with a different prompt each run.

Both calls are classifications with a handful of valid outputs, which is
exactly the shape where asking repeatedly and taking the majority helps: a
model that is right most of the time converges, while one genuine coin-flip
stays a coin-flip. It reduces variance rather than removing it, and that limit
is real — an unstable answer is still unstable, just less often.

The cost is acceptable because these are the cheap calls: three gpt-4o-mini
classifications, not three gpt-4o syntheses. The latency is not free though,
and it lands only on the hybrid system, so the benchmark's latency column
compares a 3-call router against a baseline that makes none.
"""

from collections import Counter
from typing import Callable, TypeVar

T = TypeVar("T")

# One: voting is off by default. Three votes bought 1.7 points for 3x the
# calls (D55) against an instability that turned out to be structural, not
# random -- resolve-first planning (D59) removed it, and graph fact counts
# were then identical across five runs on 58 of 60 questions (D60). The
# mechanism stays for callers that want it; the default no longer pays for
# a problem that is gone.
VOTES = 1


def majority_vote(
    call: Callable[[], T], *, trials: int = VOTES, key: Callable[[T], object] = repr
) -> T:
    """Call `call` `trials` times and return the result the most agreed on.

    `key` selects the part of the result to vote on, and defaults to the whole
    of it. It matters whenever a result mixes a discrete decision with a
    continuous field: the router's RouterDecision carries a float confidence
    that differs on every call, so voting on the whole object would find three
    distinct results and no majority at all. Voting on `.route` finds the
    agreement that is actually there.

    Returns the first result whose key won, so the rest of that result (the
    confidence, in the router's case) comes from a real call rather than being
    synthesised from parts of several.
    """
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")

    results = [call() for _ in range(trials)]
    counts = Counter(key(r) for r in results)

    # most_common breaks ties by first insertion, which is first observation —
    # so an all-different split returns the first result rather than depending
    # on hash order. Relied on deliberately; there is a test for it.
    winner = counts.most_common(1)[0][0]
    return next(r for r in results if key(r) == winner)
