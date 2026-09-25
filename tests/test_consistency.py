import pytest

from graphrag.retrieval.consistency import majority_vote


def _calls(*results):
    """A callable returning each result in turn, plus a record of its calls."""
    seq = list(results)
    log: list[int] = []

    def call():
        log.append(len(log))
        return seq[len(log) - 1]

    return call, log


def test_returns_the_majority_result():
    call, _ = _calls("A", "B", "A")
    assert majority_vote(call, trials=3) == "A"


def test_returns_the_unanimous_result():
    call, _ = _calls("A", "A", "A")
    assert majority_vote(call, trials=3) == "A"


def test_calls_exactly_trials_times():
    call, log = _calls("A", "A", "A")
    majority_vote(call, trials=3)
    assert len(log) == 3


def test_a_three_way_tie_resolves_to_the_first_result():
    """With no majority the vote can't improve on a single call, so it must at
    least not make things worse: falling back to the first result keeps the
    outcome a function of the results rather than of dict iteration order."""
    call, _ = _calls("A", "B", "C")
    assert majority_vote(call, trials=3) == "A"


def test_votes_on_the_key_not_the_whole_result():
    """The router's decision carries a float confidence that varies every call.
    Voting on the whole object would find no majority; the vote has to be over
    the discrete part only."""
    call, _ = _calls(("GRAPH", 0.81), ("VECTOR", 0.62), ("GRAPH", 0.77))

    winner = majority_vote(call, trials=3, key=lambda r: r[0])

    assert winner == ("GRAPH", 0.81)


def test_returns_the_first_result_carrying_the_winning_key():
    call, _ = _calls(("A", 1), ("B", 2), ("A", 3))
    assert majority_vote(call, trials=3, key=lambda r: r[0]) == ("A", 1)


def test_one_trial_is_a_plain_call():
    call, log = _calls("A")
    assert majority_vote(call, trials=1) == "A"
    assert len(log) == 1


def test_rejects_a_non_positive_trial_count():
    call, _ = _calls("A")
    with pytest.raises(ValueError):
        majority_vote(call, trials=0)


def test_voting_is_off_by_default():
    """Majority voting bought 1.7 points for 3x the calls (D55) against an
    instability that resolve-first planning later removed entirely: graph
    fact counts were identical across five runs on 58 of 60 questions (D60).
    The mechanism stays available; the default is one call."""
    from graphrag.retrieval.consistency import VOTES

    assert VOTES == 1
