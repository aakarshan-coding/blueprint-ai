"""Aggregating repeated benchmark runs into mean ± spread.

Every single-run delta since D51 has been inside the noise floor; a number
is only reportable with its spread beside it. This is the pure part: given
N result files, what is the table and which questions are stable.
"""

from graphrag.eval.summarize_runs import aggregate, format_table, stability


def _run(hybrid: dict[str, str], baseline: dict[str, str], category: dict[str, str]) -> dict:
    """A minimal result file: per-question verdicts for both systems."""
    results = [
        {"id": qid, "category": category[qid],
         "hybrid": {"verdict": hybrid[qid]}, "baseline": {"verdict": baseline[qid]}}
        for qid in category
    ]
    return {"results": results}


CATEGORY = {"a1": "single_hop", "a2": "single_hop", "b1": "two_hop", "b2": "two_hop"}

RUN_1 = _run(
    hybrid={"a1": "correct", "a2": "correct", "b1": "correct", "b2": "incorrect"},
    baseline={"a1": "correct", "a2": "incorrect", "b1": "incorrect", "b2": "incorrect"},
    category=CATEGORY,
)
RUN_2 = _run(
    hybrid={"a1": "correct", "a2": "incorrect", "b1": "correct", "b2": "incorrect"},
    baseline={"a1": "correct", "a2": "incorrect", "b1": "correct", "b2": "incorrect"},
    category=CATEGORY,
)


def test_aggregate_reports_mean_and_range_per_category_and_pooled():
    agg = aggregate([RUN_1, RUN_2])

    single = agg["single_hop"]
    assert single["n"] == 2
    assert single["hybrid"] == {"mean": 75.0, "min": 50.0, "max": 100.0, "values": [100.0, 50.0]}
    assert single["baseline"] == {"mean": 50.0, "min": 50.0, "max": 50.0, "values": [50.0, 50.0]}
    assert single["delta"] == {"mean": 25.0, "min": 0.0, "max": 50.0, "values": [50.0, 0.0]}

    pooled = agg["pooled"]
    assert pooled["n"] == 4
    assert pooled["hybrid"]["values"] == [75.0, 50.0]
    assert pooled["baseline"]["values"] == [25.0, 50.0]
    assert pooled["delta"]["values"] == [50.0, 0.0]


def test_aggregate_counts_only_correct_not_partial():
    run = _run(
        hybrid={"a1": "partial", "a2": "partial", "b1": "correct", "b2": "correct"},
        baseline={"a1": "incorrect", "a2": "incorrect", "b1": "incorrect", "b2": "incorrect"},
        category=CATEGORY,
    )

    agg = aggregate([run])

    assert agg["single_hop"]["hybrid"]["mean"] == 0.0
    assert agg["two_hop"]["hybrid"]["mean"] == 100.0


def test_stability_splits_questions_into_always_right_always_wrong_and_flipping():
    """The delta's spread comes from questions that flip between runs. Naming
    them is what turns 'noise' into something that can be looked at."""
    stab = stability([RUN_1, RUN_2])

    assert stab["hybrid"]["always_correct"] == ["a1", "b1"]
    assert stab["hybrid"]["always_wrong"] == ["b2"]
    assert stab["hybrid"]["flipping"] == ["a2"]
    assert stab["baseline"]["flipping"] == ["b1"]


def test_format_table_shows_mean_with_its_spread():
    agg = aggregate([RUN_1, RUN_2])

    text = format_table(agg, runs=2)

    assert "single_hop" in text
    assert "75.0" in text and "50.0" in text and "100.0" in text
    assert "pooled" in text
