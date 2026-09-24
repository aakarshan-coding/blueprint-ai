"""Re-grade frozen answers N times to isolate grader variance from answer variance.

Two runs of the benchmark with identical code disagreed on ~12% of verdicts
(D51). That disagreement has two possible sources, and the summary table can't
tell them apart:

    answer variance  - the synthesis model wrote something different
    grader variance  - the judge scored the same text differently

This harness holds the answers fixed (reading them from a completed benchmark
run) and grades them repeatedly. Any disagreement that remains is the grader's
alone, because nothing else moved.

    python -m graphrag.eval.regrade --trials 4
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from graphrag.eval.run_benchmark import CATEGORY_ORDER, QUESTIONS_PATH, RESULTS_PATH


def grade_one(question: dict, answer: str, *, client) -> tuple[str, str]:
    """Grade one frozen answer. Returns (verdict, grader_name)."""
    from graphrag.eval.grade import (
        grade_answer,
        grade_mechanically,
        grade_out_of_scope,
    )

    if question["category"] == "out_of_scope":
        return grade_out_of_scope(answer).verdict, "mechanical"
    if question.get("must_contain") or question.get("must_contain_any"):
        return (
            grade_mechanically(
                answer,
                must_contain=question.get("must_contain"),
                must_contain_any=question.get("must_contain_any"),
            ).verdict,
            "mechanical",
        )
    return (
        grade_answer(
            question["question"], question["expects"], answer, client=client
        ).verdict,
        "judge",
    )


def summarise_stability(
    trials: list[dict[tuple[str, str], str]],
) -> tuple[list[dict], dict]:
    """Per (question, system), how many distinct verdicts appeared across trials.

    A key that produced the same verdict every time is stable; anything else is
    the grader contradicting itself on identical input.
    """
    unstable = []
    keys = trials[0].keys()
    for key in keys:
        verdicts = [t[key] for t in trials]
        if len(set(verdicts)) > 1:
            unstable.append(
                {
                    "id": key[0],
                    "system": key[1],
                    "verdicts": verdicts,
                    "counts": dict(Counter(verdicts)),
                }
            )
    stats = {
        "graded_per_trial": len(keys),
        "unstable_keys": len(unstable),
        "flip_rate_pct": round(len(unstable) / len(keys) * 100, 1),
    }
    return unstable, stats


def accuracy_per_trial(
    trials: list[dict[tuple[str, str], str]], categories: dict[str, str]
) -> dict:
    """Per-category accuracy for each trial, so the table's own spread is visible."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        per_cat: dict = defaultdict(lambda: {"hybrid": [0, 0], "baseline": [0, 0]})
        for (qid, system), verdict in trial.items():
            bucket = per_cat[categories[qid]][system]
            bucket[0] += verdict == "correct"
            bucket[1] += 1
        for category, systems in per_cat.items():
            for system, (correct, total) in systems.items():
                out[category][system].append(round(correct / total * 100, 1))
    return {c: dict(s) for c, s in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--out", type=Path, default=Path("regrade_stability.json"))
    args = parser.parse_args()

    questions = {
        q["id"]: q
        for q in yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    }
    frozen = json.loads(args.results.read_text(encoding="utf-8"))["results"]
    categories = {r["id"]: r["category"] for r in frozen}

    from openai import OpenAI

    client = OpenAI()

    trials: list[dict[tuple[str, str], str]] = []
    graders: dict[tuple[str, str], str] = {}
    for trial_no in range(1, args.trials + 1):
        print(f"Trial {trial_no}/{args.trials}...")
        verdicts: dict[tuple[str, str], str] = {}
        for row in frozen:
            question = questions[row["id"]]
            for system in ("hybrid", "baseline"):
                verdict, grader = grade_one(
                    question, row[system]["answer"], client=client
                )
                verdicts[(row["id"], system)] = verdict
                graders[(row["id"], system)] = grader
        trials.append(verdicts)

    unstable, stats = summarise_stability(trials)
    by_grader = Counter(graders[(u["id"], u["system"])] for u in unstable)

    print()
    print(f"Answers held fixed. Graded {stats['graded_per_trial']} "
          f"(question, system) pairs {args.trials} times.")
    print(f"Disagreed with itself on {stats['unstable_keys']} "
          f"({stats['flip_rate_pct']}%).")
    print(f"  of which mechanical: {by_grader['mechanical']}   "
          f"judge: {by_grader['judge']}")
    print()

    for u in unstable:
        print(f"  {u['id']:<22} {u['system']:<9} {u['counts']}")

    print()
    print("Per-category accuracy, one column per trial (grader noise only):")
    per_trial = accuracy_per_trial(trials, categories)
    header = "".join(f"{f't{i}':>8}" for i in range(1, args.trials + 1))
    print(f"{'category':<16}{'system':<10}{header}{'spread':>9}")
    print("-" * (26 + 8 * args.trials + 9))
    for category in CATEGORY_ORDER:
        if category not in per_trial:
            continue
        for system in ("hybrid", "baseline"):
            values = per_trial[category][system]
            cells = "".join(f"{v:>8.1f}" for v in values)
            print(
                f"{category:<16}{system:<10}{cells}"
                f"{max(values) - min(values):>8.1f}pt"
            )

    args.out.write_text(
        json.dumps(
            {"stats": stats, "unstable": unstable, "per_trial": per_trial}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
