"""Aggregate repeated benchmark runs into mean ± spread (D51's missing step).

Two runs of identical code disagree on ~12% of verdicts, which on a 15-question
category is ±13 points and on the pooled 60 is ±5. Every single-run delta
reported since D56 has sat inside that. A number is reportable only next to
its spread, and the spread is only knowable from repeated runs.

    python -m graphrag.eval.summarize_runs benchmark_results_repeat_*.json

Per-category accuracies are recomputed from the per-question verdicts rather
than read from each file's summary, so every run is scored by exactly the
same rule: only "correct" counts.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from graphrag.eval.run_benchmark import CATEGORY_ORDER

SYSTEMS = ("hybrid", "baseline")


def _accuracy(rows: list[dict], system: str) -> float:
    correct = sum(1 for r in rows if r[system]["verdict"] == "correct")
    return round(correct / len(rows) * 100, 1)


def _partial_rate(rows: list[dict], system: str) -> float:
    """Share of questions graded 'partial'. Reported beside accuracy, never
    folded into it: partial is zero in the headline and stays zero. But hybrid
    had 22 three-hop partials to the baseline's 8 (D60) — it gets part of the
    chain far more often — and a table that hides that is hiding the effect
    the graph actually has."""
    partial = sum(1 for r in rows if r[system]["verdict"] == "partial")
    return round(partial / len(rows) * 100, 1)


def _stats(values: list[float]) -> dict:
    return {
        "mean": round(sum(values) / len(values), 1),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def aggregate(runs: list[dict]) -> dict:
    """Per category and pooled: hybrid, baseline and delta as mean/min/max
    across runs, plus the raw per-run values so nothing is hidden."""
    per_run_by_cat: dict[str, list[list[dict]]] = defaultdict(list)
    for run in runs:
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for row in run["results"]:
            by_cat[row["category"]].append(row)
        for category, rows in by_cat.items():
            per_run_by_cat[category].append(rows)
        per_run_by_cat["pooled"].append(list(run["results"]))

    out: dict = {}
    for category in [*CATEGORY_ORDER, "pooled"]:
        if category not in per_run_by_cat:
            continue
        run_rows = per_run_by_cat[category]
        hybrid = [_accuracy(rows, "hybrid") for rows in run_rows]
        baseline = [_accuracy(rows, "baseline") for rows in run_rows]
        delta = [round(h - b, 1) for h, b in zip(hybrid, baseline)]
        out[category] = {
            "n": len(run_rows[0]),
            "hybrid": _stats(hybrid),
            "baseline": _stats(baseline),
            "delta": _stats(delta),
            "hybrid_partial": _stats([_partial_rate(rows, "hybrid") for rows in run_rows]),
            "baseline_partial": _stats([_partial_rate(rows, "baseline") for rows in run_rows]),
        }
    return out


def stability(runs: list[dict]) -> dict:
    """Which questions each system always gets right, always gets wrong, and
    flips on. The flipping set is where the spread comes from."""
    verdicts: dict[str, dict[str, list[str]]] = {s: defaultdict(list) for s in SYSTEMS}
    for run in runs:
        for row in run["results"]:
            for system in SYSTEMS:
                verdicts[system][row["id"]].append(row[system]["verdict"] == "correct")

    out: dict = {}
    for system in SYSTEMS:
        buckets = {"always_correct": [], "always_wrong": [], "flipping": []}
        for qid, outcomes in verdicts[system].items():
            if all(outcomes):
                buckets["always_correct"].append(qid)
            elif not any(outcomes):
                buckets["always_wrong"].append(qid)
            else:
                buckets["flipping"].append(qid)
        out[system] = buckets
    return out


def format_table(agg: dict, *, runs: int) -> str:
    lines = [
        f"{runs} runs, identical code. Accuracy = % correct ('partial' does not count).",
        "Each cell: mean  [min .. max] across runs. The last two columns are the mean % of",
        "questions graded partial -- reported beside accuracy, never folded into it.",
        "",
        f"{'category':<14}{'n':>3}   {'hybrid':<22}{'baseline':<22}{'delta':<22}"
        f"{'h.partial':>10}{'b.partial':>10}",
        "-" * 103,
    ]
    for category, entry in agg.items():
        cells = []
        for key in ("hybrid", "baseline", "delta"):
            s = entry[key]
            sign = "+" if key == "delta" and s["mean"] >= 0 else ""
            cells.append(f"{sign}{s['mean']:.1f}  [{s['min']:.1f} .. {s['max']:.1f}]")
        marker = " <-- pooled" if category == "pooled" else ""
        lines.append(
            f"{category:<14}{entry['n']:>3}   {cells[0]:<22}{cells[1]:<22}{cells[2]:<22}"
            f"{entry['hybrid_partial']['mean']:>10.1f}{entry['baseline_partial']['mean']:>10.1f}{marker}"
        )
    return "\n".join(lines)


def main(paths: list[str]) -> None:
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    agg = aggregate(runs)
    print(format_table(agg, runs=len(runs)))

    stab = stability(runs)
    print()
    for system in SYSTEMS:
        b = stab[system]
        print(f"{system:<9} always correct {len(b['always_correct']):>2} | always wrong "
              f"{len(b['always_wrong']):>2} | flipping {len(b['flipping']):>2}: {', '.join(b['flipping'])}")

    pooled = agg["pooled"]["delta"]
    print()
    if pooled["min"] > 0:
        print(f"Pooled delta is positive in every run (min {pooled['min']:+.1f}).")
    elif pooled["max"] < 0:
        print(f"Pooled delta is negative in every run (max {pooled['max']:+.1f}).")
    else:
        print("Pooled delta changes sign across runs: no direction is supportable.")


if __name__ == "__main__":
    main(sys.argv[1:])
