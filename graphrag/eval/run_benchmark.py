"""Runs both systems over the benchmark question set and reports the result
broken out by hop count (design §10).

Every question goes to both the hybrid system and the vector-only baseline,
then both answers are graded the same way by the same judge, which never sees
which system produced which answer. Latency is measured per question per
system; results are written to JSON so the table can be regenerated without
re-running (and re-paying).

    python -m graphrag.eval.run_benchmark --dry-run   # cost estimate, no calls
    python -m graphrag.eval.run_benchmark             # the real run
"""

import argparse
import json
import time
from pathlib import Path

import yaml

QUESTIONS_PATH = Path("graphrag/eval/benchmark_questions.yaml")
RESULTS_PATH = Path("benchmark_results.json")

CATEGORY_ORDER = ["single_hop", "two_hop", "three_hop", "aggregation", "out_of_scope"]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def summarise(results: list[dict]) -> dict:
    """Accuracy per category per system, counting only 'correct' as correct —
    'partial' deliberately does not count, so the headline number can't be
    inflated by half-answers."""
    summary: dict = {}
    for category in CATEGORY_ORDER:
        rows = [r for r in results if r["category"] == category]
        if not rows:
            continue
        entry = {"n": len(rows)}
        for system in ("hybrid", "baseline"):
            correct = sum(1 for r in rows if r[system]["verdict"] == "correct")
            partial = sum(1 for r in rows if r[system]["verdict"] == "partial")
            latencies = [r[system]["latency_ms"] for r in rows]
            entry[system] = {
                "correct": correct,
                "partial": partial,
                "accuracy": round(correct / len(rows) * 100, 1),
                "median_latency_ms": int(sorted(latencies)[len(latencies) // 2]),
            }
        summary[category] = entry
    return summary


def print_table(summary: dict) -> None:
    print()
    print(f"{'category':<14} {'n':>3}  {'hybrid':>8} {'baseline':>9}  {'delta':>7}")
    print("-" * 50)
    for category in CATEGORY_ORDER:
        if category not in summary:
            continue
        e = summary[category]
        h, b = e["hybrid"]["accuracy"], e["baseline"]["accuracy"]
        print(
            f"{category:<14} {e['n']:>3}  {h:>7.1f}% {b:>8.1f}%  {h - b:>+6.1f}pt"
        )
    print()
    print(f"{'category':<14} {'hybrid p50':>11} {'baseline p50':>13}")
    print("-" * 42)
    for category in CATEGORY_ORDER:
        if category not in summary:
            continue
        e = summary[category]
        print(
            f"{category:<14} {e['hybrid']['median_latency_ms']:>9}ms "
            f"{e['baseline']['median_latency_ms']:>11}ms"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only.")
    parser.add_argument("--limit", type=int, default=None, help="Only run N questions.")
    args = parser.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]

    if args.dry_run:
        # hybrid: router + (plan) + synthesis; baseline: synthesis; judge: 2
        calls = len(questions) * 6
        print(f"{len(questions)} questions")
        print(f"~{calls} model calls (router, graph plan, 2 answers, 2 gradings)")
        print("~$1.50-3.00 estimated, mostly gpt-4o answer synthesis")
        print("\n--dry-run: no calls made.")
        return

    import psycopg
    from neo4j import GraphDatabase
    from openai import OpenAI
    from pgvector.psycopg import register_vector

    from graphrag.answer.pipeline import answer_hybrid
    from graphrag.eval.baseline import answer_vector_only
    from graphrag.eval.grade import (
        grade_answer,
        grade_mechanically,
        grade_out_of_scope,
    )
    from graphrag.ingest.embed import load_model
    from graphrag.ingest.resolve import Resolver
    from graphrag.ingest.run_ingestion import build_public_and_doc_index, run_ast_pass

    print("Building resolver...")
    node_universe, import_aliases, _edges, _types = run_ast_pass()
    public_ids, documented = build_public_and_doc_index(import_aliases)
    resolver = Resolver(
        node_universe=node_universe, import_aliases=import_aliases,
        public_ids=public_ids, documented_params=documented,
    )

    print("Loading embedding model...")
    embedding_model = load_model()

    client = OpenAI()
    conn = psycopg.connect("postgresql://graphrag:graphragpassword@localhost:5432/graphrag")
    register_vector(conn)
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "graphragpassword"))

    results = []
    with driver.session() as session:
        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q['id']}")
            row = {"id": q["id"], "category": q["category"], "question": q["question"]}

            for system, fn in (
                ("hybrid", lambda: answer_hybrid(
                    q["question"], conn=conn, neo4j_session=session,
                    openai_client=client, resolver=resolver,
                    embedding_model=embedding_model)),
                ("baseline", lambda: answer_vector_only(
                    q["question"], conn=conn, openai_client=client,
                    embedding_model=embedding_model)),
            ):
                start = time.time()
                try:
                    out = fn()
                    error = None
                except Exception as e:
                    out = {"answer": "", "citations_valid": False}
                    error = f"{type(e).__name__}: {e}"
                latency_ms = int((time.time() - start) * 1000)

                if error:
                    grade = {"verdict": "incorrect", "reason": error}
                elif q["category"] == "out_of_scope":
                    g = grade_out_of_scope(out["answer"])
                    grade = {"verdict": g.verdict, "reason": g.reason, "grader": "mechanical"}
                elif q.get("must_contain") or q.get("must_contain_any"):
                    # Checkable answers are graded without a model, removing
                    # the judge's run-to-run variance (D45) from 82% of the set.
                    g = grade_mechanically(
                        out["answer"],
                        must_contain=q.get("must_contain"),
                        must_contain_any=q.get("must_contain_any"),
                    )
                    grade = {"verdict": g.verdict, "reason": g.reason, "grader": "mechanical"}
                else:
                    g = grade_answer(
                        q["question"], q["expects"], out["answer"], client=client
                    )
                    grade = {"verdict": g.verdict, "reason": g.reason, "grader": "judge"}

                row[system] = {
                    **grade,
                    "answer": out.get("answer", ""),
                    "latency_ms": latency_ms,
                    "citations_valid": out.get("citations_valid"),
                    "route": out.get("route"),
                    "graph_facts_used": out.get("graph_facts_used", 0),
                    "error": error,
                }

            results.append(row)

    conn.close()
    driver.close()

    summary = summarise(results)
    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
    )
    print_table(summary)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
