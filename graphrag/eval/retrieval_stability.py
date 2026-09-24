"""Run retrieval N times per question and check whether it lands in the same place.

Freezing the answers showed the grader is stable (D52), which moves the ~12%
verdict noise upstream into answer generation. But "the synthesis model is
random" is only one candidate. Before accepting it, rule out the cheaper
explanation: that retrieval itself is unstable, so the synthesis model is
being handed a different prompt each time and is behaving perfectly.

Synthesis is deliberately NOT called here. Every layer before it is recorded
separately, so an instability can be attributed to the layer that introduced
it rather than to the pipeline as a whole:

    route         - the router's classification (model call)
    plan          - template id + resolved parameters (model call)
    context_hash  - the assembled prompt text (deterministic given the above)

The vector-only baseline is measured too, as a control: it makes no model call
before retrieving, so it should be perfectly stable. If it isn't, the problem
is in the database layer, not the models.

    python -m graphrag.eval.retrieval_stability --trials 4
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from graphrag.eval.run_benchmark import QUESTIONS_PATH

LAYERS = ("route", "plan", "context_hash")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def probe_hybrid(question: str, *, conn, neo4j_session, client, resolver, model) -> dict:
    """Everything answer_hybrid() does up to (not including) synthesis."""
    from graphrag.retrieval.cypher_templates import run_template
    from graphrag.retrieval.graph_query import build_template_values, plan_graph_query
    from graphrag.retrieval.merge import assemble_context, verbalize
    from graphrag.retrieval.router import classify_question, effective_route
    from graphrag.retrieval.vector_search import search_chunks

    decision = classify_question(question, client=client)
    route = effective_route(decision)
    if route == "REFUSE":
        return {"route": route, "plan": "-", "context_hash": "refused"}

    graph_facts, plan_repr = [], "-"
    if route in ("GRAPH", "BOTH"):
        try:
            plan = plan_graph_query(question, client=client)
            values, known_ids = build_template_values(plan, resolver=resolver)
            plan_repr = f"{plan.template_id}({json.dumps(values, sort_keys=True)})"
            rows = run_template(
                neo4j_session, plan.template_id, values, known_entity_ids=known_ids
            )
            graph_facts = verbalize(
                plan.template_id, rows,
                entity_id=values.get("entity_id"),
                relationship=values.get("relationship"),
            )
        except Exception as e:
            plan_repr = f"ERROR:{type(e).__name__}"

    # Mirrors answer_hybrid: passages on every non-refused route. This line
    # has already drifted from the pipeline once (it carried the old
    # fallback rule after the pipeline dropped it), which is the cost of the
    # sequence living in two places rather than one.
    passages = search_chunks(question, conn=conn, model=model)

    context, _ids = assemble_context(graph_facts=graph_facts, vector_passages=passages)
    return {"route": route, "plan": plan_repr, "context_hash": _hash(context)}


def probe_baseline(question: str, *, conn, model) -> dict:
    """The control: vector retrieval with no model call in front of it."""
    from graphrag.retrieval.merge import assemble_context
    from graphrag.retrieval.vector_search import search_chunks

    passages = search_chunks(question, conn=conn, model=model)
    context, _ids = assemble_context(graph_facts=[], vector_passages=passages)
    return {"route": "VECTOR", "plan": "-", "context_hash": _hash(context)}


def summarise(observations: dict[tuple[str, str], list[dict]]) -> dict:
    """For each system and layer, how many questions varied across trials."""
    report: dict = defaultdict(lambda: defaultdict(list))
    for (qid, system), trials in observations.items():
        for layer in LAYERS:
            values = [t[layer] for t in trials]
            if len(set(values)) > 1:
                report[system][layer].append(
                    {"id": qid, "values": dict(Counter(values))}
                )
    return {s: dict(layers) for s, layers in report.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("retrieval_stability.json"))
    args = parser.parse_args()

    questions = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    import psycopg
    from neo4j import GraphDatabase
    from openai import OpenAI
    from pgvector.psycopg import register_vector

    from graphrag.ingest.embed import load_model
    from graphrag.ingest.resolve import Resolver
    from graphrag.ingest.run_ingestion import build_public_and_doc_index, run_ast_pass

    print("Building resolver...")
    node_universe, import_aliases, _e, _t = run_ast_pass()
    public_ids, documented = build_public_and_doc_index(import_aliases)
    resolver = Resolver(
        node_universe=node_universe, import_aliases=import_aliases,
        public_ids=public_ids, documented_params=documented,
    )
    print("Loading embedding model...")
    model = load_model()

    client = OpenAI()
    conn = psycopg.connect(
        "postgresql://graphrag:graphragpassword@localhost:5432/graphrag"
    )
    register_vector(conn)
    driver = GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "graphragpassword")
    )

    observations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with driver.session() as session:
        for trial_no in range(1, args.trials + 1):
            print(f"Trial {trial_no}/{args.trials}...")
            for q in questions:
                observations[(q["id"], "hybrid")].append(
                    probe_hybrid(
                        q["question"], conn=conn, neo4j_session=session,
                        client=client, resolver=resolver, model=model,
                    )
                )
                observations[(q["id"], "baseline")].append(
                    probe_baseline(q["question"], conn=conn, model=model)
                )

    conn.close()
    driver.close()

    report = summarise(observations)
    n = len(questions)

    print()
    print(f"{n} questions x {args.trials} trials, synthesis not called.")
    print()
    print(f"{'system':<10}{'layer':<16}{'unstable':>10}{'of':>5}{'':>3}{'rate':>7}")
    print("-" * 51)
    for system in ("hybrid", "baseline"):
        for layer in LAYERS:
            bad = len(report.get(system, {}).get(layer, []))
            print(f"{system:<10}{layer:<16}{bad:>10}{n:>5}{'':>3}{bad / n * 100:>6.1f}%")

    for system in ("hybrid", "baseline"):
        for layer in LAYERS:
            entries = report.get(system, {}).get(layer, [])
            if not entries:
                continue
            print(f"\n{system} / {layer} varied on {len(entries)}:")
            for e in entries[:15]:
                print(f"  {e['id']:<22} {e['values']}")

    args.out.write_text(
        json.dumps({"trials": args.trials, "n": n, "report": report}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
