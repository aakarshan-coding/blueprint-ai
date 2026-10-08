"""The `graphrag` command (D93): ingest any Python repository, then ask it questions.

    graphrag ingest ./some-repo [./another-repo ...] [--with-llm] [--reset]
    graphrag ask "What does Session.send call?"
    graphrag ask "What does Session.send call?" --compare   # also the vector-only answer
    graphrag status                                          # what is ingested right now

Ingest detects each repository's package, source folder, docs and changelog,
writes data/corpus.json so later commands know the corpus, clears the stores
(unless --keep), runs the parser pass and the embedding pass, and, with
--with-llm, the model pass over documentation using the caller's OPENAI_API_KEY.
Ask runs the full pipeline for one question and prints the cited answer.
Status reports which repositories are ingested and what the stores hold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graphrag.corpus import ACTIVE_FILE, Corpus, Repo, detect_repo, get_corpus, set_corpus


def _ingest(args: argparse.Namespace) -> int:
    if args.corpus:
        corpus = Corpus.from_file(args.corpus)
    else:
        repos = []
        for i, path in enumerate(args.repos):
            name = args.package[i] if args.package and i < len(args.package) else None
            repos.append(detect_repo(path, name=name))
        corpus = Corpus(name=args.name or "+".join(r.name for r in repos), repos=tuple(repos))
    if not corpus.exists():
        print("corpus not found on disk:", corpus.to_dict(), file=sys.stderr)
        return 2
    set_corpus(corpus)
    ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_FILE.write_text(json.dumps(corpus.to_dict(), indent=2), encoding="utf-8")
    print(f"corpus: {corpus.name}")
    for r in corpus.repos:
        print(f"  {r.name}: {r.src_root}  docs={r.docs or '-'}  changelog={r.changelog or '-'}")

    from graphrag.ingest.embed import load_model
    from graphrag.ingest.load_graph import build_defined_in_edges
    from graphrag.ingest.run_ingestion import (
        chunk_corpus, collect_llm_inputs, estimate_llm_cost, run_ast_pass, run_full_ingestion, write_vectors,
    )
    from graphrag.settings import neo4j_driver, postgres_conn

    print("Chunking...")
    chunks = chunk_corpus(corpus)
    print(f"  {len(chunks)} chunks")
    print("Parsing...")
    node_universe, import_aliases, edges, node_types = run_ast_pass(corpus)
    print(f"  {len(node_universe)} nodes, {len(build_defined_in_edges(node_universe))} DEFINED_IN edges")
    if args.with_llm:
        est = estimate_llm_cost(collect_llm_inputs(chunks, corpus))
        print(f"LLM pass: ~{est['chunks']} inputs, ~${est['est_cost_usd']} (your OPENAI_API_KEY)")
    if args.dry_run:
        return 0

    driver = neo4j_driver()
    with driver.session() as session:
        if not args.keep:
            print("Clearing the graph...")
            session.run("MATCH (n) DETACH DELETE n")
        openai_client = None
        if args.with_llm:
            from openai import OpenAI

            openai_client = OpenAI()
        log_path = args.llm_log if args.with_llm else None
        stats = run_full_ingestion(
            chunks, node_universe, import_aliases, edges, node_types,
            openai_client=openai_client, neo4j_session=session,
            limit=None if args.with_llm else 0, llm_edge_log_path=log_path, corpus=corpus,
        )
    driver.close()

    print("Embedding and writing the vector store...")
    conn = postgres_conn()
    if not args.keep:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
        conn.commit()
    stats["chunks_embedded"] = write_vectors(conn=conn, chunks=chunks, model=load_model())
    conn.close()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


def _ask(args: argparse.Namespace) -> int:
    corpus = get_corpus()
    set_corpus(corpus)
    from openai import OpenAI

    from graphrag.answer.pipeline import answer_hybrid
    from graphrag.ingest.embed import load_model
    from graphrag.ingest.resolve import Resolver
    from graphrag.ingest.run_ingestion import build_public_and_doc_index, run_ast_pass
    from graphrag.settings import neo4j_driver, postgres_conn

    node_universe, import_aliases, _edges, _types = run_ast_pass(corpus)
    public_ids, documented = build_public_and_doc_index(import_aliases, corpus)
    resolver = Resolver(node_universe=node_universe, import_aliases=import_aliases,
                        public_ids=public_ids, documented_params=documented)
    client = OpenAI()
    model = load_model()
    conn = postgres_conn()
    driver = neo4j_driver()
    with driver.session() as session:
        out = answer_hybrid(
            args.question, conn=conn, neo4j_session=session, openai_client=client,
            resolver=resolver, embedding_model=model,
        )
    baseline = None
    if args.compare:
        # The benchmark's baseline: same passages, same answer model, same
        # prompt, no graph. Side by side, the graph's contribution is visible
        # on one question instead of asserted from a table.
        from graphrag.eval.baseline import answer_vector_only

        baseline = answer_vector_only(args.question, conn=conn, openai_client=client, embedding_model=model)
    driver.close(); conn.close()

    if args.show_plan:
        print(f"[route {out.get('route')}] [plan {out.get('graph_plan')}] [facts {out.get('graph_facts_used')}]")
    if args.show_context and out.get("context"):
        print(out["context"]); print("---")
    print(format_answer("graph + text", out) if baseline else out["answer"])
    if baseline:
        print()
        print(format_answer("text only (no graph)", baseline))
    if not out.get("citations_valid", True):
        print("(warning: an answer citation did not match anything retrieved)", file=sys.stderr)
    return 0


PARSER_SOURCES = ("ast", "jedi", "override")


def gather_store_stats(session, conn) -> dict:
    """What the two stores hold right now. Read-only."""
    one = lambda q: session.run(q).single()["n"]  # noqa: E731
    by_source = {
        r["src"]: r["n"]
        for r in session.run(
            "MATCH ()-[r]->() RETURN coalesce(r.source, 'llm') AS src, count(*) AS n"
        ).data()
    }
    by_type = [
        (r["t"], r["n"])
        for r in session.run(
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC"
        ).data()
    ]
    graph_packages = sorted(
        r["p"] for r in session.run(
            "MATCH (m:Module) WHERE NOT m.id CONTAINS '.' RETURN m.id AS p"
        ).data()
    )
    with conn.cursor() as cur:
        cur.execute("SELECT repo, count(*) FROM chunks GROUP BY repo ORDER BY repo")
        chunks = {repo: n for repo, n in cur.fetchall()}
    return {
        "nodes": one("MATCH (n) RETURN count(n) AS n"),
        "edges": one("MATCH ()-[r]->() RETURN count(r) AS n"),
        "parser_edges": sum(n for s, n in by_source.items() if s in PARSER_SOURCES),
        "model_edges": sum(n for s, n in by_source.items() if s not in PARSER_SOURCES),
        "by_type": by_type,
        "graph_packages": graph_packages,
        "chunks": chunks,
    }


def format_status(corpus: Corpus, *, from_last_ingest: bool, stats: dict) -> str:
    """The `graphrag status` report, as text."""
    origin = (
        "from the last `graphrag ingest`, saved in data/corpus.json" if from_last_ingest
        else "the default corpora/requests-urllib3.yaml; nothing has been ingested on this machine yet"
    )
    lines = [f"Corpus: {corpus.name}  ({origin})"]
    for r in corpus.repos:
        lines.append(
            f"  {r.name:<14} {r.root}   source: {r.src}   docs: {r.docs or '-'}   changelog: {r.changelog or '-'}"
        )
    lines += [
        "",
        f"Graph (Neo4j): {stats['nodes']:,} nodes, {stats['edges']:,} relationships",
        f"  extracted from the code by the parser: {stats['parser_edges']:,}",
        f"  read from the documentation by a model: {stats['model_edges']:,}",
    ]
    if stats["by_type"]:
        top = ", ".join(f"{t} {n:,}" for t, n in stats["by_type"][:6])
        lines.append(f"  most common: {top}")
    total_chunks = sum(stats["chunks"].values())
    per_repo = ", ".join(f"{repo} {n:,}" for repo, n in stats["chunks"].items())
    lines += ["", f"Text index (pgvector): {total_chunks:,} chunks" + (f"  ({per_repo})" if per_repo else "")]

    stored = set(stats["graph_packages"]) | set(stats["chunks"])
    if stats["nodes"] == 0 and total_chunks == 0:
        lines += ["", "The stores are empty. Run `graphrag ingest <repo>`."]
    elif stored != set(corpus.packages):
        lines += [
            "",
            f"Warning: the stores hold {sorted(stored)} but the corpus is {list(corpus.packages)}.",
            "Re-run `graphrag ingest` so answers and the corpus agree.",
        ]
    return "\n".join(lines)


def _status(args: argparse.Namespace) -> int:
    try:
        corpus = get_corpus()
    except RuntimeError:
        print("No corpus yet. Run `graphrag ingest <repo>` to build one.")
        return 1
    from graphrag.settings import neo4j_driver, postgres_conn

    try:
        driver = neo4j_driver()
        conn = postgres_conn()
        with driver.session() as session:
            stats = gather_store_stats(session, conn)
    except Exception as e:  # the stores are a separate process the user starts
        print(f"Corpus: {corpus.name}")
        print(f"Could not read the stores ({type(e).__name__}). Are the databases running? `docker compose up -d`")
        return 1
    driver.close(); conn.close()
    print(format_status(corpus, from_last_ingest=ACTIVE_FILE.exists(), stats=stats))
    return 0


def format_answer(label: str, out: dict) -> str:
    """One labelled answer block for --compare."""
    facts = out.get("graph_facts_used", 0)
    passages = out.get("vector_passages_used", 0)
    header = f"== {label}: {facts} graph facts, {passages} passages =="
    return f"{header}\n{out['answer']}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="build the graph and vector index for one or more Python repositories")
    ing.add_argument("repos", nargs="*", help="repository directories")
    ing.add_argument("--corpus", help="a corpus YAML/JSON file instead of repository paths (see corpora/)")
    ing.add_argument("--package", action="append", help="package name for the matching repo, when detection is ambiguous")
    ing.add_argument("--name", help="a name for this corpus")
    ing.add_argument("--with-llm", action="store_true", help="also run the model pass over documentation (uses OPENAI_API_KEY)")
    ing.add_argument("--llm-log", default="data/llm_edges_log.jsonl", help="where the model pass logs its edges for replay")
    ing.add_argument("--keep", action="store_true", help="do not clear the stores first")
    ing.add_argument("--dry-run", action="store_true", help="chunk and parse only; write nothing")
    ing.set_defaults(func=_ingest)

    ask = sub.add_parser("ask", help="answer one question about the ingested corpus")
    ask.add_argument("question")
    ask.add_argument("--show-plan", action="store_true", help="print the route and the graph plan")
    ask.add_argument("--show-context", action="store_true", help="print the context the answer was written from")
    ask.add_argument("--compare", action="store_true",
                     help="also answer with the vector-only baseline (no graph) and print both")
    ask.set_defaults(func=_ask)

    status = sub.add_parser("status", help="show which repositories are ingested and what the stores hold")
    status.set_defaults(func=_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest" and not args.repos and not args.corpus:
        parser.error("ingest needs repository paths or --corpus")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
