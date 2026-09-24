"""Orchestrates Phase 1 end to end: chunk the corpus, run the AST pass, run
the LLM pass, resolve every surface name, write the result into Neo4j.

Run with --dry-run first. It does everything except call the LLM and write to
Neo4j — chunks the whole corpus, runs AST extraction, and reports how many
prose chunks and docstrings the LLM pass would read plus an estimated cost.
Only rerun without --dry-run once that estimate is acceptable.
"""

import argparse
import ast
import json
import re
from pathlib import Path

from graphrag.ingest.ast_extract import (
    extract_calls,
    extract_imports,
    extract_inherits_from,
    extract_parameters,
    extract_raises,
    module_dotted_name,
)
from graphrag.ingest.chunk import (
    Chunk,
    chunk_changelog,
    chunk_python,
    chunk_rst,
    iter_docstrings_for_llm,
    iter_symbols,
)
from graphrag.ingest.load_graph import (
    build_defined_in_edges,
    infer_external_type,
    merge_edge,
    merge_node,
)
from graphrag.ingest.node_ids import concept_id, parameter_id
from graphrag.ingest.resolve import Resolver

REPOS = {
    "requests": {
        "root": Path("requests_repo"),
        "src": "src",
        "docs": "docs",
        "changelog": "HISTORY.md",
    },
    "urllib3": {
        "root": Path("urllib3_repo"),
        "src": "src",
        "docs": "docs",
        "changelog": "CHANGES.rst",
    },
}

# From a 20-chunk sample: ~3.5 chars/token holds for both code and prose in
# this corpus. A char-count estimate, not a real tokenizer count — good
# enough for a go/no-go decision, not for billing.
CHARS_PER_TOKEN = 3.5
INPUT_PRICE_PER_MTOK = 2.50   # gpt-4o, see llm_extract.MODEL
OUTPUT_PRICE_PER_MTOK = 10.00
EST_OUTPUT_TOKENS_PER_CHUNK = 150  # typical relationships-found response size


def collect_python_files(repo: str, cfg: dict) -> list[Path]:
    return sorted((cfg["root"] / cfg["src"]).rglob("*.py"))


def collect_doc_files(repo: str, cfg: dict) -> list[Path]:
    return sorted((cfg["root"] / cfg["docs"]).rglob("*.rst"))


def chunk_corpus() -> list[Chunk]:
    """Chunk every source, doc, and changelog file across both repos."""
    chunks: list[Chunk] = []

    for repo, cfg in REPOS.items():
        for p in collect_python_files(repo, cfg):
            text = p.read_text(encoding="utf-8", errors="replace")
            chunks.extend(chunk_python(text, repo=repo, path=str(p)))

        for p in collect_doc_files(repo, cfg):
            text = p.read_text(encoding="utf-8", errors="replace")
            chunks.extend(chunk_rst(text, repo=repo, path=str(p)))

        changelog_path = cfg["root"] / cfg["changelog"]
        if changelog_path.exists():
            text = changelog_path.read_text(encoding="utf-8", errors="replace")
            chunks.extend(chunk_changelog(text, repo=repo, path=str(changelog_path)))

    return chunks


def run_ast_pass() -> tuple[
    set[str], dict[str, dict[str, str]], dict[str, list], dict[str, str]
]:
    """Run the deterministic pass over every module. Returns the node
    universe, the per-module import alias tables, every raw edge type keyed
    by its extractor name, and each node's entity type.

    Node typing is deliberately simple: every class-shaped symbol is typed
    "Class", never "Exception" — distinguishing them would need walking the
    (still-unresolved-at-this-point) INHERITS_FROM chain to see whether a
    class transitively derives from a real exception base, which isn't
    knowable until resolution has already run. See D31.
    """
    node_universe: set[str] = set()
    node_types: dict[str, str] = {}
    import_aliases: dict[str, dict[str, str]] = {}
    edges = {"inherits_from": [], "raises": [], "parameters": [], "calls": []}

    for repo, cfg in REPOS.items():
        src_root = cfg["root"] / cfg["src"]
        for p in collect_python_files(repo, cfg):
            text = p.read_text(encoding="utf-8", errors="replace")
            dotted = module_dotted_name(
                str(p), src_root=str(src_root), package_root=repo
            )
            node_universe.add(dotted)
            node_types[dotted] = "Module"

            for name, node, _start, _end in iter_symbols(ast.parse(text)):
                symbol_id = f"{dotted}.{name}"
                node_universe.add(symbol_id)
                node_types[symbol_id] = (
                    "Class" if isinstance(node, ast.ClassDef) else "Function"
                )

            import_aliases[dotted] = {
                e.local_name: e.imported
                for e in extract_imports(
                    text, dotted_module=dotted, is_package=(p.name == "__init__.py")
                )
            }

            edges["inherits_from"] += extract_inherits_from(
                text, repo=repo, path=str(p), dotted_module=dotted
            )
            edges["raises"] += extract_raises(
                text, repo=repo, path=str(p), dotted_module=dotted
            )
            param_edges = extract_parameters(text, dotted_module=dotted)
            edges["parameters"] += param_edges
            edges["calls"] += extract_calls(text, dotted_module=dotted)

            # A Parameter's identity is fully determined once we know its
            # function — registering it here means an LLM-extracted fact
            # like "verify controls certificate verification" can actually
            # resolve, instead of the Resolver never having heard of "verify"
            # at all. See D33 — this was missing in the first full run.
            for param_edge in param_edges:
                for param in param_edge.parameters:
                    pid = parameter_id(param_edge.source_id, param.name)
                    node_universe.add(pid)
                    node_types[pid] = "Parameter"

    return node_universe, import_aliases, edges, node_types


_PARAM_DOC_PATTERN = re.compile(r":param\s+(\w+)\s*:")


def build_public_and_doc_index(
    import_aliases: dict[str, dict[str, str]],
) -> tuple[set[str], dict[str, set[str]]]:
    """Return (publicly exported ids, {function_id: documented param names}).

    Both feed `Resolver`'s public-API filter, which is what disambiguates a
    parameter name reused across a call chain. "Public" means re-exported from
    a package's own __init__ — which only became trustworthy once D35 fixed
    relative-import resolution inside __init__ files.
    """
    public_ids: set[str] = set()
    for repo in REPOS:
        public_ids |= set(import_aliases.get(repo, {}).values())

    documented: dict[str, set[str]] = {}
    for repo, cfg in REPOS.items():
        src_root = cfg["root"] / cfg["src"]
        for p in collect_python_files(repo, cfg):
            text = p.read_text(encoding="utf-8", errors="replace")
            dotted = module_dotted_name(
                str(p), src_root=str(src_root), package_root=repo
            )
            for name, node, _start, _end in iter_symbols(ast.parse(text)):
                docstring = ast.get_docstring(node) or ""
                params = set(_PARAM_DOC_PATTERN.findall(docstring))
                if params:
                    documented[f"{dotted}.{name}"] = params

    return public_ids, documented


def module_of(canonical_id: str, node_types: dict[str, str]) -> str | None:
    """Walk up a dotted id's segments to find the module it lives in — the
    right module_context for resolving a surface name written inside it."""
    parts = canonical_id.split(".")
    while parts:
        candidate = ".".join(parts)
        if node_types.get(candidate) == "Module":
            return candidate
        parts.pop()
    return None


def collect_llm_inputs(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """Every (chunk_id, text) pair the LLM pass would read: doc/changelog
    chunks plus every symbol's docstring, pulled separately per D6."""
    inputs = [(c.chunk_id, c.text) for c in chunks if c.kind in ("doc", "changelog")]

    for repo, cfg in REPOS.items():
        for p in collect_python_files(repo, cfg):
            text = p.read_text(encoding="utf-8", errors="replace")
            for doc in iter_docstrings_for_llm(text, repo=repo, path=str(p)):
                inputs.append((doc.chunk_id, doc.docstring))

    return inputs


def estimate_llm_cost(llm_inputs: list[tuple[str, str]]) -> dict:
    total_input_chars = sum(len(text) for _cid, text in llm_inputs)
    input_tokens = total_input_chars / CHARS_PER_TOKEN
    output_tokens = len(llm_inputs) * EST_OUTPUT_TOKENS_PER_CHUNK

    cost = (input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK) + (
        output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
    )
    return {
        "chunks": len(llm_inputs),
        "est_input_tokens": int(input_tokens),
        "est_output_tokens": int(output_tokens),
        "est_cost_usd": round(cost, 2),
    }


def _write_ast_nodes(session, node_universe: set[str], node_types: dict[str, str]) -> None:
    for canonical_id in node_universe:
        merge_node(session, node_types.get(canonical_id, "Function"), canonical_id)


def _write_parameters(session, parameter_edges: list) -> int:
    """Deterministic — no resolution needed, a parameter's identity is
    already fully determined by its function and its own name."""
    count = 0
    for edge in parameter_edges:
        for param in edge.parameters:
            pid = parameter_id(edge.source_id, param.name)
            merge_node(session, "Parameter", pid)
            merge_edge(
                session, source_id=pid, relationship="DEFINED_IN",
                target_id=edge.source_id, chunk_id="ast", confidence=1.0,
            )
            count += 1
    return count


def run_full_ingestion(
    chunks: list[Chunk],
    node_universe: set[str],
    import_aliases: dict[str, dict[str, str]],
    edges: dict[str, list],
    node_types: dict[str, str],
    *,
    openai_client,
    neo4j_session,
    limit: int | None = None,
    llm_edge_log_path: str | None = None,
) -> dict:
    """Write the AST pass and (bounded) LLM pass into a real Neo4j graph.

    Scope cut, stated plainly (see D31): LLM-extracted edges whose source or
    target the model typed as DocSection or Release are counted as
    unresolved and skipped, not written. Resolver only knows about
    Module/Class/Function/Exception/Parameter ids; giving it DocSection and
    Release lookups too is real, separate work not built in this pass.
    Concept endpoints are the exception — a Concept's id is just its own
    normalized surface text, so it never needs resolving against anything.
    """
    from graphrag.ingest.llm_extract import extract_relationships

    stats = {
        "ast_edges_written": 0, "ast_edges_unresolved": 0,
        "llm_calls_ok": 0, "llm_calls_failed": 0,
        "llm_edges_written": 0, "llm_edges_unresolved_docsection_release": 0,
        "llm_edges_unresolved_other": 0,
    }

    resolver = Resolver(node_universe=node_universe, import_aliases=import_aliases)

    print("Writing AST-derived nodes...")
    _write_ast_nodes(neo4j_session, node_universe, node_types)

    print("Writing parameters...")
    param_count = _write_parameters(neo4j_session, edges["parameters"])
    print(f"  {param_count} parameters")

    print("Writing DEFINED_IN edges...")
    for child, parent in build_defined_in_edges(node_universe):
        merge_edge(
            neo4j_session, source_id=child, relationship="DEFINED_IN",
            target_id=parent, chunk_id="ast", confidence=1.0,
        )

    def type_if_external(canonical_id: str, relationship: str) -> None:
        """Label a node that resolved to something outside our own corpus.

        `origin` records where it came from; `type_source` records that its
        type was inferred from an edge role rather than proven by parsing
        the code — both queryable, so a guess never looks like a fact.
        """
        if canonical_id in node_universe:
            return
        inferred = infer_external_type(relationship)
        if inferred is None:
            return
        merge_node(
            neo4j_session, inferred, canonical_id,
            origin="external", type_source="inferred",
        )

    print("Writing AST relationship edges (INHERITS_FROM, RAISES, CALLS)...")
    for edge in edges["inherits_from"]:
        module_context = module_of(edge.source_id, node_types)
        for surface in edge.target_surfaces:
            r = resolver.resolve(surface, module_context=module_context)
            if r.canonical_id is None:
                stats["ast_edges_unresolved"] += 1
                continue
            type_if_external(r.canonical_id, "INHERITS_FROM")
            merge_edge(
                neo4j_session, source_id=edge.source_id, relationship="INHERITS_FROM",
                target_id=r.canonical_id, chunk_id=edge.chunk_id, confidence=1.0,
            )
            stats["ast_edges_written"] += 1

    for edge in edges["raises"]:
        module_context = module_of(edge.source_id, node_types)
        for surface in edge.exceptions_raised:
            r = resolver.resolve(surface, module_context=module_context)
            if r.canonical_id is None:
                stats["ast_edges_unresolved"] += 1
                continue
            type_if_external(r.canonical_id, "RAISES")
            merge_edge(
                neo4j_session, source_id=edge.source_id, relationship="RAISES",
                target_id=r.canonical_id, chunk_id=edge.chunk_id, confidence=1.0,
            )
            stats["ast_edges_written"] += 1

    for edge in edges["calls"]:
        module_context = module_of(edge.source_id, node_types)
        for surface in edge.calls:
            r = resolver.resolve(surface, module_context=module_context)
            if r.canonical_id is None:
                stats["ast_edges_unresolved"] += 1
                continue
            type_if_external(r.canonical_id, "CALLS")
            merge_edge(
                neo4j_session, source_id=edge.source_id, relationship="CALLS",
                target_id=r.canonical_id, chunk_id="ast", confidence=1.0,
            )
            stats["ast_edges_written"] += 1

    print("Running the LLM pass...")
    llm_inputs = collect_llm_inputs(chunks)
    if limit is not None:
        llm_inputs = llm_inputs[:limit]

    all_llm_edges = []
    for i, (chunk_id, text) in enumerate(llm_inputs):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(llm_inputs)}")
        try:
            all_llm_edges += extract_relationships(text, chunk_id=chunk_id, client=openai_client)
            stats["llm_calls_ok"] += 1
        except Exception as e:
            stats["llm_calls_failed"] += 1
            print(f"  LLM call failed on chunk {chunk_id}: {e}")

    print(f"Writing {len(all_llm_edges)} LLM-derived edges...")
    log_file = open(llm_edge_log_path, "w", encoding="utf-8") if llm_edge_log_path else None

    for edge in all_llm_edges:
        record = edge.model_dump()
        record["resolved_source_id"] = None
        record["resolved_target_id"] = None
        record["skip_reason"] = None

        if edge.source_type == "Concept":
            src_id = concept_id(edge.source_surface)
            merge_node(neo4j_session, "Concept", src_id)
        elif edge.source_type in ("DocSection", "Release"):
            stats["llm_edges_unresolved_docsection_release"] += 1
            record["skip_reason"] = "source_docsection_or_release"
            if log_file:
                log_file.write(json.dumps(record) + "\n")
            continue
        else:
            r = resolver.resolve(edge.source_surface)
            if r.canonical_id is None:
                stats["llm_edges_unresolved_other"] += 1
                record["skip_reason"] = "source_unresolved"
                if log_file:
                    log_file.write(json.dumps(record) + "\n")
                continue
            src_id = r.canonical_id
        record["resolved_source_id"] = src_id

        if edge.target_type == "Concept":
            tgt_id = concept_id(edge.target_surface)
            merge_node(neo4j_session, "Concept", tgt_id)
        elif edge.target_type in ("DocSection", "Release"):
            stats["llm_edges_unresolved_docsection_release"] += 1
            record["skip_reason"] = "target_docsection_or_release"
            if log_file:
                log_file.write(json.dumps(record) + "\n")
            continue
        else:
            r = resolver.resolve(edge.target_surface)
            if r.canonical_id is None:
                stats["llm_edges_unresolved_other"] += 1
                record["skip_reason"] = "target_unresolved"
                if log_file:
                    log_file.write(json.dumps(record) + "\n")
                continue
            tgt_id = r.canonical_id
        record["resolved_target_id"] = tgt_id

        if log_file:
            log_file.write(json.dumps(record) + "\n")

        merge_edge(
            neo4j_session, source_id=src_id, relationship=edge.relationship,
            target_id=tgt_id, chunk_id=edge.chunk_id, confidence=edge.confidence,
        )
        stats["llm_edges_written"] += 1

    if log_file:
        log_file.close()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk + AST-extract only; report LLM cost estimate, call nothing.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the LLM pass on the first N prose inputs — for a cheap "
        "validation run before committing to the full corpus.",
    )
    args = parser.parse_args()

    print("Chunking corpus...")
    chunks = chunk_corpus()
    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    print(f"  {len(chunks)} chunks: {by_kind}")

    print("Running AST pass...")
    node_universe, import_aliases, edges, node_types = run_ast_pass()
    print(f"  {len(node_universe)} nodes")
    for edge_type, edge_list in edges.items():
        print(f"  {len(edge_list)} {edge_type} facts")

    defined_in = build_defined_in_edges(node_universe)
    print(f"  {len(defined_in)} DEFINED_IN edges (derived)")

    llm_inputs = collect_llm_inputs(chunks)
    estimate = estimate_llm_cost(llm_inputs)
    print("LLM pass would read:")
    print(f"  {estimate['chunks']} chunks (doc/changelog + docstrings)")
    print(f"  ~{estimate['est_input_tokens']:,} input tokens (char-count estimate)")
    print(f"  ~${estimate['est_cost_usd']} estimated cost")

    if args.dry_run:
        print("\n--dry-run: stopping before any LLM call or Neo4j write.")
        return

    from neo4j import GraphDatabase
    from openai import OpenAI

    openai_client = OpenAI()
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "graphragpassword"))

    with driver.session() as neo4j_session:
        stats = run_full_ingestion(
            chunks, node_universe, import_aliases, edges, node_types,
            openai_client=openai_client, neo4j_session=neo4j_session, limit=args.limit,
            llm_edge_log_path="llm_edges_log.jsonl",
        )
    driver.close()

    print()
    print("=== ingestion stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
