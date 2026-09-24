"""Re-resolve and re-write logged LLM facts against the current Resolver,
without calling the LLM again. Exists because run_ingestion.py's first full
run had a real resolution bug (D33 — Parameter ids were never registered)
found only after the $1.74 extraction had already run; this replays the
already-paid-for raw facts through the fix instead of re-paying for them.
"""

import json

from graphrag.ingest.load_graph import merge_edge, merge_node
from graphrag.ingest.doc_release_index import lookup
from graphrag.ingest.node_ids import concept_id, parameter_concept_id
from graphrag.ingest.resolve import Resolver


def _resolve_endpoint(surface: str, entity_type: str, *, resolver: Resolver, session):
    """Resolve one endpoint of an LLM fact, or fall back to a shared concept.

    A Parameter surface that resolves to several peer candidates (`url` lives
    on 21 public documented functions, none of them canonical) is genuinely
    ambiguous — the sentence never picked one. Rather than drop the fact, it
    attaches to a concept node standing for the parameter as documentation
    discusses it, and every real candidate is linked to that concept with
    IMPLEMENTS so the graph still reaches the code.

    Returns (canonical_id, None) on success or (None, reason) on failure.
    """
    resolution = resolver.resolve(surface)
    if resolution.canonical_id is not None:
        return resolution.canonical_id, None

    if entity_type == "Parameter":
        cid = parameter_concept_id(surface)
        merge_node(session, "Concept", cid, kind="parameter", surface=surface)
        # Link the real signature slots to the shared notion, so a question
        # answered from the concept can still be traced to actual code.
        for candidate in resolution.candidates:
            merge_edge(
                session, source_id=candidate, relationship="IMPLEMENTS",
                target_id=cid, chunk_id="ast", confidence=1.0,
            )
        return cid, None

    return None, "unresolved"


def replay(
    log_path: str, *, resolver: Resolver, neo4j_session,
    doc_index: dict[str, str] | None = None,
    release_index: dict[str, str] | None = None,
) -> dict:
    stats = {
        "written": 0,
        "unresolved_docsection_release": 0,
        "unresolved_other": 0,
        "via_parameter_concept": 0,
        "via_doc_or_release": 0,
    }

    with open(log_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    for r in records:
        ids = {}
        failed = False
        for role in ("source", "target"):
            etype, surface = r[f"{role}_type"], r[f"{role}_surface"]

            if etype == "Concept":
                ids[role] = concept_id(surface)
                merge_node(neo4j_session, "Concept", ids[role])
                continue
            if etype in ("DocSection", "Release"):
                table = doc_index if etype == "DocSection" else release_index
                found = lookup(table, surface) if table else None
                if found is None:
                    stats["unresolved_docsection_release"] += 1
                    failed = True
                    break
                merge_node(
                    neo4j_session, etype, found, surface=surface
                )
                ids[role] = found
                stats["via_doc_or_release"] += 1
                continue

            resolved, reason = _resolve_endpoint(
                surface, etype, resolver=resolver, session=neo4j_session
            )
            if resolved is None:
                stats["unresolved_other"] += 1
                failed = True
                break
            if resolved.startswith("concept:parameter:"):
                stats["via_parameter_concept"] += 1
            ids[role] = resolved

        if failed:
            continue

        merge_edge(
            neo4j_session, source_id=ids["source"], relationship=r["relationship"],
            target_id=ids["target"], chunk_id=r["chunk_id"], confidence=r["confidence"],
        )
        stats["written"] += 1

    return stats


if __name__ == "__main__":
    from neo4j import GraphDatabase

    from graphrag.ingest.run_ingestion import (
        build_public_and_doc_index,
        module_of,
        run_ast_pass,
    )

    print("Rebuilding the resolver with every resolution fix applied...")
    node_universe, import_aliases, edges, node_types = run_ast_pass()
    public_ids, documented_params = build_public_and_doc_index(import_aliases)

    from graphrag.ingest.doc_release_index import build_doc_release_index
    from graphrag.ingest.run_ingestion import chunk_corpus

    doc_index, release_index = build_doc_release_index(chunk_corpus())
    print(f"  {len(doc_index)} doc sections, {len(release_index)} releases indexed")
    print(f"  {len(public_ids)} public ids, {len(documented_params)} documented functions")
    plain_resolver = Resolver(node_universe=node_universe, import_aliases=import_aliases)

    call_graph = set()
    for edge in edges["calls"]:
        module_context = module_of(edge.source_id, node_types)
        for surface in edge.calls:
            r = plain_resolver.resolve(surface, module_context=module_context)
            if r.canonical_id is not None and node_types.get(r.canonical_id) == "Function":
                call_graph.add((edge.source_id, r.canonical_id))
    print(f"  {len(call_graph)} caller->callee pairs in the call graph")

    resolver = Resolver(
        node_universe=node_universe,
        import_aliases=import_aliases,
        call_graph=call_graph,
        public_ids=public_ids,
        documented_params=documented_params,
    )

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "graphragpassword"))
    with driver.session() as session:
        stats = replay("llm_edges_log.jsonl", resolver=resolver, neo4j_session=session,
                       doc_index=doc_index, release_index=release_index)
    driver.close()

    print(stats)
