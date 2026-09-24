# Hybrid Graph + Vector RAG — Design

**Status:** draft, pending review
**Date:** 2026-09-16
**Corpus:** `psf/requests` + `urllib3`

---

## 1. What we're building

A retrieval system that answers questions requiring two or three hops across related entities, benchmarked honestly against a plain vector RAG baseline.

**Done when:** a FastAPI endpoint answers questions with validated citations, and the README opens with a benchmark table broken out by hop count.

---

## 2. Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Corpus | `requests` + `urllib3` | urllib3 is what requests delegates to — cross-package facts never co-occur in one chunk |
| Extraction | Hybrid: AST skeleton + LLM semantics | Answers "why not just parse the AST?" instead of dodging it |
| Stores | Neo4j + Postgres/pgvector | Two stores joined on `chunk_id` is the project's thesis |
| Working mode | Pair — decide piece by piece | — |
| Timeline | 1–2 weeks part time | Full five phases, real benchmark |

### Why hybrid extraction is the strong version

A deterministic AST pass emits structural edges perfectly and for free. An LLM pass runs only over prose and emits what AST structurally cannot see. This gives you:

- **A defensible answer** to the obvious objection — "I did parse the AST, that's the cheap 60%; here's the 40% it can't reach, and here's the accuracy delta from adding it."
- **Ground truth for free.** AST edges are known-correct, so you can measure LLM extraction precision against them rather than guessing.
- **Lower cost.** The LLM only reads prose, not 18k lines of source.

---

## 3. Store topology

| Store | Holds | Does not hold |
|---|---|---|
| Postgres + pgvector | chunk text, embeddings, metadata | any relationships |
| Neo4j | nodes + edges, each edge tagged with `chunk_id` | any chunk text |

`chunk_id` is the only shared key. A graph edge gives you a `chunk_id`; Postgres turns that into the exact sentence that justified the edge. **That is the citation mechanism.**

Postgres is the single source of truth for text. Neo4j never duplicates it — so there is no way for the two to drift.

### Schema

```sql
CREATE TABLE chunks (
  chunk_id     TEXT PRIMARY KEY,   -- sha1(repo:path:start:end)
  repo         TEXT NOT NULL,      -- 'requests' | 'urllib3'
  path         TEXT NOT NULL,
  section      TEXT,               -- 'Advanced Usage > Session Objects'
  kind         TEXT NOT NULL,      -- 'code' | 'doc' | 'docstring' | 'changelog'
  start_line   INT,
  end_line     INT,
  text         TEXT NOT NULL,
  doc_hash     TEXT NOT NULL,      -- for extraction cache invalidation
  entity_ids   TEXT[],             -- canonical ids mentioned in this chunk
  embedding    VECTOR(384)
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE routing_log (          -- Phase 3 writes, Phase 5 reads
  id SERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT now(),
  question TEXT,
  route TEXT,
  confidence REAL,
  latency_ms INT,
  template_id TEXT,
  outcome TEXT
);
```

---

## 4. Ontology (frozen before code)

**8 entity types:** `Module`, `Class`, `Function`, `Exception`, `Parameter`, `Concept`, `DocSection`, `Release`

**14 relationship types:**

| AST-derived (deterministic) | LLM-derived (semantic) |
|---|---|
| `DEFINED_IN` | `DOCUMENTED_IN` |
| `INHERITS_FROM` | `EXPLAINS` |
| `CALLS` | `IMPLEMENTS` |
| `IMPORTS` | `CONTROLS` |
| `RAISES` | `DELEGATES_TO` |
| `HAS_PARAMETER` | `WRAPS_EXCEPTION` |
| | `CHANGED_IN` |
| | `CONSTRAINS` |

`DELEGATES_TO` and `WRAPS_EXCEPTION` cross the requests→urllib3 boundary. No single chunk contains both ends. These are the edges the vector baseline cannot reach — the whole benchmark rests on them.

`Concept` is prose-only ("connection pooling", "certificate verification"). It exists in no source file, which is precisely why AST can't produce it.

---

## 5. Phase 1 — Extraction

### 5.1 Chunking

| Content | Strategy |
|---|---|
| Source `.py` | one chunk per top-level def/class, split if > 1500 tokens |
| Docs `.rst` | one chunk per section heading, preserving `section` path |
| Docstrings | separate chunks, linked to their symbol |
| `HISTORY.md` | one chunk per release |

`chunk_id = sha1(f"{repo}:{path}:{start_line}:{end_line}")` — stable across reruns provided the file hasn't changed.

### 5.2 AST pass

Walk `src/` with `ast`. Emit nodes with a **fully-qualified dotted path as canonical id** (`requests.sessions.Session`). Emit the six structural edge types. Every edge carries the `chunk_id` of the code it came from.

This pass is deterministic, free, and re-runnable. It runs first and establishes the node universe the LLM pass must resolve against.

### 5.3 LLM pass

Runs over prose chunks only (docs, docstrings, changelog).

- **Model:** `claude-opus-5`
- **Structured output:** `output_config.format` with a strict schema; validate every response and retry failures with the validation error fed back
- **Prompt caching:** the ontology + few-shot prefix is byte-identical across every call — cache it, expect ~90% off the repeated prefix
- **Batch API:** halves cost; extraction is not latency-sensitive
- **Cache by `doc_hash`:** re-running ingestion on unchanged files costs nothing

Schema per extracted edge:

```json
{
  "source_surface": "Session",
  "source_type": "Class",
  "relationship": "DELEGATES_TO",
  "target_surface": "PoolManager",
  "target_type": "Class",
  "chunk_id": "...",
  "confidence": 0.0
}
```

The LLM emits *surface forms*, never canonical ids. Resolution is a separate, auditable step — see §6.

### 5.4 Cost estimate

Prose is roughly 300 chunks × ~800 input tokens ≈ 240k input tokens, ~180k output. At Opus 5 rates with batch + caching, a full cold run lands around **$5–8**. Warm reruns are ~free via the `doc_hash` cache. Budget is measured per-document before the full run.

---

## 6. Entity resolution

This is the part that gets skipped. It gets its own module and its own metrics.

`Session`, `requests.Session`, `requests.sessions.Session`, `sessions.Session`, and "the Session object" in prose must all collapse to one node.

### Resolution ladder

1. **Canonical id from AST.** The dotted path is ground truth for anything defined in code.
2. **Deterministic alias table.** Built from AST: every import alias and every `__init__.py` re-export. `requests.Session -> requests.sessions.Session` is a fact, not a guess. This covers the large majority of cases at zero risk.
3. **Normalized match.** Lowercase, strip articles and code punctuation, then exact match.
4. **Embedding similarity.** Compare the surface form (plus surrounding sentence) against canonical name + docstring first line. Accept above a **tuned** threshold.
5. **Unresolved.** Anything that falls through is written to a `_Unresolved` label — **never silently dropped.** Resolution rate is a reported metric.

### Threshold tuning

Hand-label ~100 `(surface form -> canonical id)` pairs. Sweep the threshold, report precision/recall, pick the knee. Publish the curve — "tuned threshold" without a number is the thing interviewers probe.

Each node stores `aliases[]`, `resolution_method`, and `confidence` so any resolution can be audited after the fact.

### Idempotent writes

All writes use `MERGE`, never `CREATE`. Re-running ingestion converges rather than duplicating. Every edge carries `chunk_id` and `confidence` as properties.

---

## 7. Phase 2 — Vector index

Same chunks, same `chunk_id`s, embedded into pgvector with document id, section path, `kind`, and `entity_ids`.

- **Embeddings:** local `sentence-transformers` (BGE-small, 384-dim). Verified to install on this machine's Python 3.14 — `torch 2.14.0` ships a `cp314` wheel. Keeps the project to a single API key.
- **Index:** HNSW, cosine.
- **Measure before building on it.** Label a small set of (question -> relevant chunk_id) pairs, sweep `ef_search`, report recall@k. The router does not get built on top of retrieval whose recall is unknown.

---

## 8. Phase 3 — Routing

### Classifier

- **Model:** `claude-haiku-4-5` — cheap and fast, as the brief calls for
- **Output:** strict enum — `GRAPH | VECTOR | BOTH | REFUSE`
- **Few-shot** prompt with examples drawn from each stratum
- **Low confidence -> `BOTH`**, run both paths and merge

| Route to graph | Route to vectors |
|---|---|
| connection questions | definitions |
| multi-hop chains | policy / usage lookups |
| comparisons across entities | single-fact questions |
| aggregations over relationships | |

### The Cypher template library — a security control

**The model never emits raw Cypher.** It selects a template id and fills parameters, returned through a strict schema.

| Template | Purpose |
|---|---|
| `T1_NEIGHBORS` | entity's immediate neighborhood, filtered by rel type |
| `T2_PATH_BETWEEN` | shortest path between two resolved entities |
| `T3_EXCEPTION_WRAP_CHAIN` | follow `WRAPS_EXCEPTION` transitively |
| `T4_PARAM_IMPACT` | what a `Parameter` `CONTROLS`, and where it's documented |
| `T5_DELEGATION_CHAIN` | follow `DELEGATES_TO` across the package boundary |
| `T6_COUNT_BY_REL` | aggregation over a relationship type |
| `T7_DOCS_FOR_SYMBOL` | `DocSection`s documenting a symbol |

Parameter validation before execution:

- entity ids must exist in the set resolved from the question — not model-invented
- relationship types must be members of the ontology enum
- `max_hops` bounded (≤ 4)
- values passed as Neo4j query parameters, never string-interpolated

This is both a security control and what makes results reproducible.

### Logging

Every routing decision is written to `routing_log` — question, route, confidence, template, latency, outcome. Phase 5 needs this data and it cannot be reconstructed later.

---

## 9. Phase 4 — Synthesis

1. **Verbalize paths.** Graph traversal returns triples; raw triples generate awkward text. Convert each path to a readable statement *before* it reaches the prompt: `(ConnectionError)-[:WRAPS_EXCEPTION]->(MaxRetryError)` becomes "requests.exceptions.ConnectionError wraps urllib3.exceptions.MaxRetryError."
2. **Deduplicate** across graph-derived and vector-retrieved sets by `chunk_id`.
3. **Assemble context with explicit labels** separating `GRAPH FACTS` from `RETRIEVED PASSAGES`.
4. **Require a citation per claim**, and validate it.

### Citation validation loop

- Parse every `[chunk_id]` out of the answer
- Check membership in the set actually retrieved this turn
- Any miss -> one regeneration, naming the invalid ids
- Still failing -> strip unsupported claims, return partial, flag it
- Report `citation_validity_rate` as a metric

This costs nothing and eliminates invented sources.

---

## 10. Phase 5 — Benchmark

### Question set — 60 items, stratified

| Stratum | Count |
|---|---|
| single hop | 15 |
| two hop | 15 |
| three hop | 12 |
| aggregation | 8 |
| out of scope (must refuse) | 10 |

### Building ground truth honestly

Seeding questions by sampling real paths out of the graph is efficient — and **biased toward the graph**, because every question is guaranteed answerable by traversal.

Mitigations, all of which get stated in the README:

- a portion of questions written by hand from the docs, before looking at the graph
- questions deliberately included that the graph *cannot* answer
- every item hand-verified against source, not trusted from generation
- the bias itself disclosed

Naming this weakness is what makes the rest of the number credible.

### Baseline

Same chunks, same embeddings, same generator model, top-k vector retrieval only. **The only variable is retrieval.** Anything else and the comparison means nothing.

### Reported metrics

- accuracy by hop count — both systems, same set
- latency p50 / p95, per route
- cost per query
- one-time graph ingestion cost
- citation validity rate
- refusal accuracy on out-of-scope

**Expected shape:** parity at single hop, a widening gap as hops increase. That curve is the story. Graph RAG being slower and more expensive to build gets reported plainly — that honesty is what makes the accuracy claim believable.

---

## 11. Repo layout

```
graphrag/
  ingest/
    chunk.py            # chunking strategies per content type
    ast_extract.py      # deterministic structural edges
    llm_extract.py      # semantic edges, schema-validated, cached
    resolve.py          # entity resolution ladder + alias table
    load_graph.py       # MERGE writes into Neo4j
    load_vectors.py     # embed + upsert into pgvector
  retrieval/
    router.py           # Haiku classifier -> enum
    cypher_templates.py # the template library + param validation
    graph_search.py
    vector_search.py
    merge.py            # dedupe + verbalize paths
  answer/
    synthesize.py
    citations.py        # validation loop
  eval/
    questions.yaml
    run_benchmark.py
    grade.py
  api/
    main.py             # FastAPI endpoint
docker-compose.yml      # neo4j:5 + pgvector
```

Each module has one job and can be tested without the others. `resolve.py`, `router.py`, and `cypher_templates.py` are the three files an interviewer will actually read.

---

## 12. Model choices

| Job | Model | Rationale |
|---|---|---|
| Semantic extraction | `claude-opus-5` | one-time cost, quality compounds into every downstream metric |
| Router classification | `claude-haiku-4-5` | cheap + fast, per the brief's "small model call" |
| Answer synthesis | `claude-opus-5` | quality of the final answer is what's being measured |
| Embeddings | BGE-small (local) | free, no second API key, adequate at this corpus size |

---

## 13. Build order

| Phase | Deliverable | Gate before moving on |
|---|---|---|
| 1 | Graph in Neo4j | resolution rate + AST-vs-LLM precision measured |
| 2 | pgvector index | recall@k measured on labeled set |
| 3 | Router + templates | routing decisions logging to Postgres |
| 4 | Grounded answers | citation validity rate ≥ 95% |
| 5 | Benchmark table | README opens with it |

Each gate is a number, not a vibe. Phase 2's gate exists specifically because the brief warns against building a router on unmeasured retrieval.

---

## 14. Open questions

1. **Git.** The project root is not a git repo yet. Needed before anything else.
2. **Docker.** Docker Desktop is installed but the daemon isn't running — needs to be up before Neo4j and Postgres can start.
3. **`Test` as a 9th entity type?** Tests ground behavioral claims well, but the ontology is deliberately constrained. Deferred unless the benchmark shows a gap.
4. **Extraction model.** Opus 5 is the default above. Sonnet 5 would cut extraction cost ~60%; worth a side-by-side on a 20-chunk sample before committing to the full run.

---

## 15. Known failure modes for this project

| Risk | Mitigation in this design |
|---|---|
| Open-ended ontology -> unqueryable graph | 8 entities / 14 relations, frozen in §4 before code |
| Entity resolution skipped -> one class, four disconnected nodes | §6 is its own module with its own reported metrics |
| Model writes raw Cypher | §8 template library; params validated, never interpolated |
| Corpus has no real relationships | cross-package `DELEGATES_TO` / `WRAPS_EXCEPTION` edges |
| Benchmark is self-serving | §10 discloses generation bias, includes unanswerable items |
