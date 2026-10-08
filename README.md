# Blueprint: a code-aware question answering system, and the benchmark that measures it

[![tests](https://github.com/aakarshan-coding/blueprint-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/aakarshan-coding/blueprint-ai/actions/workflows/tests.yml)

This project answers questions about the source code of two Python libraries, `requests`
and `urllib3`. Questions like "if urllib3 raises `ProtocolError`, what does requests
raise instead?" or "how many exception classes inherit from `RequestException`?"

Plain text search over the code cannot answer those well. The answer to the first
question is one line inside an `except` block; the answer to the second is a count that
appears nowhere in the text. So the system builds a **knowledge graph** of the code
first: which class inherits from which, which function raises what, which exception is
raised in place of another. Then it answers a question by looking up facts in that graph
and combining them with ordinary text search.

The main deliverable is not the system but the **measurement** of it. A 220-question
benchmark compares the graph-plus-text system against a text-only baseline, five runs
per change, with a grader that was itself audited for ways to be fooled. The graph's
contribution is the gap between the two.

**Contents:** [One question, end to end](#one-question-end-to-end) · [Results](#results) ·
[How it works](#how-it-works) · [Why it is built this way](#why-it-is-built-this-way) ·
[How the numbers were earned](#how-the-numbers-were-earned) · [Repository map](#repository-map) ·
[Quickstart](#quickstart) · [Limits and open items](#limits-and-open-items) · [Reading further](#reading-further)

## One question, end to end

Take the benchmark question *"What requests exception corresponds to urllib3's
ReadTimeoutError?"* Here is what happens to it.

1. **Routing.** A small model decides the question needs the graph, not just text.
2. **Finding the anchor.** The name `ReadTimeoutError` is read out of the question and
   resolved to a node in the graph, `urllib3.exceptions.ReadTimeoutError`. The planner
   is only ever offered nodes that really exist.
3. **Planning.** The planner picks one query template from a fixed list and fills in its
   parameters. Here: `T3_EXCEPTION_WRAP_CHAIN(entity_id='urllib3.exceptions.ReadTimeoutError',
   max_hops=2)`, meaning "follow the wraps relationship up to two steps from this node".
4. **Running and expanding.** The template runs as a fixed Cypher query. The facts it
   returns are expanded by one more step that the question shape usually needs; for a
   wrap chain, that is the parent classes of what was found.
5. **Building the context.** Each fact becomes one line, tagged by where it came from.
   `(code)` means a parser extracted it from the source; `(docs)` means a model read it
   in documentation. Trimmed:

   ```
   Relationship meanings:
   - WRAPS_EXCEPTION: this exception is raised in place of that one, inside an except handler that caught it.
   - INHERITS_FROM: this class is a subclass of that one.
   Lines marked (code) were extracted from the source by a parser; lines marked (docs) by a model
   reading documentation. Where they disagree, prefer (code).

   === GRAPH RELATIONSHIPS ===
   [c1] (code) requests.exceptions.ReadTimeout wraps urllib3.exceptions.ReadTimeoutError.
   [c2] (code) requests.exceptions.ConnectionError wraps urllib3.exceptions.ReadTimeoutError.
   [c3] (code) urllib3.exceptions.ReadTimeoutError wraps socket.timeout.
   [c4] (code) requests.exceptions.ReadTimeout inherits from requests.exceptions.Timeout.
   [c5] (code) requests.exceptions.Timeout inherits from requests.exceptions.RequestException.
   ```

   The top text-search passages are appended below the graph lines.
6. **Answering.** A model writes the answer from that context only, citing the `[c…]`
   ids. A validator rejects any citation that does not point at something retrieved.

The text-only baseline skips steps 2 to 4: it gets the same passages, the same answer
model and the same prompt, and nothing else. That is the comparison the benchmark makes.

## Results

220 questions in five categories. Each category is run five times on identical code,
because the models are not deterministic even at temperature 0. Numbers are the mean
across runs; the pooled row shows the worst and best run in brackets. An answer counts
only if it is fully correct.

| category | n | graph + text | text only | lead |
|---|---|---|---|---|
| single-hop: one fact | 45 | 95.1 | 95.1 | +0.0 |
| two-hop: two linked facts | 45 | 72.4 | 46.2 | +26.2 |
| three-hop: three linked facts | 45 | 71.1 | 28.4 | +42.7 |
| aggregation: a count or a list | 45 | 81.3 | 14.2 | +67.1 |
| out of scope: should refuse | 40 | 97.0 | 85.0 | +12.0 |
| **pooled** | 220 | **83.1** [82.3 .. 84.1] | 53.1 [52.3 .. 54.1] | **+30.0** [28.7 .. 31.4] |

With partial credit, where an answer that states two of three required parts scores
two thirds: graph + text 88.2, text only 63.0.

What each row means:

- **Single facts need no graph.** Both systems see the same passages, so they tie.
- **Linked facts are where the graph earns its keep, mostly on the last link.** The
  text-only system usually finds the first fact and loses the chain after it.
- **Counting is something text search cannot do.** No passage says "15 classes inherit
  from this one". The graph computes it, and the context states the number.
- **An empty graph result is a good refusal signal.** When the question is about Django,
  nothing in the graph matches, and the system says so.

The numbers come from [`results/exp9/`](results/exp9); reproduce the table with
`python -m graphrag.eval.summarize_runs results/exp9/*.json`. One caveat on
reproducing the runs themselves: a fresh ingest from this repository rebuilds every
parser-derived edge exactly, but the model-derived edges come out slightly different
from the graph these runs used, which had accumulated the project's history of replays
and cleanups. The gap is logged in D93 and a confirmation run on the fresh graph is the
open step. Every earlier table, going
back to the first single run, is kept in [`results/`](results/README.md) with the
decision it belongs to.

## How it works

Two phases. Ingestion runs once and builds the stores. Answering runs per question.

### Ingestion: from source files to a graph and a vector index

**1. Split the corpus into chunks.** The system needs pieces small enough to embed and
precise enough to cite. Code is split at symbol boundaries, so one function or one
class header is one chunk. Documentation is split at section headings. Every chunk gets
a stable id, and that id is the one key shared by the graph and the vector store: a graph
fact cites the chunk it came from, and that same chunk is what text search returns.
([`ingest/chunk.py`](graphrag/ingest/chunk.py))

**2. Extract structural facts with a parser.** For facts the code states outright, a
parser is both cheaper and more accurate than a model, so Python's `ast` module reads
every file and records: which class inherits from which, which exceptions a function
raises, which exception a handler raises in place of the one it caught, each function's
parameters and declared return type, and where a parameter's value is passed next.
([`ingest/ast_extract.py`](graphrag/ingest/ast_extract.py))

**3. Resolve calls with a type-aware tool.** "What does `conn.urlopen(...)` call?" needs
to know what `conn` is, which a plain parser cannot. The system uses jedi, a static
analysis library, to resolve each call to the function it reaches. When the call is to a
base-class method, an edge is also added to every subclass that overrides it, because
that is what runs at runtime. ([`ingest/jedi_calls.py`](graphrag/ingest/jedi_calls.py))

**4. Let a model read the prose, for the facts only prose has.** Documentation says
things like "this parameter controls certificate verification" that no parser can see.
A model extracts those relationships from documentation and docstrings. Its output is
logged, so later ingestion runs replay the log instead of paying for extraction again.
([`ingest/llm_extract.py`](graphrag/ingest/llm_extract.py),
[`ingest/replay_llm_edges.py`](graphrag/ingest/replay_llm_edges.py))

**5. Turn names into graph nodes.** The parser sees surface names like `ReadTimeoutError`
or `self.send`; the graph needs canonical ids like `urllib3.exceptions.ReadTimeoutError`.
A resolver tries a fixed ladder of rules, from "this exact id exists" down to "this is a
Python builtin", and gives up rather than guess. Each rung was added for a measured
reason, recorded in the log. ([`ingest/resolve.py`](graphrag/ingest/resolve.py))

**6. Write the stores.** Nodes and edges go to Neo4j, with every edge carrying the chunk
it came from and whether a parser or a model produced it. Chunk embeddings go to
Postgres with pgvector. ([`ingest/load_graph.py`](graphrag/ingest/load_graph.py),
[`ingest/load_vectors.py`](graphrag/ingest/load_vectors.py))

The result: 8 kinds of node and 17 kinds of relationship, defined in one place with the
plain-language meaning the answer model is shown for each.
([`ontology.py`](graphrag/ontology.py))

### Answering: from a question to a cited answer

**1. Route.** A small model classifies the question: graph, text, both, or refuse. A
low-confidence verdict falls back to "both". ([`retrieval/router.py`](graphrag/retrieval/router.py))

**2. Find anchors.** The system needs a node to start from. A small model lists the
code names in the question; a regular expression adds any dotted name it missed; a check
drops any name that is not actually in the question text, because the model sometimes
invents one. Each surviving name is resolved to graph nodes, which become the
candidates. If nothing resolves, the top text-search hits supply candidates instead.
([`retrieval/graph_query.py`](graphrag/retrieval/graph_query.py))

**3. Plan one query.** The model never writes a database query. It is shown the
candidates and a short list of templates with descriptions, and it fills in one
template's typed parameters. A parameter is validated before anything runs: an entity
must be one of the candidates, a relationship must be in the ontology, a hop count must
be in range. ([`retrieval/cypher_templates.py`](graphrag/retrieval/cypher_templates.py))

**4. Repair the plan where the graph proves it wrong.** The planner is a small model and
makes a few predictable mistakes: a corpus-wide count when the question named one
module, a relationship that a module cannot have, a wrap chain started from a function.
A short list of deterministic rules fixes those shapes. If a plan runs and returns
nothing, the system falls back to the next reasonable plan, such as the named module's
members. Every repair is recorded on the result so the benchmark can see how often it
fired. ([`retrieval/graph_query.py`](graphrag/retrieval/graph_query.py) `repair_plan`,
[`retrieval/retrieve.py`](graphrag/retrieval/retrieve.py))

**5. Expand by plan shape.** A question that asks "what does requests raise, and what
does that inherit from" needs two different relationships, and one template walks one.
So after the plan's facts are ranked and capped, one more hop is fetched for the nodes
that survived, chosen by the plan's shape: parents after a wrap chain, what each raised
exception wraps after a raises plan, what callees raise after a call chain. Fetching
after the cap keeps the expansion from crowding out the facts it explains.
([`retrieval/retrieve.py`](graphrag/retrieval/retrieve.py) `_EXPANSIONS`)

**6. Build the context.** Each fact becomes one sentence in the direction the edge really
runs, tagged `(code)` or `(docs)`, citing its chunk. Counting questions get derived
lines the model would otherwise have to compute: "15 classes are defined in
requests.exceptions: …", "9 members have more than one base class: …". A legend states
what each relationship means. Text passages follow.
([`retrieval/merge.py`](graphrag/retrieval/merge.py))

**7. Write and check the answer.** The answer model is instructed to use only the
context and to cite a chunk id after each claim. A validator checks that every cited id
was actually retrieved. ([`answer/synthesize.py`](graphrag/answer/synthesize.py),
[`answer/citations.py`](graphrag/answer/citations.py))

One function, `retrieve()`, runs steps 1 to 6 for both the answer pipeline and the
measurement tools, so a probe can never measure a different pipeline than the one that
answers.

## Why it is built this way

**A parser for structure, a model for prose.** The runtime oracle (below) measured it:
parser-extracted "wraps" edges are right 71% of the time against real execution,
model-extracted ones 29%. So anything the code states directly is parsed, the model is
kept to documentation, and every edge is tagged with which one produced it. The answer
model is told to prefer `(code)` lines when they disagree.

**Templates instead of model-written queries.** Letting the model write Cypher is more
flexible, and the nearest open-source project does it. This project does not, for two
reasons. A fixed template with validated parameters cannot touch anything the plan did
not name, which is the security boundary. And the same plan always produces the same
query, which is what makes retrieval stable enough to measure: on 58 of 60 questions,
five runs retrieved exactly the same facts.

**Repairs act on evidence, not labels.** One repair rule keyed on a node label that
ingestion does not assign reliably. It replaced the wrap chain on five flagship
questions with an empty query. The rule was removed and replaced with "run the planned
query; fall back only if it returns nothing". That principle now governs every repair.

**The baseline shares everything but the graph.** Same chunks, same passages, same answer
model, same prompt. An early experiment changed the shared prompt to help the graph
side and watched the baseline's refusals collapse; the rule since then is that a
graph-side change may not touch anything the baseline uses.

## How the numbers were earned

The decisions log, [`docs/DECISIONS.md`](docs/DECISIONS.md), has 92 entries. Each
records what was tried, what was measured, and what was kept or reversed. The practices
that mattered most:

- **Measure the noise before claiming a gain.** Two runs of identical code disagreed on
  about 12% of verdicts. Since then every number is a five-run mean with its spread, and
  a change counts only when its worst run clears the previous best. (D51, D60)
- **Check the graph against reality, not against itself.** The runtime oracle runs the
  `requests` test suite under `sys.settrace`, records every call, raise and wrap that
  actually happens, and scores the graph's edges against them. It is the one measure of
  the graph that does not depend on the benchmark. (D67, [`results/oracle/`](results/oracle))
- **Read the failing context before touching the model.** Three of the largest gains
  were plumbing: a deduplication keyed on chunk id was discarding every fact after the
  first from the same chunk, so nine raise edges reached the model as one; a chain
  template stopped one hop short of what the question asked; a plan anchored on a
  function ran a query only exceptions can satisfy. Those fixes took three-hop from 30
  to 67. No model changed. (D71 to D74)
- **Audit the grader.** Seven questions could be passed by repeating the question. Class
  names matched ordinary words, so "read timeout" satisfied `Timeout`. A hedge in a later
  sentence cancelled an earlier correct assertion. All fixed, with a test that no
  question passes its own grader. Every fix made grading stricter, and the strictness
  fell mostly on the baseline. (D79, D87)
- **Correct the references when the system is right.** Four hand-written answers were
  wrong; one counted typing overloads as separate functions. The 130 questions added last
  take their facts from an independent parse of both libraries, and every count is
  computed and asserted before the question is written. (D88)

Grading, for the record: 180 questions are graded by required terms with no model, 40 by
a refusal check with no model, and 10 by a model judge that lists the reference's key
facts and which are missing, with the verdict derived from that list.

## Repository map

```
graphrag/            the package, about 6,500 lines
  ontology.py        node and relationship types, each with its meaning in plain words
  ingest/            chunking, parser and jedi extraction, name resolution, store loading
  retrieval/         router, anchors, planner and repairs, templates, expansion, context
  answer/            the synthesis prompt and citation validation
  eval/              grader, benchmark, five-run summaries, runtime oracle
tests/               380 unit tests; no database and no model calls needed, about 5,700 lines
docs/DECISIONS.md    the decisions log, D1 to D92
docs/research/       two research notes: how other tools build code graphs; a close read of code-graph-rag
results/             every benchmark run the log cites, mapped to its decision
data/                the model-extraction log that ingestion replays
scripts/             fetch the corpus at pinned commits; reproduce the table end to end
```

## Quickstart

You need Docker, Python 3.12 or newer, and your own OpenAI API key in `OPENAI_API_KEY`.
Routing, planning and the judge use `gpt-4o-mini`; answers use `gpt-4o`. The key is read
from the environment and never stored.

### Use it on your own Python code

```bash
git clone https://github.com/aakarshan-coding/blueprint-ai.git && cd blueprint-ai
pip install -e ".[embed]"
docker compose up -d                        # Neo4j 5 and Postgres with pgvector
graphrag ingest ../some-python-repo         # detects the package, docs and changelog; parser pass + embeddings, no model calls
graphrag ask "What does Session.send call?" --show-plan
```

`ingest` takes several repositories at once when they import each other. Add
`--with-llm` to also run the model pass over documentation and docstrings; it costs a
few dollars for a library-sized repository and its edges are logged so a re-ingest can
replay them. `ask --show-context` prints the context the answer was written from, and
`graphrag status` shows which repositories are ingested and how much the graph and text
index hold, which is the way to ask the tool about itself: `ask` only answers questions
about the ingested code, and refuses anything else. `ask --compare` prints the vector-only
baseline's answer to the same question beneath
the graph's, which is the benchmark's comparison on a single question.

`scripts/demo.sh` runs a five-step demo on `itsdangerous`, a library the system was
never tuned on: ingest, a structural question with its plan, a side-by-side comparison,
the tagged context, and a refusal. Set `PAUSE=1` to step through it.

What a new repository gets: everything a parser can see (the 9 structural relationship
types) and text search over its docs. The model-derived relationship types need
`--with-llm`. Only Python is supported; see [limits](#limits-and-open-items).

### Reproduce the benchmark

```bash
pip install -e ".[embed,dev]"
scripts/fetch_corpus.sh                                   # requests and urllib3 at the measured commits
docker compose up -d
graphrag ingest --corpus corpora/requests-urllib3.yaml    # the measured two-library corpus
python -m graphrag.ingest.replay_llm_edges                # the model-derived edges, replayed from the log
python -m graphrag.eval.run_benchmark                     # one run, about 28 minutes, about $3
```

`scripts/reproduce.sh` does all of that and runs the benchmark five times.
`python -m graphrag.eval.runtime_oracle` scores the graph against the corpus's own test
suite in about three minutes. The unit tests need none of it: `pip install -e ".[dev]" && pytest`.

## Limits and open items

- **Python only, and measured on two libraries.** `graphrag ingest` works on any Python
  repository with a conventional layout (a package under `src/` or the root, docs in
  reStructuredText or Markdown). Another language needs a new extractor behind the same
  ontology. The benchmark numbers are specific to `requests` and `urllib3`; nothing is
  measured on another codebase.
- **Two-hop is the weakest graph category at 72.** The remaining misses are urllib3
  exception questions where two classes have similar names and the planner anchors on
  the wrong one.
- **The planner drifts when the template list grows.** Two questions changed template on
  identical inputs when the list went from six to eight. Deterministic repairs cover the
  shapes seen so far; voting the planner is the untried alternative.
- **Value flow stops at attributes.** A parameter is followed through dict literals,
  `update()` calls, returned dicts and `**kwargs`. A value stored on `self` and read back
  later is not followed.
- **The ten judge-graded questions have no human labels.** The judge once contradicted
  its own structured output in its free-text reason; only the structured field is scored.

The full list is at the end of [`docs/DECISIONS.md`](docs/DECISIONS.md) under *Open items*.

## Reading further

- [`docs/DECISIONS.md`](docs/DECISIONS.md). Good entry points: D51 (the noise floor),
  D67 (the oracle), D72 (the deduplication bug), D79 (the grader audit), D89 to D91 (the
  220-question set).
- [`docs/research/2026-09-24-codebase-knowledge-graphs.md`](docs/research/2026-09-24-codebase-knowledge-graphs.md):
  eleven tools that build graphs from code, with citations, and what this project took from each.
- [`docs/research/2026-09-26-code-graph-rag.md`](docs/research/2026-09-26-code-graph-rag.md):
  a close read of the nearest open-source project and why its model-written-query design was not adopted.
- [`docs/superpowers/specs/2026-09-16-graph-rag-design.md`](docs/superpowers/specs/2026-09-16-graph-rag-design.md):
  the original design document, written before any of the above was learned.
