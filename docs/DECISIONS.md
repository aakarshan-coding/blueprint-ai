# Decisions Log

Why this system looks the way it does. The spec says *what* it is; this says *why*.
Newest at the bottom. Written to be read cold, months later.

---

## D1 — Corpus: `requests` + `urllib3`, not one library

**Decided:** index two libraries, not one.

**Why:** `requests` is a friendly wrapper; `urllib3` does the actual network work.
They're separate projects with separate docs, so nobody documents the seam between them.
Example: `requests.ConnectionError` really means "urllib3 ran out of retries" — that fact
exists only as one `except` clause in `adapters.py`, in no prose anywhere.

Those cross-library facts are what vector search structurally cannot find, because no
single chunk contains both halves. That gap is the entire reason this project beats a
vector baseline.

**Cost:** ~3x the corpus. Still cheap.

---

## D2 — Hybrid extraction: AST for structure, LLM for meaning

**Decided:** a plain Python parser extracts structural facts; Claude only reads prose.

**Why:** Python's `ast` module knows *exactly* which class inherits from which, for free
and instantly. Paying a model to guess at that is slower, costlier, and less accurate.
Claude gets the facts a parser can't see — concepts described in English, the
requests→urllib3 handoff, what a parameter controls.

**Also:** it answers the obvious interview objection. "Why not just parse the AST?" →
"I did. Here's the 40% it can't reach, and here's the accuracy it adds."

**Cost:** two extraction paths to maintain instead of one.

---

## D3 — Two databases joined by `chunk_id`

**Decided:** Postgres holds text + embeddings. Neo4j holds entities + relationships.
Neither stores the other's data.

**Why:** a graph edge carries the `chunk_id` of the text that justified it. Follow the id
into Postgres and you get the exact sentence. That *is* the citation mechanism.

Because Postgres is the only place text lives, the two stores cannot drift out of sync.

---

## D4 — Chunk size is a parameter, not a constant

**Decided:** `max_chars` defaults to 2000, but it's an argument.

**Why:** the right value isn't knowable yet. It's settled by the recall@k measurement in
Phase 2. Hardcoding a guess would mean you can't sweep it later.

**What the data showed:** the limit only affects the tail — median chunk size barely
moves between settings. Below ~1000 it stops working entirely, because code examples
refuse to be split, so you get 400-char prose sitting next to 1800-char code.

**Non-obvious risk:** chunk-length *variance* is itself a retrieval problem. Cosine
similarity isn't comparable across wildly different lengths — long chunks accumulate
topic mass and quietly win. Most writeups only discuss the mean.

**Honest status:** 2000 was chosen by judgment, not evidence. Shape metrics (how many
chunks, how big) say nothing about whether retrieval works. That needs labeled questions.

---

## D5 — One chunk per symbol, never split by size

**Decided:** each function, class, and method is its own chunk. Code is never cut by
character count.

**Why:** when a graph edge says "`HTTPAdapter.send` delegates to `PoolManager`," the
citation should point at `send`, not at a 900-line file. And half a function is neither
readable as evidence nor valid as a citation.

**Cost:** one function came out at 16,016 characters.

---

## D6 — `text` and `embed_text`: one chunk, two views

**Decided:** every chunk stores what you *cite* (`text`, the full symbol) separately from
what you *search on* (`embed_text`, signature + docstring). Prose chunks use the same
string for both.

**Why:** these were originally two separate chunks — a code chunk and a docstring chunk.
That was wrong. The docstring text was stored twice, the two chunks competed for the same
queries, and they could collide on the same id.

The real insight: searching and citing are different jobs and don't need the same text.
Most RAG implementations conflate them and are forced to choose.

**The decisive constraint:** the embedding model (BGE-small) caps at 512 tokens, ~2000
characters, and **silently discards** anything beyond. So for large chunks you were never
embedding the whole thing — you were embedding whatever happened to come first.
`embed_text` doesn't avoid truncation; it decides what the 512 tokens get spent on.

**Measured effect:** 1,828 chunks → 1,396. Truncated chunks 67 → 33 (4.8% → 2.4%).
Largest code embedding 16,016ch → 6,895ch.

**Honest status:** halved, not solved. 33 chunks still overflow. And the fallback for
undocumented symbols (embed the full body) just relocates the problem for 12 of them.

**Known cost:** the function body contains searchable terms the summary throws away.
"Which function catches `LocationValueError`?" matches on full text, misses on a summary.
Real tradeoff, unresolved.

---

## D7 — Accept truncation on the 33 oversized chunks, for now

**Decided:** don't fix the 33 chunks whose `embed_text` still overflows the embedding
model's 512-token limit. Ship with the truncation, document it, revisit only if measured.

**Options considered:**
- **Split large functions at AST statement boundaries** (e.g. break `urlopen` into its
  `try` block, its redirect handling, etc. as separate searchable fragments, all citing
  the same whole function). Correct fix — doesn't throw anything away — but makes
  `embed_text` one-to-many with a chunk, which changes the Postgres schema (a fragments
  table keyed by `chunk_id`) and needs dedup-by-`chunk_id` at query time so one function
  can't fill multiple result slots.
- **Docstring only, drop the signature.** Cheap, loses parameter names as search terms.
- **A bigger embedding model** (some allow 8000 tokens vs. 512). Solves it with zero
  chunking changes, but is its own tradeoff (slower, heavier) — not evaluated.
- **Accept it.** ← chosen.

**Why accept:** 33 of 1,396 chunks (2.4%). No evidence yet that it actually hurts
retrieval — that requires the recall@k measurement Phase 2 already calls for. Building
schema changes for an unconfirmed problem is exactly the kind of unmeasured work the
project's own design argues against (see D4 — chunk size wasn't tuned until measured
either).

**Revisit when:** Phase 2's recall@k shows these specific chunks failing to retrieve.

---

## D8 — Structural edges cite raw surface names, not canonical ids

**Decided:** `extract_inherits_from` (and the other AST edges to follow) emit a class's
bases exactly as written — `"IOError"`, `"CompatJSONDecodeError"` — not resolved dotted
ids. Resolution is entirely resolve.py's job (§6 of the design), done later using the
alias table this same AST pass also produces.

**Why:** a base class can be local (resolvable immediately), an external builtin (never
resolvable — `IOError` has no node), or an imported name (resolvable once `IMPORTS` edges
exist). This pass only asserts what `ast` can prove outright about the file in front of
it; deciding what a name *refers to* needs project-wide context this pass doesn't have.
Keeping the two jobs apart means a raw fact is never wrong, only not yet resolved.

**Also decided:** one `INHERITS_FROM` edge per class, carrying its full base list, not
one edge per base pair. `ConnectTimeout(ConnectionError, Timeout)` is one edge with two
targets — its bases are one fact about the class, not two independent facts.

**Verified against real code:** the full `requests.exceptions` hierarchy (25 classes)
extracts correctly, including multiple inheritance and mixed builtins/warnings.

---

## D9 — Imports are the alias table

**Decided:** `extract_imports` records every name an import brings into scope, resolved
to its full dotted path (`MaxRetryError` → `urllib3.exceptions.MaxRetryError`), including
relative imports resolved against the importing module's own path.

**Why:** this is exactly the alias table §6 needs. When code later says
`except MaxRetryError`, resolution just looks up `MaxRetryError` in this module's import
edges — no guessing.

**Verified against real code:** `adapters.py` imports 20 names from `urllib3`, including
`MaxRetryError` — the name the flagship `WRAPS_EXCEPTION` example depends on.

---

## D10 — RAISES is per-function, not per-statement

**Decided:** `extract_raises` returns one edge per function, listing every exception it
raises anywhere inside it (including nested try/except), not one edge per `raise` line.

**Why:** consistent with D8's inheritance edges — a function's exception surface is one
fact about the function.

**Verified against real code:** `HTTPAdapter.send` raises 13 exceptions, including
`ConnectTimeout` — the one raised from the `MaxRetryError` handler. Combined with D9's
import edges (`MaxRetryError` → `urllib3.exceptions.MaxRetryError`), this is what
resolve.py will use to derive the flagship fact: `send` raises `ConnectTimeout` in
response to urllib3's `MaxRetryError`.

**Bug caught while checking:** two methods in the file are both named `send`
(`BaseAdapter.send`, `HTTPAdapter.send`). A naive `endswith(".send")` filter on the
dotted id silently grabs whichever comes first in the file. Dotted ids disambiguate
correctly; string-suffix matching on them does not.

---

## D11 — HAS_PARAMETER drops self/cls, keeps defaults as source text

**Decided:** `extract_parameters` skips a leading `self`/`cls`, and records each
default as the code text of the expression (`"None"`, `"True"`), not an evaluated value.

**Why:** `self` names the method's own object, not a fact about what callers pass — not
worth an edge. Defaults are kept as text because some (a class reference, a sentinel)
aren't safely eval-able without the module's full context; source text is always correct
even when a real value isn't obtainable.

**Verified against real code:** `Session.request`'s 16 real parameters extract in order,
including `verify` — the parameter the "SSL Cert Verification" doc section (D-earlier)
describes, and a `CONTROLS` edge target once resolve.py runs.

**AST pass complete:** all six structural edges from the spec are implemented —
`DEFINED_IN` (implicit in every node's dotted id), `INHERITS_FROM`, `CALLS` (next),
`IMPORTS`, `RAISES`, `HAS_PARAMETER`.

---

## D12 — CALLS records the surface expression, resolvable or not

**Decided:** `extract_calls` records every call target as written — `"self.get_connection"`,
`"conn.urlopen"`, `"os.path.join"` — without trying to resolve `conn`'s type.

**Why:** `self.x` always resolves, because `self` means the enclosing class — resolve.py
can settle those immediately. `conn.urlopen` can't be resolved from this file alone;
`conn`'s type isn't known without type inference, which this pass doesn't attempt. Same
call recorded either way — resolvability is resolve.py's problem, not this pass's.

**Verified against real code:** captured `conn.urlopen` inside `HTTPAdapter.send` — the
literal call where requests hands off to urllib3.

**AST pass complete.** All six structural edges implemented: `INHERITS_FROM`, `IMPORTS`,
`RAISES`, `HAS_PARAMETER`, `CALLS`, and `DEFINED_IN` (implicit in every dotted id).

---

## D13 — LLM extraction uses the OpenAI API, not Anthropic

**Decided:** `extract_relationships` calls OpenAI's Responses API
(`client.responses.parse(..., text_format=PydanticModel)`), not Claude, for the
prose extraction pass.

**Why:** no Anthropic API key was available in this environment; the user chose to
switch providers rather than set one up. Contradicts the design doc's Phase 1/§12
("Claude Opus 5" for extraction) and the resume line's implicit framing — that's a
real gap between what's written and what's built, not just an implementation detail.

**What made the swap cheap:** `extract_relationships()` was already the only place any
LLM client touches the pipeline (chunking, AST extraction, resolution, Neo4j writes are
all provider-agnostic), and its test suite ([test_llm_extract.py](../tests/test_llm_extract.py))
was written against a fake client, not a real one. Swapping providers meant changing one
file's internals and the fake client's shape in tests, not the pipeline.

**Real API differences found by inspecting the installed SDK rather than guessing:**
- `client.messages.parse(output_format=...)` → `client.responses.parse(text_format=...)`
- response field `parsed_output` → `output_parsed`
- `system=` + `messages=[...]` → `instructions=` + `input=`

**Known gap, unresolved:** `MODEL = "gpt-4o"` is a placeholder picked only because it's
a model name that's verifiably real, not because it was evaluated against this task.
Needs a deliberate choice before the real extraction run.

**Verified end to end with one real API call**, on the "Session Objects" doc chunk:
extracted `Session --[DELEGATES_TO]--> urllib3` correctly, with the right `chunk_id`
attached. One extraction looked off — an `EXPLAINS` edge sourced from `Session` where
the doc section itself should probably be the source — worth checking across more
chunks before trusting the ontology mapping at scale.

**If this ships as a portfolio piece:** the design doc, model-choice table (§12), and
resume line need to say "OpenAI" wherever they currently say "Claude" — or the two
providers need to be reconciled before anyone reads this project closely.

---

## D14 — Entity resolution: four rungs implemented, embeddings deferred

**Decided:** `resolve.py`'s `Resolver` implements the first four rungs of §6's ladder in
order — exact canonical id, per-module import alias, package re-export, then name
matching (exact case first, case/space-folded second) — and stops there. Anything left
unresolved carries its candidate list rather than being dropped or guessed at.

**Bug caught by running against the real corpus:** the first version case-folded every
name before matching, which collided classes with modules that share a name only when
case is ignored — `PoolManager` (a class) vs `poolmanager` (the module it lives in),
`Session` (a class) vs a real `def session()` convenience function in the same file.
Fixed by trying an exact-case match first and only falling back to folded matching
(needed for prose like "Connection Error") when that fails. Caught with a regression
test before touching the fix, per TDD.

**Verified against the real corpus** (888 nodes across both packages): `MaxRetryError`
resolves via import alias to `urllib3.exceptions.MaxRetryError`; `PoolManager`, `Session`,
`Response` all resolve to their one real class; `IOError` correctly stays unresolved
with no candidates (external builtin, no node exists for it, no false match either).

**Deferred:** step 4 of the ladder (embedding similarity for near-misses) isn't built.
Nothing so far has needed it — every ambiguous case seen has resolved once the exact-case
bug was fixed. Building it now would be tuning a threshold against cases that don't
exist yet; revisit if real ambiguous candidates show up once the LLM pass runs at scale.

---

## D15 — DEFINED_IN is derived at load time, not extracted

**Decided:** no extractor emits `DEFINED_IN` directly. `build_defined_in_edges` derives
it from dotted-id containment when writing to Neo4j — if a node's id, minus its last
segment, is itself a known node, that's a `DEFINED_IN` edge.

**Why:** the containment fact is already fully encoded in every dotted id
(`requests.adapters.HTTPAdapter.send` implies "defined in `HTTPAdapter`"). Extracting it
separately would just be re-stating something already true by construction. But leaving
it un-materialized would force every "what's defined in this module" query to string-split
ids in Cypher — an actual edge is what makes that a normal traversal.

**Also decided:** ontology names now live in one place, `ontology.py`, instead of being
duplicated across modules. `llm_extract.py`'s `Literal` couldn't be derived from it
directly (Python typing can't build a `Literal` from a runtime set), so a test instead
asserts the two stay identical — drift becomes a failing test, not a silent gap.

**Verified against real code:** 888 nodes across both packages produce 886 `DEFINED_IN`
edges — every node has a parent except the two package roots, which is correct.

**Phase 1 building blocks are now all built and individually verified against real data:**
chunking, AST extraction, LLM extraction, entity resolution, graph writes. Not yet built:
the orchestration script that runs all five over the whole corpus in one pass — that
needs Docker running (for Neo4j) and a decision to spend real OpenAI budget on the full run.

---

## D16 — Real infrastructure is up; load_graph.py verified against real Neo4j

**Decided:** `docker-compose.yml` runs Neo4j 5 and `pgvector/pgvector:pg16` locally.
Brought up successfully; both reachable from Python (`neo4j` driver, `psycopg`).

**Verified for real, not against the fake session this time:** wrote the flagship fact
(`Session --[DELEGATES_TO]--> PoolManager`, citing the real `chunk_id` from D13's live
LLM call) into an actual Neo4j instance, then ran the identical writes a second time.
Node count stayed at 2, edge count stayed at 1 — `MERGE` idempotency holds against real
infrastructure, not just the mocked assertions in test_load_graph.py.

**Phase 1 is now fully verified end to end for a single fact**, from raw doc prose all
the way to a queryable edge in a running graph database. What's left is running all five
steps over the *whole* corpus, not proving the mechanism works.

---

## D17 — Docstrings pulled back out just for the LLM pass

**Decided:** added `iter_docstrings_for_llm` to chunk.py — not a `Chunk`, never stored
or embedded, purely a `(symbol, chunk_id, docstring_text)` triple for feeding the LLM
pass. The `chunk_id` matches the code chunk the docstring lives inside.

**Why:** D6 removed docstrings as their own stored chunk to kill duplicate text and a
real id-collision bug. That was the right call for storage — but it also silently
removed docstrings from the LLM extraction pass's input, and docstrings are exactly
where `CONTROLS`/`EXPLAINS`-type facts often live (`:param verify: whether to verify SSL
certs`). This restores that input without undoing D6 — nothing new is persisted, this
exists only to build one API call's prompt.

## D18 — run_ingestion.py orchestrates all five Phase 1 steps; dry-run measured for real

**Decided:** built `run_ingestion.py`, tying chunking → AST pass → LLM-input collection →
(resolution + Neo4j load, not yet wired) into one script. `--dry-run` does everything
except spend money or touch Neo4j, and ends with a real cost estimate.

**Measured against the full real corpus** (not an estimate from before code existed):

| | |
|---|---|
| Chunks | 1,396 (882 code, 231 doc, 283 changelog) |
| Graph nodes | 888 |
| Structural edges | 1,586 raw + 886 derived `DEFINED_IN` |
| LLM pass would read | 946 chunks (doc + changelog + docstrings) |
| Estimated cost | **~$1.74** |

Cheaper than the design doc's original $5–8 estimate (§5.4) — that figure assumed Opus 5
pricing; the actual extraction model is gpt-4o (D13), priced lower, and docstrings are
short. The estimate itself is a char-count proxy (÷3.5), not a real tokenizer count —
good enough for a go/no-go call, not for billing.

**Not yet built:** the full-run branch (LLM pass + resolution + Neo4j writes for every
chunk) — it currently raises `NotImplementedError` on purpose, so `--dry-run` is the only
way to run this script until that's built and the cost is approved.

---

## D19 — Embeddings + pgvector built and verified against real infrastructure

**Decided:** `embed.py` wraps `sentence-transformers` (BGE-small, 384-dim), model
injected so tests never load real weights. `schema.sql` defines the Postgres table +
HNSW index. `load_vectors.py` upserts chunks with `ON CONFLICT (chunk_id) DO UPDATE` —
the SQL equivalent of `MERGE`, same idempotency principle as `load_graph.py`.

**Verified for real, not just against fakes:**
- Loaded the actual model, confirmed 384 dimensions, cached locally
- Applied `schema.sql` to the live Postgres container
- Wrote 5 real embedded chunks, reran the exact same write, row count stayed at 5
- **Proved the design's central mechanic end to end**: pulled the real `DELEGATES_TO`
  edge out of Neo4j, took its `chunk_id`, looked that id up in Postgres, and got back the
  actual "Session Objects" sentence that justifies the edge. Two databases, one join key,
  both populated with real data, working together — not simulated.

**Honest note on embedding quality:** a quick similarity check (not a real recall@k
measurement) showed weak separation between clearly-different sentences (scores
0.73–0.78, not sharply spread). Expected at this stage — the real recall@k measurement
against a labeled question set is what Phase 2 actually requires before trusting this.

---

## D20 — Whole corpus embedded into Postgres

**Decided:** ran embedding + upsert over all 1,396 chunks (not a sample). Local model,
no API cost. ~164s total. Row count in Postgres: 1,396 — matches exactly.

**Still not done:** recall@k measurement against a labeled question set. The spec is
explicit that the router doesn't get built until this number exists — vectors being
*loaded* isn't the same as vectors being *known to work*.

---

## D21 — recall@k measured for real: 67% at k=1, 100% at k=5

**Decided:** built `recall_at_k` (pure function, unit tested) plus a hand-labeled set of
6 real questions in `recall_labels.yaml`, each paired with the exact chunk_id(s) that
should answer it (verified by reading the real chunk text first, not generated from the
graph — that would bias toward answerable-by-construction questions).

**Measured against the real, fully-loaded vector index:**

| k | recall |
|---|---|
| 1 | 67% (4/6) |
| 3 | 83% (5/6) |
| 5 | 100% (6/6) |
| 10 | 100% (6/6) |

**The one miss at k=1/k=3:** "Does requests follow redirects automatically?" — the
correct chunk ranked 4th, beaten by `Response.is_redirect` (a code chunk) and a
"Retrying Requests" doc section. All three are topically close; the embedding wasn't
wrong to find them relevant, just not decisive about which is most relevant.

**What this means for later design gates:** k=5 already gets 100% on this small set —
that's evidence for choosing k≥5 in Phase 3 retrieval, not just habit. 6 questions is too
small to trust the exact percentages, but the shape (sharp gain from k=1 to k=5, flat
after) is a real signal.

**Gate satisfied:** the design explicitly says not to build the router on unmeasured
retrieval. This is that measurement — small, but real and honest about its size.

---

## D22 — ef_search swept: no effect at this corpus size, kept at default (40)

**Decided:** kept pgvector's `hnsw.ef_search` at its default (40) rather than raising it.

**Measured** across 10, 40, 100, 200 against the same 6 labeled questions:

| ef_search | recall@1 | recall@5 | avg latency |
|---|---|---|---|
| 10 | 4/6 | 6/6 | 23.4ms |
| 40 | 4/6 | 6/6 | 5.2ms |
| 100 | 4/6 | 6/6 | 4.4ms |
| 200 | 4/6 | 6/6 | 4.4ms |

**Why it made no difference:** `ef_search` trades search speed for accuracy by
controlling how many candidates HNSW checks before returning results — it matters when
the index is too large to search almost-exhaustively. At 1,396 vectors, Postgres can
check nearly everything regardless of the setting, so there's no real approximation
happening yet for this knob to trade against. Confirmed real: had to first run a vector
operator in-session before `hnsw.ef_search` even appeared as a recognized setting —
pgvector's custom GUCs register lazily, per backend, on first use.

**Revisit when:** the corpus grows by an order of magnitude or more (e.g. the requests +
urllib3 + httpx stretch goal from the original brief). At that scale this sweep should be
rerun — this result doesn't generalize past the corpus it was measured on.

---

## D23 — Router built: classify + low-confidence fallback

**Decided:** `router.py` has two pieces, tested separately. `effective_route` is pure
logic (no API) — it applies the low-confidence-falls-back-to-BOTH rule from §8,
including for an uncertain `REFUSE`: an unsure refusal still runs both paths rather than
trusting the refusal, since the fallback exists precisely to catch uncertain guesses.
`classify_question` is the one cheap model call (`gpt-4o-mini`) that produces a route +
confidence via structured output — same `client.responses.parse` pattern as
`llm_extract.py`, tested against a fake client first.

**Blocked on a real check:** tried 4 representative questions against the live API and
hit `insufficient_quota` — the OpenAI key currently set has no credits. Not a code bug;
`effective_route` and `classify_question`'s call shape are both verified by their 9 unit
tests, but the router's actual classification *behavior* on real questions is unverified
until credits are added and this run succeeds.

**Real check, once a funded key was available:** 4 representative questions, 3 correct.

| Question | Got | Expected |
|---|---|---|
| Which requests exception wraps urllib3's MaxRetryError? | GRAPH (0.90) | GRAPH ✓ |
| What does the verify parameter do? | VECTOR (0.90) | VECTOR ✓ |
| How do requests/urllib3 differ on connection pooling? | GRAPH (0.85) | GRAPH ✓ |
| How do I use requests with Django templates? | VECTOR (0.80) | REFUSE ✗ |

**The miss is a real prompt gap, not noise.** The Django question mentions "requests," so
the model treated it as in-scope and answered from the vector index instead of refusing.
The system prompt currently defines REFUSE as "not about requests or urllib3 at all" —
too literal. It should mean "not something this corpus can actually answer," which is a
different, harder judgment the prompt doesn't currently ask for. Needs a prompt fix
before Phase 5's out-of-scope question stratum runs against this, or REFUSE accuracy
will look worse than the router's real routing accuracy warrants.

**Also note:** the credit-balance debugging along the way was real — the first key had a
project-level spending cap of $0 separate from the org's actual balance, which produces
the identical `insufficient_quota` error as a genuinely empty account. Worth remembering
if this recurs: check the specific project's limit, not just the org balance.

**Fixed and reverified.** Reworded REFUSE's definition from "doesn't mention
requests/urllib3" to "can't be answered from the requests/urllib3 docs and source, even
if it mentions them by name," with an explicit framework-integration example. Reran the
Django question plus a second out-of-scope case (Flask sessions) — both now correctly
REFUSE at 0.90 confidence, and all 4 original questions still route correctly. 5/5 on
the real API, 9/9 unit tests still green.

**Still open:** the Cypher template library (§8's other half) — not started yet.

## D24 — Cypher template library: real bug caught before real infrastructure hit it

**Decided:** `cypher_templates.py` implements 6 of the 7 templates from §8, validated
generically by parameter *kind* (`entity_id`, `relationship_type`, `hop_limit`) rather
than per-template — so a new template can't skip a check an older one remembered.
`entity_id` values are only accepted if they're in the set the question already
resolved; `relationship_type` values must be in the 14-name ontology; `hop_limit` is
bounded to 1–4.

**Real bug caught by testing against live Neo4j, not just the fake session:** the first
version wrote `[:WRAPS_EXCEPTION*1..$max_hops]`, assuming a variable-length relationship
bound could be a runtime parameter like any other value. Neo4j rejects this outright —
`CypherSyntaxError: Parameter maps cannot be used in MATCH patterns`. The fake-session
tests all passed regardless, because a fake session never validates real Cypher syntax —
this is exactly why "wire it to a real database" is its own step, not implied by unit
tests passing.

**Fix, same shape as `T6`'s relationship-type handling:** `relationship_type` and
`hop_limit` are the two param kinds Cypher structurally can't bind at runtime (a label, a
type, or a hop count all have to be literal query text). Both get interpolated into the
query string, but only *after* `_validate_param` has already checked them — a hop count
against a numeric range, a relationship name against the ontology set. `entity_id`
values stay as real bound parameters, since those are just property values Cypher can
bind normally.

**Verified against real Neo4j** with a real 2-hop chain
(`ConnectTimeout -[:WRAPS_EXCEPTION]-> MaxRetryError -[:WRAPS_EXCEPTION]-> NewConnectionError`):
`max_hops=1` correctly stops at one hop, `max_hops=2` correctly returns the full chain —
this is the project's flagship multi-hop example, working end to end against a real
database. `T1_NEIGHBORS` and `T6_COUNT_BY_REL` also verified for real.

**Not yet built:** `T2_PATH_BETWEEN` and `T4_PARAM_IMPACT` from §8's full list of 7 —
deferred, not yet needed by anything downstream.

## D25 — Routing decisions logged for real

**Decided:** added `routing_log` to `schema.sql` and `log_decision` in `routing_log.py`,
same fake-then-real verification pattern as everything else in Phase 1/2.

**Verified end to end with real infrastructure**: a real question went through the real
router (`classify_question` → `GRAPH`, confidence 0.9, 3,465ms), and the result was
written to the real `routing_log` table in Postgres and read back correctly — timestamp
auto-populated, all fields intact.

**Still open:** the piece that actually resolves entities out of a question and picks a
specific template + parameters to run — the part that turns a `GRAPH`-routed question
into an actual `run_template` call. Router, templates, and logging all exist and work;
they aren't wired together into one path yet.

## D26 — Question → template plan → resolved entity → real answer, wired end to end

**Decided:** built `graph_query.py`. `plan_graph_query` is one cheap model call that
reads a `GRAPH`-routed question and picks a template + names what it needs, in plain
English (`entity_surface="ConnectTimeout"`) — the model never sees or invents a
canonical id. `build_template_values` then resolves that surface name through the same
`Resolver` used in ingestion, and only keeps the values the chosen template actually
needs (via `TEMPLATES[...].params`).

**The security boundary this preserves:** the model chooses *what to look up, by name*.
Our own code — not the model — decides what that name resolves to. `GraphQueryPlan`'s
schema has no field for a raw canonical id, so there's no way for the model to skip
resolution and hand `run_template` an id it invented.

**One real bug caught before it shipped:** built `relationship: Literal[tuple(sorted(...))]`
from the runtime `RELATIONSHIP_TYPES` set, since `Literal` can't easily be built from a
collection. This relies on an obscure typing quirk (a single tuple argument to `Literal`
unpacks into individual values) that could easily have silently degraded into one
literal tuple value instead of 14 real choices. Checked directly: a real relationship
name is accepted, a fake one raises `ValidationError` — confirmed genuine, not a silent
no-op.

**Verified fully end to end, real API + real resolver + real Neo4j, two questions:**
- *"What does ConnectTimeout ultimately wrap, up to 2 levels deep?"* → correctly planned
  as `T3_EXCEPTION_WRAP_CHAIN`, resolved `ConnectTimeout` → `requests.exceptions.ConnectTimeout`,
  returned the real 2-hop chain to `NewConnectionError`.
- *"What is directly connected to Session?"* → correctly planned as `T1_NEIGHBORS`,
  returned the real `DELEGATES_TO → PoolManager` edge with its citation `chunk_id`.

**Phase 3's graph path is now fully wired**, from a plain English question to a cited
graph answer. What's not yet built: the vector path's equivalent wiring, and the merge
step that combines both when the router says `BOTH`.

## D27 — Vector path wired: search_chunks

**Decided:** `vector_search.py`'s `search_chunks` embeds a question and returns its
nearest chunks. Default `k=5` — not a guess, that's the exact value D21's real recall@k
measurement showed hits 100% on this corpus, versus 67% at k=1.

**Verified against the real, fully-loaded corpus**: results for "Does requests verify
SSL certificates by default?" matched the earlier ad-hoc recall@k script exactly — same
top chunk, same distance, same ranking. `search_chunks` is now the real, reusable
function doing what that one-off script did by hand.

**Phase 3 status:** both retrieval paths (graph, vector) are now independently wired
end to end. Still missing: the merge step for when the router says `BOTH`.

## D28 — Fixed a real citation gap: chain templates had no per-hop chunk_id

**Decided:** `T3_EXCEPTION_WRAP_CHAIN` and `T5_DELEGATION_CHAIN` now return
`chunk_ids` alongside `chain` — one chunk_id per edge in the path.

**Why this mattered:** caught while starting Phase 4's "turn graph facts into sentences"
step. Both chain templates returned only the sequence of node ids — nothing to cite. A
sentence built from `chain` alone ("ConnectTimeout wraps MaxRetryError wraps
NewConnectionError") would have no chunk_id to attach, which breaks Phase 4's citation
requirement before it even starts. `T1_NEIGHBORS` and `T7_DOCS_FOR_SYMBOL` were already
fine — this gap was specific to the two transitive-chain templates.

**Verified against the real 2-hop chain** (`ConnectTimeout -> MaxRetryError -> NewConnectionError`):
1-hop query returns `chunk_ids: ['c1']`, 2-hop query returns `chunk_ids: ['c1', 'c2']` —
aligned correctly with each edge in the chain, in order.

## D29 — Verbalizer built: raw graph results become cited sentences

**Decided:** `merge.py`'s `verbalize` turns each template's raw result shape into
`GraphFact(statement, chunk_id)` objects — one function per template shape, dispatched
by `template_id`, since `T1_NEIGHBORS`'s rows look nothing like `T6_COUNT_BY_REL`'s.

**Real problem solved, not just anticipated:** Neo4j's variable-length match returns one
row *per path length* up to `max_hops` — a 2-hop query returns both the 1-hop and 2-hop
paths, which share their first edge. Verbalizing each row independently would produce a
duplicate sentence for that shared edge. Fixed by flattening every returned chain into
its individual (source, target, chunk_id) edges and deduplicating on that triple before
turning any of it into text — the dedup happens at the edge level, not the sentence
level, so it's correct regardless of how many overlapping paths Neo4j returns.

**`REL_PHRASES` covers exactly the 14 ontology relationships** (`INHERITS_FROM` →
"inherits from", `WRAPS_EXCEPTION` → "wraps", etc.), and a test asserts the mapping's
keys equal `ontology.RELATIONSHIP_TYPES` exactly — a relationship added to the ontology
without a phrase here now fails a test instead of silently producing broken text the
first time it appears in a real answer.

**`T6_COUNT_BY_REL` facts carry no `chunk_id`** — deliberately. A count is an aggregate
over the whole graph, not something one chunk of text justifies, so citing a chunk for
it would be a fabricated citation.

**Verified against the real chain**: `ConnectTimeout wraps MaxRetryError` (cites `c1`)
and `MaxRetryError wraps NewConnectionError` (cites `c2`) — two sentences, not three,
correctly deduplicated, each with its own real citation.

## D30 — Phase 4 complete: full pipeline verified end to end on a real question

**Decided:** built the three remaining Phase 4 pieces —
- `assemble_context` (merge.py): labels graph facts and vector passages separately,
  dedupes by `chunk_id` across both sources, returns the set of ids actually included
- `citations.py`: pure regex-based `extract_citations` + `validate_citations` — an
  answer is only valid if it cites *something* and every citation it makes was actually
  retrieved this turn; zero citations is a fail, not a pass by default
- `synthesize.py`: the final model call, instructed to end every claim with a
  `[chunk_id]` tag or say it doesn't know

**Full real run, novel question** ("What does Session delegate to for connection
pooling, and how does it keep connections alive?"):

```
Router: GRAPH (0.8) -> effective GRAPH
Plan: T5_DELEGATION_CHAIN, entity_surface="Session"
Graph fact: requests.sessions.Session delegates to urllib3.poolmanager.PoolManager. [e1f3ec2877dc6c73]
Answer: "The Session delegates to urllib3.poolmanager.PoolManager for connection
         pooling. However, the context does not provide information on how it
         keeps connections alive [e1f3ec2877dc6c73]."
Citations valid: True, invalid: []
```

**Real observation, not a bug:** the router chose `GRAPH` only, so vector search never
ran — and the model correctly, honestly declined to answer the "keep connections alive"
half rather than inventing an answer. That's the citation discipline working as
intended. But this question arguably needed `BOTH` (a graph fact plus the Keep-Alive doc
passage). Worth revisiting during Phase 5's router tuning — logged here rather than
"fixed" now, since it's a judgment call about routing confidence thresholds, not a clear
error the way the Django question was.

**All five phases' core mechanism is now built and proven for real, end to end**:
extraction → resolution → graph write → vector write → routing → retrieval → merged,
cited answer. 110 tests passing. What remains is Phase 5 (the benchmark against a plain
vector baseline) and running the full corpus through ingestion rather than a handful of
hand-verified examples.

## D31 — Full corpus ingestion built: node typing, id schemes for 4 more entity types

**Decided:** before Phase 5 could get a real graph to benchmark against, the ontology's
remaining entity types needed id schemes — `node_ids.py` adds `parameter_id` (scoped
under its function), `release_id` (scoped by repo), `doc_section_id` (reuses the doc
chunk's own `chunk_id` rather than inventing a second identity for the same thing), and
`concept_id` (normalized surface text — a Concept's only identity is what the LLM called
it, so its id has to come from that text directly).

**Explicit scope cut, not a silent gap:** classes are all typed uniformly as `"Class"`,
never `"Exception"`. Distinguishing them needs walking the (not-yet-resolved-at-typing-
time) `INHERITS_FROM` chain to check whether a class transitively derives from a real
exception base — real work, not done here. `ENTITY_TYPES` still allows `"Exception"`;
nothing currently produces it.

**Second scope cut:** LLM-extracted edges where the model typed an endpoint as
`DocSection` or `Release` are counted as unresolved and skipped, not written. Resolver
only knows Module/Class/Function/Exception/Parameter ids — teaching it to also resolve
"Session Objects" (prose) against a `DocSection` id is separate, unbuilt work. Counted
honestly (`llm_edges_unresolved_docsection_release` in the run's stats), not hidden.

**Validated on a 5-chunk slice before committing to the full spend:** no crashes; 816
AST edges written, 2,206 unresolved. Checked what the unresolved ones actually were
rather than assuming the number was fine: `inherits_from`/`raises` unresolved were all
Python builtins (`ValueError`, `OSError`, `IOError`) that were never meant to resolve;
`calls` unresolved were mostly local-variable method calls (`conn.urlopen`-shaped) and
builtins (`len`, `int`) — exactly D12's documented, expected limitation of AST-only
resolution, not a bug.

**Full corpus run (946 chunks) launched after validation passed, but was interrupted**
when the session ended before completion — not a crash, just cut off. Real state,
checked directly against Neo4j rather than assumed:

| | |
|---|---|
| AST pass | complete — 888 nodes typed (692 Function, 142 Class, 54 Module), 510 CALLS, 96 RAISES, 82 INHERITS_FROM |
| LLM pass | partial — only 7 WRAPS_EXCEPTION edges exist; the other 7 LLM relationship types have zero |
| Postgres | untouched, still 1,396 chunks |

**Real bug found while checking this, not fixed yet:** 54 nodes have no entity-type
label — external stdlib names (`http.cookiejar.CookieJar`, `enum.Enum`, `base64.b64encode`)
that `resolve.py` correctly resolves (the import really does point there), but which
`_write_ast_nodes` never types, because they're outside our own two-repo `node_universe`
by definition — we never AST-parsed the standard library. The *edges* pointing at them
are accurate facts (e.g. some function genuinely does call `base64.b64encode`); only the
target nodes are untyped. Not corrupted data, just unlabeled nodes. Not fixed now —
the ontology has no "External" type, and adding one is a bigger call than fixing this
inline deserves.

**Not resumable from a checkpoint** — `run_ingestion.py` has no progress tracking, so
rerunning restarts the LLM pass from chunk 0. Re-running is safe for correctness (`MERGE`
is idempotent) but re-spends money on the chunks already processed before the interruption.

## D32 — Full corpus ingestion completed for real: 2,999 nodes, all 12 relationship types present

**Decided:** re-ran the full ingestion to completion (cleared the graph first for a clean
count; ran with unbuffered output logged to a file this time, after the earlier
background run got cut off by a session boundary — see D31's note).

**Real final numbers, checked directly against Neo4j:**

| | |
|---|---|
| Total nodes | 2,999 (1,135 Parameter, 831 Concept, 692 Function, 142 Class, 54 Module, 145 unlabeled external) |
| AST edges | 816 written, 2,206 unresolved (builtins/local-variable calls, per D12/D28) |
| LLM calls | 946/946 succeeded, 0 failed |
| LLM edges extracted | 2,271 |
| LLM edges written | 238 |
| LLM edges skipped (DocSection/Release, D31's scope cut) | 788 |
| LLM edges unresolved (other) | 1,245 |

**All 8 LLM relationship types are present with real counts** — `CONTROLS` (183),
`INHERITS_FROM`-adjacent `WRAPS_EXCEPTION` (112), `IMPLEMENTS` (143), `DELEGATES_TO`
(69), `CONSTRAINS` (51), `DOCUMENTED_IN` (45), `EXPLAINS` (38), `CHANGED_IN` (18). This
is a real, populated knowledge graph, not a handful of hand-built test edges.

**Checked whether the low LLM write rate (238/2,271 ≈ 10%) is a bug, not just reported
the number:** sampled live extraction on 3 real doc chunks (8 edges). Findings:
- Most failures were **correctly** unresolved — genuine external references
  (`RequestException wraps IOError`, `ChunkedEncodingError wraps httplib.IncompleteRead`
  — `httplib` is Python 2's old stdlib name, not part of this corpus at all). The
  resolver rejecting these is right, not a gap.
- A **real, separate quality issue** surfaced: `MissingSchema wraps
  requests.exceptions.MissingSchema` — the model extracted a class wrapping *itself*
  under its fully-qualified name. Harmless (collapses to a self-loop after resolution),
  but reveals that **`confidence` is extracted but never used to filter what gets
  written** — every edge is written regardless of how confident the model was in it.
  Worth adding a confidence floor before Phase 5, not fixed now.
- Sample size is 8 edges — real evidence the ~10% write rate isn't obviously a
  resolution bug, but too small to characterize what the full 1,245 "unresolved-other"
  edges actually are.

**Known, already-logged gaps that still stand:** the `HAS_PARAMETER` relationship type
is never actually emitted with that label — parameter membership is written as
`DEFINED_IN(parameter, function)` instead, which is a naming inconsistency against the
ontology (the fact is correct, the edge type name isn't the one the ontology reserves
for it). 145 nodes (up from 54) are unlabeled externals, per the earlier accepted
decision. `Exception` is still never used as a type — everything class-shaped is `Class`.

**Phase 1 is now genuinely complete against real, full-scale data** — not a proof of
mechanism on a handful of hand-picked examples. Ready for Phase 5's benchmark.

## D33 — Real bug found and fixed: Parameter facts could never resolve; deeper limitation found and left alone

**Decided:** `run_ast_pass` now registers every `Parameter` id (via `parameter_id`) into
the node universe the `Resolver` searches, alongside Module/Class/Function ids.

**The bug:** the first full run's `Resolver` never knew `Parameter` ids existed at all
— `_write_parameters` created real `Parameter` nodes in Neo4j directly, but never told
the `Resolver` about them. Every LLM-extracted fact about a parameter — including the
project's own flagship example, `verify controls certificate verification` — was
therefore unresolvable by construction, not by legitimate rejection. Found by actually
reading the logged failures (see below) instead of trusting the aggregate number.

**Why this was fixable without re-spending the extraction cost:** D32's earlier run had
already spent the ~$1.74 and thrown the raw results away after writing to Neo4j. Fixed
that first — `run_full_ingestion` now takes `llm_edge_log_path` and writes every
extracted fact to `llm_edges_log.jsonl` before any resolution filtering, regardless of
outcome. Re-ran the full extraction once more (946 calls, all succeeded) specifically to
get a complete log, then wrote `replay_llm_edges.py`, which re-resolves and re-writes
every logged fact against a fixed `Resolver` — no LLM calls, pure local computation.

**A deeper, genuine limitation surfaced once the bug was fixed, not "fixed away":**
adding `Parameter` ids only raised the write rate modestly (255 → 273). Checked why
directly: `resolver.resolve("verify")` returns **9 candidates** —
`Session.request.verify`, `HTTPAdapter.send.verify`, and 7 others — because `verify` is
the same conceptual flag threaded through 9 different function signatures across two
repos as it's passed down a call chain. The `Resolver` is correctly refusing to guess
which one a documentation sentence means, exactly as designed (§6's non-negotiable:
never silently pick one). Properly solving this — recognizing `verify` as one canonical
concept rather than 9 disconnected per-function copies, or picking a "public API owner"
function as its canonical home — is real, separate design work. Not attempted here;
logged as a known, structural limitation of per-function `Parameter` scoping.

**Real final graph state, replayed and confirmed against Neo4j:**

| relationship | count |
|---|---|
| DEFINED_IN | 2,021 |
| CALLS | 510 |
| CONTROLS | 135 |
| RAISES | 96 |
| INHERITS_FROM | 82 |
| WRAPS_EXCEPTION | 37 |
| IMPLEMENTS | 37 |
| EXPLAINS | 31 |
| CONSTRAINS | 30 |
| DELEGATES_TO | 21 |
| DOCUMENTED_IN | 19 |
| CHANGED_IN | 16 |

2,677 total nodes. All 12 relationship types genuinely populated from real extraction —
not hand-built test data. This is the graph Phase 5's benchmark will run against.

## D34 — Call-graph disambiguation built, correct, but doesn't fix the real "verify" case

**Decided:** `Resolver` now takes an optional `call_graph` (caller, callee) set. When
normalized matching finds multiple candidates, it prefers the one no other candidate's
owning function calls into — the "outermost" one — and only if exactly one candidate
qualifies; otherwise it stays unresolved exactly as before. Unit-tested (13 tests,
including the exact "verify on 2 functions, one forwards to the other" scenario) — the
logic is correct.

**Real check against the actual corpus: it doesn't help.** Built the real call graph
from the already-extracted, already-resolved `CALLS` edges (265 real caller→callee
pairs) and tried it on `verify`, `stream`, `timeout`, `headers` — all stayed unresolved.

**Root cause, confirmed by inspection, not assumed:** none of `verify`'s 9 candidate-
owning functions ever appear calling each other in the call graph. The real forwarding
path is `Session.request` → `Session.send` → `adapter.send(...)`, where `adapter` is a
local variable whose concrete type (`HTTPAdapter`) isn't known without type inference.
This is the exact same limitation as `conn.urlopen` (D12/D28) — dynamic dispatch through
a variable is invisible to an AST-only call graph — just surfacing in a new place.

**Conclusion:** this is a correct, tested feature that happens not to move the needle on
the specific case that motivated it, because the real blocker is one level down (no type
inference), not the disambiguation logic itself. Properly fixing the `verify` case would
mean inferring `adapter`'s type from `self.get_adapter(...)`'s return type — real,
separate work, out of scope here. The call-graph rung stays in `resolve.py` (harmless,
and may help in corpora with more directly-visible call chains); the `verify`-style
ambiguity remains a documented, accepted limitation, not silently declared "fixed."

---

## D35 — External nodes typed by role; two real bugs found in the process

**Decided:** external symbols (stdlib, third-party) now get an entity type inferred from
the edge role that reached them — `INHERITS_FROM` target → `Class`, `RAISES` → `Exception`,
`CALLS` → `Function` — plus `origin="external"` and `type_source="inferred"` properties.
The ontology stays frozen at 8 types; provenance lives in a property, not a label, because
"external" answers *where it came from*, not *what it is* — a different axis, and jamming
both into one label field forces a false choice.

`infer_external_type` returns `None` for relationships that imply nothing definite
(`DELEGATES_TO`, `CONTROLS`), leaving those nodes unlabeled rather than asserting a type
the edge doesn't support. Known weakness, recorded rather than hidden: a class constructor
call (`CookieJar()`) is syntactically identical to a function call, so `CALLS` can mis-type
a class — which is exactly why `type_source="inferred"` is queryable.

**Bug 1, self-inflicted, caught by verifying instead of trusting the happy path:** the
first backfill created **49 duplicate nodes**. `MERGE (n:Class {id: $id})` matches on
label *and* id, so it does not match an existing *unlabeled* node with that id — it
creates a second one. `merge_edge` routinely leaves unlabeled anchors behind, so every
external node got twinned, breaking `id` as a unique key. Fixed `merge_node` to
`MERGE (n {id: $id}) SET n:Type` — match on id alone, then add the label. Added
`label_if_unlabeled` for the inferred case, so when `CALLS` and `RAISES` disagree about
the same external symbol, the reliable inference (applied first) wins instead of the last
one processed.

**Bug 2, pre-existing and worse, found while checking bug 1's fix:** ids like
`.exceptions.ConnectionError` — leading dot, malformed. `_resolve_relative` dropped one
segment for every relative import, which is right for a regular module
(`requests.adapters` → `requests`) but wrong inside a package's `__init__.py`:
`module_dotted_name` already strips `__init__`, so `requests/__init__.py` *is* `requests`
and has no segment to drop. Dropping one anyway produced an empty package. Real
consequence: `requests.exceptions.ConnectionError` was being split across two nodes — the
real one and a garbage twin — with real `WRAPS_EXCEPTION` / `DELEGATES_TO` / `CONTROLS`
edges attached to the wrong one. Fixed by passing `is_package` through to
`_resolve_relative`. Verified: **0 malformed import targets** across the whole corpus.

**Graph repaired without re-spending on extraction**, using the raw fact log from D33:
deleted the 49 duplicates and 12 malformed nodes, replayed all LLM edges, rewrote all 820
AST edges against corrected ids.

| | before | after |
|---|---|---|
| unlabeled nodes | 145 | **3** |
| duplicate ids | 49 | **0** |
| malformed ids | 12 | **0** |

Final: 2,666 nodes — 1,135 Parameter, 730 Function, 594 Concept, 147 Class, 54 Module,
3 Exception, 3 unlabeled. 124 tests passing.

---

## D36 — Parameter ambiguity solved: public-API + docstring filter

**Decided:** `Resolver` takes `public_ids` and `documented_params`. When several
candidates share a name, it keeps only those whose owning function is (a) exported from
a package's `__init__` and (b) documents that parameter with a `:param <name>:` line. If
exactly one survives, it resolves (`method="public_api"`); otherwise it falls through to
the call-graph rung and then to unresolved, with the *narrowed* candidate list.

**Why this works where the call-graph attempt (D34) didn't:** documentation talks about a
parameter as users meet it — on the public, documented entry point — not as it's forwarded
through internals. `HTTPAdapter` isn't exported from `requests/__init__.py` at all, so
every `HTTPAdapter.*` candidate drops out immediately. That export table only became
trustworthy once D35 fixed relative-import resolution inside `__init__` files — this fix
was literally unavailable before that bug was found.

**Measured on the real corpus:**

| parameter | candidates before | after filter |
|---|---|---|
| `verify` | 9 | **1** — `Session.request.verify` ✓ |
| `stream` | 8 | **1** ✓ |
| `timeout` | 30 | 6 |
| `headers` | 34 | 5 |
| `proxies` | 11 | 3 |

The ones still ambiguous are *legitimately* ambiguous — `timeout` is a real, distinct
parameter in both `requests` and `urllib3`, so declining to guess is correct behaviour,
not a shortfall.

**Real graph impact:** LLM edges written 275 → **332** (+21%). The flagship fact
`verify --CONTROLS--> concept:certificateverification` is now in the graph, along with
`stream`, `auth`, `hooks`, and `allow_redirects` — the parameter-level facts that the
`CONTROLS` relationship type existed for, and that were entirely missing through D32.

Final graph: 2,688 nodes, 0 duplicates, all 12 relationship types populated.
127 tests passing.

---

## D37 — Phase 5 benchmark question set written (60 questions, hand-authored)

**Decided:** `benchmark_questions.yaml` holds 60 questions matching §10's stratification
exactly — 15 single-hop, 15 two-hop, 12 three-hop, 8 aggregation, 10 out-of-scope.

**Written from source material, not from the graph** — the bias §10 explicitly warns
about. Generating questions by sampling graph paths guarantees every question is
answerable by traversal, which rigs the comparison before it starts. These were written
by reading the actual docs and source; each carries a `source` field naming where its
answer was verified, so any claim is checkable.

**Two-hop questions are grounded in verified code, not guesses:** the `except`/`raise`
blocks in `HTTPAdapter.send` were read directly to get the real urllib3→requests
exception mappings (`ProtocolError` → `ConnectionError`, `ReadTimeoutError` →
`ReadTimeout`, `MaxRetryError` → `ConnectTimeout`/`RetryError`/`ProxyError`/`SSLError`
depending on `.reason`, etc.). These span the two-library seam no single doc page
describes — the exact gap the project claims to close.

**Out-of-scope questions are adversarial on purpose:** several mention "requests" or
"urllib3" by name (Django integration, Flask sessions, httpx comparison) precisely
because D23 found the router keyword-matching on those words instead of judging real
scope. If that regression returns, this stratum catches it.

**Not yet built:** the runner that executes both systems over this set, the vector-only
baseline, and the grading. Question set is the input, not the result.

---

## D38 — Vector-only baseline built; the predicted failure showed up immediately

**Decided:** `eval/baseline.py`'s `answer_vector_only` reuses the *same* components as
the hybrid path — same chunks, same embedding model, same `k=5`, same
`assemble_context`, same `synthesize_answer` model, same `validate_citations`. The only
difference is that it never consults the graph. If the prompt or model also differed, a
win couldn't be attributed to retrieval and the benchmark would prove nothing.

**Fairness fix made along the way:** `assemble_context` used to print a section header
even when that section was empty, so the baseline would have seen a bare
`=== GRAPH FACTS ===` heading with nothing under it — a source that exists and is
silent, which the model shouldn't have to interpret. Empty sections are now omitted
entirely. This helps both systems (the hybrid hits the same case when one path returns
nothing), so it's a genuine improvement rather than a baseline-specific accommodation.

**First real run, and the predicted curve appeared unprompted:**

| question | baseline result |
|---|---|
| "Does requests verify SSL certificates by default?" (1 hop) | correct ✓ |
| "If urllib3 raises ProtocolError, what does requests raise?" (2 hop) | *"The context does not provide specific information..."* ✗ |

The two-hop failure is exactly the thesis: that fact exists only in one `except` clause
in `adapters.py` spanning two libraries, so no single retrieved passage contains it.
Notably the baseline **failed honestly** rather than hallucinating — the citation
discipline holds even when the retrieval can't answer.

**Still to build:** the runner that executes both systems over all 60 questions, and the
grading.

---

## D39 — Benchmark runner built; a measurement artifact caught before it became a result

**Decided:** built the three missing pieces — `answer/pipeline.py` (the hybrid path as
one reusable function; it had only ever existed as an ad-hoc demo script),
`eval/grade.py` (grading), and `eval/run_benchmark.py` (the runner).

**Grading uses two different methods on purpose.** Out-of-scope questions are graded
*mechanically* — did the answer decline? That's checkable, so no model is involved and no
model bias can enter. Everything else goes to an LLM judge against the hand-written
reference, with `JUDGE_RUBRIC` published in the source rather than hidden. The judge
never sees which system produced an answer. Only `correct` counts toward accuracy;
`partial` deliberately does not, so the headline can't be inflated by half-answers.

**A validation run on 3 questions produced a fake result, and catching it mattered
more than the result itself.** Hybrid scored 66.7% vs baseline 100% on single-hop — a
−33pt gap. Inspecting it: all three routed to `VECTOR`, meaning **the two systems ran
identical retrieval and built identical context**. They were the same system. The judge
had simply graded two near-identical answers differently (`sh-03`: both said to pass a
dict to `cookies`; one got "partial").

**Root cause: nondeterministic sampling, not capability.** Fixed by setting
`temperature=0` on every model call — router, graph planner, answer synthesis, and the
judge.

| | before | after |
|---|---|---|
| single-hop delta | **−33.3pt** (artifact) | **+0.0pt** |
| byte-identical answers on VECTOR-routed questions | 0/3 | 2/3 |

**Why this is worth recording rather than quietly fixing:** had this gone unnoticed, the
published table would have shown the hybrid system *losing* on easy questions, and the
honest-looking explanation ("graph adds overhead without helping on simple lookups")
would have been completely wrong. The number would have been defensible-sounding and
false. A VECTOR-routed question is a free control — the two systems are provably
identical there, so any measured delta on those is pure noise, and it should read zero.

---

## D40 — First full benchmark: the hybrid system LOST, and the reason was a bug

**Result of the first real 60-question run** — recorded as it came out, not as hoped:

| category | n | hybrid | baseline | delta |
|---|---|---|---|---|
| single_hop | 15 | 53.3% | 60.0% | **−6.7pt** |
| two_hop | 15 | 26.7% | 40.0% | **−13.3pt** |
| three_hop | 12 | 16.7% | 16.7% | +0.0pt |
| aggregation | 8 | 0.0% | 12.5% | **−12.5pt** |
| out_of_scope | 10 | 90.0% | 70.0% | **+20.0pt** |

The hybrid system lost almost everywhere. That is the exact opposite of the project's
thesis, and the honest first reading is "the graph isn't earning its cost."

**Diagnosis before accepting that reading — and it was a bug, not a finding.** 24
questions routed to `GRAPH`. Of those, **15 got zero graph facts back**. Because a
`GRAPH`-only route never runs vector search, those 15 questions were answered from an
*empty context*: literally *"No supporting information was retrieved for this question."*
Meanwhile the baseline had 5 retrieved passages for the same question.

So a quarter of the benchmark wasn't measuring "graph vs vector" at all — it was
measuring "nothing vs vector", which vector wins by default.

**Fix:** `answer_hybrid` now falls back to vector search when a `GRAPH` route produces no
facts, and records `fell_back_to_vector` so the behaviour is visible in results rather
than silent. §8's design had a low-confidence → `BOTH` fallback but nothing for the
"graph came back empty" case; that gap is what this closes.

**What this run is still worth, even though the headline is invalid:** the
`out_of_scope` column is real and unaffected by the bug — the hybrid system refuses
correctly 90% of the time vs the baseline's 70%, because the router's REFUSE path exists
and plain vector search has no way to decline. That +20pt is a genuine, measured
advantage.

**Also still unexplained and NOT attributable to this bug:** `aggregation` scored 0.0%
for the hybrid even though half those questions did get graph facts. Counting questions
("how many exceptions wrap a urllib3 exception?") are supposed to be the graph's
strongest category. That needs its own diagnosis after the re-run.

---

## D41 — Diagnosing aggregation's 0% found the worst bug in the project: false facts

**Investigated why aggregation scored 0.0%** even on questions that did receive graph
facts. Three separate causes, one of them serious.

**Cause 1 (serious): `T1_NEIGHBORS` was producing factually false statements.** The
template matches edges in both directions (`-[r]-`), which is correct — "what connects to
X" should find edges pointing *at* X. But `_verbalize_neighbors` always rendered the
queried entity as the grammatical subject. Verified against the real graph:

| the graph actually contains | what the model was told |
|---|---|
| `HTTPAdapter INHERITS_FROM BaseAdapter` | ❌ "BaseAdapter inherits from HTTPAdapter" |
| `BaseAdapter.__init__ DEFINED_IN BaseAdapter` | ❌ "BaseAdapter is defined in BaseAdapter.__init__" |

Every incoming edge was reversed. The hybrid system was feeding the answer model
**false statements presented as retrieved fact** — strictly worse than returning nothing,
and it explains the "the context does not contain enough information" answers: the model
was being handed claims that contradicted themselves and sensibly declined to use them.

Fixed: the template now returns `startNode(r).id = $entity_id AS outgoing`, and the
verbalizer renders the neighbour as subject for incoming edges. Verified against the real
graph — all four sampled statements are now true.

**Cause 2: `T6_COUNT_BY_REL` counts globally, with no way to scope to an entity.**
"How many requests exceptions inherit from RequestException?" got answered with "there
are 82 INHERITS_FROM relationships" — a true number, answering a different question.
The template vocabulary has no entity-scoped count. Not yet fixed.

**Cause 3: entity resolution found nothing** for several aggregation questions
(`ag-03`, `ag-05`), so the graph path returned empty — now mitigated by D40's
vector fallback.

**Why this justifies having diagnosed before re-running:** two of these three causes
would have persisted through a re-run, and the false-facts bug would have kept
suppressing exactly the categories the project claims as its strength — while the
numbers looked like an honest negative result about graph RAG.

---

## D42 — T8_RELATED_BY: entity-scoped relationship queries, fixing aggregation

**Decided:** added `T8_RELATED_BY(entity_id, relationship)` — everything connected to one
entity by one relationship type, with direction. Chosen over adding an entity-scoped
*count* template because the same query answers both shapes the aggregation questions
take: "which classes inherit from BaseAdapter" is the list, "how many do" is the list's
length. A count template would only answer one.

The planner prompt now tells the model to prefer `T8_RELATED_BY` over `T6_COUNT_BY_REL`
whenever a question names an entity, since T6 counts across the entire graph and can't
be scoped — that mismatch is what produced "there are 82 INHERITS_FROM relationships" in
answer to "how many requests exceptions inherit from RequestException?".

**Verified against the real graph on the three aggregation questions that scored 0%:**

| question | result |
|---|---|
| how many inherit from `RequestException`? | 15 correct facts |
| which classes inherit from `BaseAdapter`? | 1 — `HTTPAdapter` ✓ |
| which exceptions does `HTTPAdapter.send` raise? | 8 correct facts |

All statements are correctly directed (D41's fix holds through the new template, which
reuses the same verbalizer path). 147 tests passing.

---

## D43 — Corrected benchmark, and the real reason the graph doesn't win

**Second full run, with D40/D41/D42's fixes in place:**

| category | n | hybrid | baseline | delta | (run 1) |
|---|---|---|---|---|---|
| single_hop | 15 | 60.0% | 66.7% | −6.7pt | −6.7 |
| two_hop | 15 | 33.3% | 40.0% | −6.7pt | −13.3 |
| three_hop | 12 | 16.7% | 16.7% | +0.0pt | +0.0 |
| aggregation | 8 | 12.5% | 12.5% | +0.0pt | −12.5 |
| out_of_scope | 10 | 90.0% | 60.0% | **+30.0pt** | +20.0 |

The fixes moved every category they were meant to. **The thesis still isn't
demonstrated** — the hybrid system ties or slightly trails on the hop-count
categories. Recording that plainly.

**Root cause, and it is not another bug: the graph does not contain the facts the
questions need.** Checked directly:

| fact the two-hop questions ask about | edges in the graph |
|---|---|
| `ProtocolError` → `ConnectionError` | 0 |
| `MaxRetryError` → `ConnectTimeout` | 0 |
| `ReadTimeoutError` → `ReadTimeout` | 0 |

**Why they're missing — a real gap in D2's extraction split.** These facts live in the
`except`/`raise` blocks of `adapters.py`:

- the **AST pass** extracts `RAISES` (what a function throws) but has no concept of
  "throws X *in response to catching* Y" — it never inspects except handlers
- the **LLM pass** reads prose only (docs, docstrings, changelog), by deliberate design

So exception-wrapping — the project's flagship relationship, the thing the
requests+urllib3 corpus was *chosen* for (D1) — is extracted by neither pass. The 37
`WRAPS_EXCEPTION` edges that exist came from prose mentions and are mostly weak
(`Timeout wraps concept:requesttimedout`).

D2's framing was "AST for structure, LLM for meaning." The gap is a third category
neither covers: **facts that are structural in the code but semantic in meaning** — an
except/raise pair is plainly visible in the syntax tree, yet expresses a relationship,
not a structure.

**Honesty note on what comes next.** This is the third round of fix-and-rerun, which is
exactly how a benchmark gets tuned until it says what its author wanted. The distinction
being drawn: `ProtocolError → ConnectionError` is objectively true, objectively present
in the source, and objectively absent from the graph. Extracting it adds a missing true
fact. That is different from adjusting scoring, prompts, or question wording to move a
number — none of which has been done, and none of which should be.

---

## D44 — extract_exception_wrapping: the missing third extraction category

**Decided:** added `extract_exception_wrapping` — a deterministic AST pass over `except`
handlers that pairs each caught exception type with each exception raised in that
handler. Free (no LLM), exact, and it closes the gap D43 identified.

**Design details worth keeping:**
- `except (A, B)` produces one edge per caught type — two facts, not one
- a `raise` nested inside an `if` within the handler still counts; the raise is a
  consequence of that catch regardless of the control flow around it
- a bare `raise` (re-raising the same exception) produces nothing — that isn't wrapping
- `raised WRAPS_EXCEPTION caught`, so the direction reads "the requests exception wraps
  the urllib3 one"

**Real extraction from `adapters.py` — exactly the facts the benchmark asks for:**

```
catches ProtocolError      -> raises ConnectionError
catches MaxRetryError      -> raises ConnectTimeout
catches MaxRetryError      -> raises RetryError / ProxyError / SSLError
catches ClosedPoolError    -> raises ConnectionError
catches LocationValueError -> raises InvalidURL
```

**Written to the graph:** 88 wrapping facts found corpus-wide, 38 resolved and written,
50 unresolved (aliased imports like `_SSLError`, and stdlib types outside the corpus).
`WRAPS_EXCEPTION` edges went **37 → 73**. Three of the four flagship benchmark facts now
exist in the graph; `ReadTimeoutError → ReadTimeout` is still missing because it's caught
under an aliased import the resolver couldn't map.

151 tests passing.

---

## D45 — The benchmark is underpowered, and I had been over-reading it

**Third run, with the exception-wrapping facts now in the graph:**

| category | n | hybrid | baseline | delta |
|---|---|---|---|---|
| single_hop | 15 | 66.7% | 66.7% | +0.0pt |
| two_hop | 15 | 20.0% | 40.0% | −20.0pt |
| three_hop | 12 | 16.7% | 16.7% | +0.0pt |
| aggregation | 8 | 12.5% | 12.5% | +0.0pt |
| out_of_scope | 10 | 70.0% | 60.0% | +10.0pt |

Two-hop got *worse* despite adding exactly the facts those questions ask about, and
out-of-scope fell 90% → 70% even though nothing in the refusal path changed. That second
one is impossible as a real effect, which is what prompted measuring the noise instead of
explaining the number.

**Measured run-to-run variance on identical code:** the baseline path is byte-identical
between runs 2 and 3. **3 of its 60 verdicts still flipped.** A ~5% noise floor from
nothing. (`temperature=0` reduces nondeterminism; it does not eliminate it, and the LLM
judge has its own variance on borderline answers.)

| | |
|---|---|
| noise floor | ~5% of verdicts flip with zero code change |
| hybrid flips | 9/60 vs baseline 3/60 |
| one question is worth | 6.7pt (n=15) → 12.5pt (n=8) |
| **resolvable delta** | **>13pt**; anything smaller is indistinguishable from noise |

**Why the hybrid is noisier to measure than the baseline** — a real property, not an
accident: it makes three model calls per question (router, graph planner, synthesis)
against the baseline's one. Three times the stochastic steps, three times the chances to
land differently. A hybrid architecture is intrinsically harder to benchmark than a
simple one.

**The honest correction to everything reported in D40/D43:** those deltas were inside the
error bars and I presented them as findings. Across all three runs, the only result that
holds directionally is **out_of_scope favouring the hybrid** (+20, +30, +10 — always
positive, magnitude swinging with noise). Every hop-count delta has been noise.

**What this benchmark can honestly say right now:** nothing about hop-count accuracy. It
is not powerful enough at n=8–15 per category to detect the effect it was built to
detect. Fixing that means more questions per category, repeated runs with error bars, or
both — not another round of fixes to the system.

---

## D46 — Systematic debugging found the root cause four fixes had missed

**Invoked `systematic-debugging` after four consecutive fix-and-rerun cycles failed to
move the result.** Its rule at 3+ failed fixes is to stop fixing and question
fundamentals, which is exactly what was needed — the previous four fixes were all real
bugs, but none was *the* problem.

**Phase 1 evidence — the clue that had been walked past:** on three-hop and aggregation,
**both systems score identically** (16.7%, 12.5%) and stably across all three runs. Two
systems with different retrieval failing identically means the cause isn't in retrieval.
Both failed on 10 of 12 three-hop questions.

Checking whether the graph contained the needed facts separated "retrieval failed" from
"facts absent":

| question needs | in graph? |
|---|---|
| `get → Session.request → Session.send → HTTPAdapter.send` call chain | **0 edges** |
| `ConnectTimeoutError → ConnectTimeout` | 0 edges |
| 15 exceptions inheriting from `RequestException` | present (15) |

**Phase 3 hypothesis, confirmed:** `self.X` call surfaces never resolved. `self.send`
inside `Session.request` returned nothing, even though `requests.sessions.Session.send`
was in the node universe the whole time. Resolution had rules for imports, re-exports,
and name matching — but none mapping `self` to the enclosing class, the one receiver
whose type needs no inference at all.

**Scale: 451 of 2,646 call surfaces (17%) were `self.*`** — silently dropped. These are
the *most* reliable calls in any codebase.

**Phase 4 fix and measured result:**

| | before | after |
|---|---|---|
| call surfaces resolved | 608 (23.0%) | **811 (30.7%)** |
| CALLS edges in graph | 512 | **691** |
| `Session.request → Session.send` | missing | present |

**This retroactively explains D34.** The call-graph disambiguation looked useless and
was written off as "the call graph is too sparse because of dynamic dispatch" — but a
large part of that sparsity was this bug, not dynamic dispatch. A wrong diagnosis was
recorded as a finding; it stood for several decisions before systematic investigation
overturned it.

**The lesson worth keeping:** four rounds of "find a bug, fix it, re-run" each fixed
something genuinely broken and never touched the actual cause. What broke the cycle was
the discipline of asking *why do both systems fail identically* instead of asking *what
else can I fix*.

---

## D47 — Parameters as concepts: the `url` case, and why it wasn't a resolution problem

**Decided:** when a `Parameter` surface resolves to several peer candidates, the fact
attaches to `concept:parameter:<name>` — the parameter as *documentation* discusses it —
instead of being dropped. Every real candidate is linked to that concept with
`IMPLEMENTS`, so an answer drawn from the concept still traces back to actual code.

**The distinction that makes this correct rather than a dodge** — `verify` and `url` fail
for opposite reasons:

| | `verify` (D36) | `url` |
|---|---|---|
| shape | vertical chain — one public entry, 8 internal passthroughs | 21 horizontal peers (`get`, `post`, `put`, `Session.request`, …) |
| canonical owner | exists (top of the chain) | **does not exist, and never will** |
| right answer | resolve to the owner | there isn't one to resolve to |

`requests.get` does not call `requests.post`. They are siblings. When docs say "the `url`
parameter", the honest referent is *all of them* — so no cleverer rule could have found a
winner. The mistake wasn't weak resolution; it was modelling one shared notion as 62
separate entities and then demanding prose pick one.

**Diagnosis that led here:** 94% of Parameter nodes (1,069 of 1,135) carried no semantic
facts at all — only a structural `DEFINED_IN`. The 6% that did were all *behaviour
switches* (`allow_redirects`, `trust_env`, `backoff_jitter`, `verify`), never *data
inputs* (`url`, `data`, `headers`). Behaviour switches get distinctive names, stay where
they're defined, and get documented because they're non-obvious. Data inputs get generic
names and are forwarded everywhere — which is exactly what manufactures the ambiguity.

**Measured impact:** LLM facts written **332 → 690** (+108%), of which **514 arrived via
the parameter-concept path** — facts previously discarded in full. The `url` concept now
carries real `CONTROLS` facts and links to 68 signature slots. Graph: 3,043 nodes,
4,258 edges. 157 tests passing.

**Still open, and deliberately not done here:** 927 facts remain blocked on DocSection and
Release having no ids (D31's shortcut) — now the single largest remaining loss, and
mechanical to fix since `doc_section_id` and `release_id` already exist and are tested.

---

## D48 — DocSection and Release wired up; D31's shortcut finally closed

**Decided:** `doc_release_index.py` builds lookups from readable prose names to ids —
"Session Objects" → `doc:<chunk_id>`, "2.34.2" → `requests:2.34.2`. Doc sections are
findable by leaf heading *and* full breadcrumb, since prose uses either.

**Ambiguous headings are dropped, not guessed.** "Timeouts" is a section in both
requests and urllib3; attaching a fact to the wrong library is worse than not attaching
it. The unambiguous full breadcrumbs ("Advanced Usage > Timeouts") still resolve.

This closes D31's stated scope cut — the ids (`doc_section_id`, `release_id`) had been
built and tested back then and simply never connected to anything.

**Cumulative effect of today's resolution work:**

| stage | LLM facts written | share of extracted |
|---|---|---|
| original full run | 255 | 11% |
| + Parameter ids (D33) + self-reference (D46) | 332 | 15% |
| + parameter concepts (D47) | 690 | 30% |
| + doc sections & releases (D48) | **1,148** | **50%** |

**4.5× the facts**, with no additional LLM spend — every gain came from replaying the
same already-paid-for extraction log through better resolution.

**Graph now: 3,284 nodes / 4,713 edges, and all 8 ontology entity types are populated**
for the first time — `DocSection` (82) and `Release` (150) had zero nodes before today.
162 tests passing.

**What remains unresolved (660 facts):** names that match nothing in the corpus —
external libraries, paraphrases, and vague references. That's the irreducible tail, not
another missing bridge.

---

## D49 — Deterministic grading, and the first trustworthy benchmark numbers

**Decided:** 39 questions now carry `must_contain` / `must_contain_any` terms and are
graded by checking the answer text — no model. With the 10 mechanically-checked
out-of-scope questions, **82% of the benchmark no longer involves an LLM judge**, which
removes D45's main variance source. A negation guard prevents "requests does *not* raise
ConnectionError" from being credited for containing the term.

**Result — and the grading was hiding a lot:**

| category | n | hybrid | baseline | delta | (run 3, AI-judged) |
|---|---|---|---|---|---|
| single_hop | 15 | 73.3% | 86.7% | −13.4pt | 66.7 / 66.7 |
| two_hop | 15 | 40.0% | 60.0% | **−20.0pt** | 20.0 / 40.0 |
| three_hop | 12 | 50.0% | 58.3% | −8.3pt | 16.7 / 16.7 |
| aggregation | 8 | 25.0% | 12.5% | **+12.5pt** | 12.5 / 12.5 |
| out_of_scope | 10 | 90.0% | 70.0% | **+20.0pt** | 70.0 / 60.0 |

Three-hop went from 17% to 50–58% for *both* systems. The systems were never that bad —
the LLM judge was marking correct answers wrong. Every earlier absolute number in this
log understated both systems.

**Where the hybrid genuinely wins:** aggregation (+12.5, the T8_RELATED_BY work) and
refusing out-of-scope questions (+20, consistent across all four runs).

**Where it genuinely loses, and why — diagnosed, not guessed.** All three two-hop
questions the baseline won and the hybrid lost share one shape: *the hybrid had graph
facts but lacked the passage containing the answer.*

- `th-03`: graph returned 1 fact, not the relevant one (`ReadTimeoutError→ReadTimeout` is
  among D44's 50 unresolved wrapping facts). D40's fallback only fires on **zero** facts,
  so one irrelevant fact was enough to suppress vector search entirely.
- `th-05`: 12 graph facts, model picked the wrong one; the baseline's passage stated
  `ProxyError` plainly.
- `th-15`: answered with a raw node id (`Session.request.stream`) where the baseline gave
  a readable explanation.

12 of 15 two-hop questions routed to `GRAPH`, and a `GRAPH` route **withholds** passages
rather than adding to them.

**The architectural question this raises:** why does using the graph mean *not* using
vector search? The graph should add facts, not subtract passages. `BOTH` may be the right
default whenever the graph is consulted at all — the cost is one extra retrieval, and the
measured cost of getting it wrong is a 20-point loss.

---

## D50 — Why two-hop was failing: the chain query walked the wrong way

**Traced the flagship failure** (`th-02`: "if urllib3 raises ProtocolError, what does
requests raise?") through the pipeline instead of guessing. The graph *did* contain
`ConnectionError WRAPS_EXCEPTION ProtocolError`. The query returned two other facts and
not that one.

**Cause:** `T3_EXCEPTION_WRAP_CHAIN` traversed `-[:WRAPS_EXCEPTION*1..N]->` — outgoing
only. Edges are stored as "requests exception wraps urllib3 exception" (D44), so from
`ProtocolError`'s side that edge points *inward*. Both directions are natural questions
and the template supported only the rarer one:

| question | direction | supported? |
|---|---|---|
| "what does requests raise when urllib3 raises X?" | what **wraps** X (incoming) | ✗ — the flagship shape |
| "what does ConnectionError ultimately wrap?" | what X wraps (outgoing) | ✓ |

`T1_NEIGHBORS` and `T8_RELATED_BY` had already been made bidirectional; the chain
templates were never updated to match.

**Fixed, and immediately broke something else — caught by reading the output.** Making
traversal undirected meant chain *order* no longer implies edge *direction*, so
`_verbalize_chain` began emitting "ProtocolError wraps ConnectionError" — the exact
reverse of the truth, and the same class of false-statement bug as D41, reintroduced by
my own fix in a different template.

Properly fixed: the chain templates now also return
`[r IN relationships(p) | startNode(r).id] AS starts`, and the verbalizer orders each hop
by its real source rather than by position in the walk. Verified against the live graph —
all eleven statements now read in the correct direction, including
`requests.exceptions.ConnectionError wraps urllib3.exceptions.ProtocolError`.

**Worth noting as a pattern:** this is the second time making a query bidirectional has
silently produced reversed facts. Any template that matches both directions must return
direction *and* have a verbalizer that uses it — the two changes are not separable, and
splitting them produces confident falsehoods rather than errors.

171 tests passing.

---

## D51 — The benchmark was measuring noise, not architecture

Two benchmark runs with **no code change between them** disagreed on ~12% of verdicts.
40% of answers came back different text. Per-category deltas swung by up to 20 points:

| category | run5 Δ | run6 Δ |
|---|---|---|
| single_hop | +0.0 | +20.0 |
| three_hop | +0.0 | −16.6 |
| aggregation | +12.5 | +0.0 |

With 8–15 questions per category, a 12% flip rate is ±1–2 questions, which is ±7–13
points of pure noise. Every swing above fits inside that.

Pooled over all 60 questions the delta across four runs was −3.3, −5.0, +5.0, +3.3 — it
doesn't even hold a sign.

**The rule this establishes:** measurement resolution has to be finer than the effect
being measured. Ours was ±13pt trying to detect maybe +4pt. No amount of staring at the
table fixes that.

An earlier claim in this log — that `temperature=0` made the benchmark deterministic —
was **wrong**. OpenAI does not guarantee determinism at temperature 0; there is no seed,
and routing varies. The fix was applied and its effect was never verified.

---

## D52 — Isolating the noise: freeze the answers, re-grade them

Rather than guess which stage was random, hold one stage fixed and vary the rest.
`graphrag/eval/regrade.py` reads answers from a completed run and grades them N times.
Anything that disagrees is the grader's own doing, because nothing else moved.

**Result: the grader is clean.** 1 disagreement in 480 gradings (0.8%), on one question
that flipped `correct`/`partial`. Two reasons it held:

- 49 of 60 questions never touch a model (10 mechanical refusal checks, 39 term matches)
- the 11 judge questions score short factual claims, where the judge is stable

Side effect: run 6's recorded `single_hop` baseline of 66.7% was the unlucky draw — three
of four re-grades say 73.3%. The headline `+20.0` was really `+13.4`.

---

## D53 — The instability is in our own router and planner, not the synthesis model

Same isolation, one layer up. `graphrag/eval/retrieval_stability.py` runs retrieval N
times per question and records each layer separately — route, plan, assembled context —
without ever calling synthesis. The vector-only baseline is the control: it makes no model
call before retrieving, so it should be perfectly stable.

| system | layer | unstable / 60 |
|---|---|---|
| hybrid | route | 4 (6.7%) |
| hybrid | plan | 7 (11.7%) |
| hybrid | context | 6 (10.0%) |
| baseline | all three | **0 (0.0%)** |

The control came back perfectly stable, which rules out Postgres, the embeddings and
context assembly in one shot. One hybrid question in ten gets a **different prompt** each
run. The synthesis model was behaving reasonably; we were feeding it different inputs.

**This is a product bug, not just a measurement artifact.** What flips:

```
oos-04   {'BOTH': 3, 'REFUSE': 1}       router can't decide if it's answerable
oos-10   {'REFUSE': 3, 'BOTH': 1}
th-03    max_hops 2 (x3) vs 3 (x1)      same question, different search depth
3h-12    T5_DELEGATION_CHAIN vs T1_NEIGHBORS, entity resolved to bare "requests"
```

`oos-04`/`oos-10` are out-of-scope questions the router sometimes refuses and sometimes
answers — and out-of-scope is the one category where hybrid consistently beat baseline.
Part of that win was luck.

**Two independent noise sources, both real:**

1. *Router/planner non-determinism* (~10% of questions) — ours, fixable, and worth fixing
   regardless of the benchmark. A system that routes one question three different ways is
   worse in production too.
2. *Synthesis wording noise* (~8%) — baseline has perfectly stable retrieval and still
   flipped 5 verdicts. Inherent to the model; only averaging over runs removes it.

Fixing (1) will not remove (2). The final numbers still need repeated runs with a reported
spread, not a single run.

**The generalisable move:** when a number is untrustworthy, don't tune the system — hold
each stage fixed in turn and find which one moves. Two cheap experiments turned "the LLM
is random" into a specific, addressable defect in two of our own components.

---

## D54 — One source for what templates exist, and what the planner is told

The planner had two hand-written lists that had to agree and didn't:
`TemplateId` (what the model may return) listed six templates, `_SYSTEM_PROMPT` (what the
model is told about) described five. `T8_RELATED_BY` was returnable but undescribed — so
the template added specifically for the aggregation questions was invisible to the thing
that picks templates. Aggregation was the worst category at 12.5% for both systems.

**Fix:** a template carries its own `description`, and `graph_query` derives both the
allowed ids and the prompt from it:

```python
PLANNABLE = {name: t for name, t in sorted(TEMPLATES.items()) if t.description}
TemplateId = Literal[tuple(PLANNABLE)]
SYSTEM_PROMPT = _PREAMBLE + "\n".join(f"- {n}: {t.description}" for n, t in PLANNABLE.items()) + _CLOSING
```

A template can no longer be offered without an explanation, or described without being
offered. `description=None` means "not offered", which `T2_PATH_BETWEEN` uses because it
needs two entities and a plan carries one `entity_surface`.

The test that matters isn't "the lists match" — it's **every offered template has
parameters a plan can actually fill**. That encodes *why* T2 is excluded, so the reasoning
is enforced rather than remembered. 180 tests passing.

**Result, measured on the 8 aggregation questions rather than assumed:** T8 is now reached
(ag-05, ag-07), and ag-07 plans correctly — `T8_RELATED_BY, BaseAdapter, INHERITS_FROM`.

**But 6 of 8 still plan badly, which is a different problem:**

| q | question | planned | wrong because |
|---|---|---|---|
| ag-01 | How many exceptions inherit from RequestException? | T6, entity=None | counts the edge type corpus-wide, ignoring the named entity |
| ag-03 | Which exceptions does HTTPAdapter.send raise? | T3 + rel=RAISES | T3 takes no relationship param at all |
| ag-04 | What parameters does Session.request accept? | T7_DOCS_FOR_SYMBOL | wants HAS_PARAMETER via T8 |
| ag-05 | Which requests exceptions are also ValueErrors? | T8 + rel=RAISES | right template, wrong relationship |

ag-01 is the instructive one. The T8 description says in plain words *"prefer this over
T6_COUNT_BY_REL whenever the question names an entity"*, and the planner chose T6 anyway.

**What that means:** making an option visible is not the same as making it chosen. This
fix removed a hard blocker (T8 was unreachable); it did not fix template *selection*,
which is a separate weakness now cleanly exposed. Prompt wording alone is evidently not
enough to steer it — the next attempt should be structural (e.g. rejecting a plan that
names an entity but picks the entity-less template) and, as always, measured on these 8
before being believed.

---

## D55 — Majority vote on the router and planner (verification PENDING)

D53 traced the benchmark's noise to our own two classification calls: the router picked a
different route on 6.7% of questions, the planner a different plan on 11.7%, so one
question in ten reached synthesis with a different prompt each run.

`graphrag/retrieval/consistency.py` adds `majority_vote(call, trials=3, key=repr)` — call
it three times, return the result most agreed on. Applied to both:

```python
classify_question:  majority_vote(ask, trials=votes, key=lambda d: d.route)
plan_graph_query:   majority_vote(ask, trials=votes)
```

**The router needed a `key`.** Its `RouterDecision` carries a float confidence that differs
on every call, so voting on the whole object finds three distinct results and no majority
at all. Voting on `.route` finds the agreement that is actually there. Every field of a
`GraphQueryPlan` is discrete, so the planner votes on the whole plan.

Three trials because it is the smallest number that can produce a majority; a 1-1-1 split
falls back to the first result, so the outcome stays a function of the results rather than
of dict ordering.

**Two limits, both real:**

- *It reduces variance, it does not remove it.* A question the model genuinely coin-flips
  on stays a coin-flip. Synthesis wording noise (~8%, D53) is untouched — the final
  benchmark still needs repeated runs with a reported spread.
- *It costs latency, asymmetrically.* Three router calls and three planner calls land on
  hybrid only; the vector baseline makes none. The benchmark's latency column now compares
  a slower hybrid against an unchanged baseline, and the README must say so rather than
  quietly report the gap.

194 tests passing (was 180).

**VERIFIED — and it barely worked.** `retrieval_stability_voted.json`, same probe, 4 trials:

| layer | before vote | after vote | change |
|---|---|---|---|
| route | 6.7% | 5.0% | −1 question |
| plan | 11.7% | 10.0% | −1 question |
| context | 10.0% | **8.3%** | −1 question |
| baseline (control) | 0.0% | 0.0% | unchanged ✓ |

One question in sixty. For 3x the model calls and 3x the latency on the hybrid system
only. **This is a bad trade and the fix should probably be reverted.**

**Why the premise was wrong.** Voting assumes the model is sampling randomly among options
it mostly gets right, so repetition converges. The arithmetic says what to expect: at a
70/30 split, best-of-3 lifts consistency to 78% — modest. At a genuine 50/50 it lifts
nothing at all. The measured gain (90% → 91.7%) matches that, which means the residual
instability is concentrated on questions the model is genuinely split on, not spread
thinly across easy ones. More votes cannot fix a coin-flip.

**What the surviving instability actually is** — and none of it is randomness for its own
sake:

```
3h-05   route {'VECTOR': 2, 'BOTH': 2}          a real 50/50; voting is powerless
oos-01  route {'REFUSE': 3, 'BOTH': 1}          is this in scope? genuinely arguable
oos-04  route {'BOTH': 3, 'REFUSE': 1}
3h-08   plan  {'ERROR:ValueError': 3, 'T1_NEIGHBORS(entity_id="requests")': 1}
oos-04  plan  {'ERROR:ValueError': 3, '-': 1}
th-08   plan  entity RetryError (x3) vs MaxRetryError (x1)
3h-12   plan  max_hops 2 (x1) vs 3 (x3)
```

Two of the six unstable plans are flips between *our entity resolver failing* and *the
resolver returning something useless* (bare `"requests"`, the whole module). That is not
model noise — it is D46/D54's resolution weakness showing up as apparent randomness.
`th-08` flipping between `RetryError` and `MaxRetryError` is the planner confusing the
requests exception with the urllib3 one it wraps, which is a comprehension failure, not a
sampling failure.

**The lesson, and it cost 3x on two components to learn:** before reaching for a
variance-reduction technique, check whether the variance is random or structural. Voting,
temperature, seeds and retries all attack random variance. Here the variance was a
*symptom* of ambiguous questions and a weak resolver, and treating the symptom moved the
number by 1.7 points. The earlier `temperature=0` fix (D51) failed for the same reason and
was not verified either; this one was at least measured before being believed.

**Recommended next, in preference order:**

1. **Revert the vote** unless a cheaper variant justifies itself — it buys 1.7pt for 3x
   cost, and it distorts the benchmark's latency column asymmetrically.
2. **For a reproducible benchmark, cache (question -> route, plan)** instead. That makes
   runs comparable for free and is honest as long as the README says retrieval decisions
   were pinned.
3. **Treat the remaining instability as the quality bug it is** — the resolver failures on
   `3h-08`/`oos-04` and the `RetryError`/`MaxRetryError` confusion on `th-08` are worth
   fixing on their own merits, and would reduce variance as a side effect.

Synthesis wording noise (~8%, D53) is untouched by any of this. The final benchmark still
needs repeated runs with a reported spread.

(Run it with `python -u` next time; stdout buffers when not attached to a terminal, so the
background log stayed empty and progress was invisible until it exited.)

---

## D56 — Why hybrid lost to baseline: the GRAPH route withheld passages

**Question asked:** what makes hybrid underperform the baseline, and fix it.

**Evidence, before any change.** In run 6, six questions were baseline-right / hybrid-wrong.
Five of the six were route `GRAPH` with graph facts present (11–38 facts). On that route the
pipeline handed the answer model *only* the verbalized triples and **no passages**, while the
baseline had five. Across all 16 questions that hit this path: hybrid 5 correct, baseline 7.

Reconstructing those questions showed the facts were mostly irrelevant — the planner had
chosen the wrong entity or template — so the model said "not enough information" with
36 facts in front of it. Controlled test on 3h-10, one variable changed: same facts alone →
*partial*; same facts plus passages → *correct*. Four of the five losses become correct with
passages present.

This is the untested hypothesis from D49, now tested.

**Fix:** `answer_hybrid` fetches passages on every non-refused route. Graph facts are
additive, never a substitute. `fell_back_to_vector` is gone — there is nothing to fall back
from. The stability probe carried the old rule inline and had already drifted; corrected.
195 tests passing.

**What this makes the benchmark mean.** Hybrid is now exactly *baseline + facts*, so the
comparison measures whether facts help on top of passages. That is the fair question, and
the honest expectation is that this stops hybrid *losing* rather than making it *win* — on
the five losses, passages alone were already enough.

**The one loss this does not fix (th-03) is a data problem, logged here for the next
thread.** The question asks what `ReadTimeoutError` becomes in requests. Two things are true:
`ReadTimeout` in `HTTPAdapter.send`, `ConnectionError` in `Response.iter_content`. The graph
holds only the second. Presented as a GRAPH FACT it overrides the correct passage — facts +
passages still answered `ConnectionError`. The missing edge is produced by

```python
except (_SSLError, _HTTPError) as e:
    ...
    elif isinstance(e, ReadTimeoutError):
        raise ReadTimeout(e, request=request)
```

and neither extractor sees through the `isinstance` narrowing: the LLM pass recorded
`ReadTimeout wraps HTTPError` (the tuple element), and `extract_exception_wrapping` would
record the same — it pairs the *caught* surface with the raise, not the narrowed one. All 77
WRAPS_EXCEPTION edges in the graph are LLM-extracted; the AST extractor is never called.
Fix is deterministic (treat `isinstance(e, X)` inside a handler as catching X), then wire the
extractor in and rebuild WRAPS edges. Separate thread; needs re-ingestion.

**Generalisable:** a *wrong* fact is worse than no fact, because the prompt labels it as
fact. The graph's value depends on precision more than recall; an incomplete graph that
confidently answers the wrong code path loses to a passage that happens to contain the right
one.

**Run 7 result.** On the 16 questions that had answered from facts alone:

| | hybrid | baseline |
|---|---|---|
| run 6 (facts only) | 5 | 7 |
| run 7 (facts + passages) | **9** | 7 |

The four that flipped are exactly the four the controlled test predicted (th-05, th-15,
3h-04, 3h-10). The one loss left in the whole benchmark is th-03 — the missing
`ReadTimeout` edge, also predicted. Prediction and outcome matched one for one, which is
the standard a fix should meet before it is believed.

Pooled over 60: hybrid 34 → 39, baseline 32 → 30, delta +2 → **+9**. Per category,
three_hop +25 and two_hop +13 — but D51's per-category noise is ±13, so only the pooled
number and the targeted 16 carry weight from a single run. Ten questions are now
hybrid-right / baseline-wrong, six of them multi-hop (th-01, th-02, th-04, 3h-08, 3h-09,
3h-11): the first evidence that facts *add* to passages rather than merely not hurting.
One run; needs the repeat-runs-with-spread treatment before it goes in a README.

The route is still called `GRAPH` in the results but no longer means graph-only. It now
means "graph consulted"; every route except REFUSE retrieves passages.

---

## D57 — The AST wrapping extractor: fixed, wired in, and the graph rebuilt from it

D56 left one benchmark loss (th-03) to a missing edge: the graph had `ConnectionError wraps
ReadTimeoutError` (true, `iter_content`) and not `ReadTimeout wraps ReadTimeoutError` (true,
`HTTPAdapter.send`). The deterministic extractor that would have found it was never called,
and would have got it wrong anyway.

**What was wrong with `extract_exception_wrapping`**, measured over the whole corpus:

1. *It paired every caught type with every raise in the handler.* One handler in `send` —
   `except (_SSLError, _HTTPError)` with three `isinstance`-guarded raises — became six
   edges, four of them false (`ReadTimeout wraps _SSLError`). 11 handlers cross-multiplied.
2. *It never read the `isinstance` guard.* `except (A, B) as e: if isinstance(e, C): raise D`
   recorded D wraps A and D wraps B, never D wraps C. And `isinstance(e.reason, X)` — how
   `send` narrows a `MaxRetryError` to the urllib3 error it carries — was invisible too.
3. *Variables leaked in as exception names.* `raise new_e` (six times in urllib3), `raise e`,
   `raise reraise` produced surfaces called `new_e`, `e`, `reraise`. Same bug in RAISES.

**The fix, in `ast_extract.py`:** a walk down the handler body that carries the narrowed
type into the branch an `isinstance` guard protects, and the handler's own caught set
everywhere else. Negated or compound tests (`not isinstance`, `isinstance(...) and retry`)
do not narrow — they say nothing certain about the raise's path. A raised variable is
followed to a direct `name = Class(...)` assignment in scope, or dropped. Lowercase bare
identifiers are never exception surfaces.

Corpus-wide: 88 edges → 81, junk surfaces 7 → 0 in WRAPS and 7 → 0 in RAISES. `send()` now
yields 13 edges, every one correct, including `ReadTimeout wraps ReadTimeoutError` and
`ConnectTimeout wraps ConnectTimeoutError` (through `.reason`).

**Wiring:** `run_ast_pass` collects `exception_wrapping`; `run_full_ingestion` resolves both
endpoints in the handler's module (so `_SSLError` resolves through its import alias) and
writes `raised WRAPS_EXCEPTION caught` with `source="ast"` and `in_function`. `merge_edge`
grew `**properties` for that. `infer_external_type` maps WRAPS_EXCEPTION endpoints to
Exception. A new `--ast-only` flag re-applies the AST pass to the live graph with no LLM
calls and without opening the LLM edge log for writing — which `--limit 0` would have
truncated, destroying the replay source.

**Result in the graph:** WRAPS_EXCEPTION went 77 (all LLM) → 85: 35 `ast`, 50 `llm`. 27 of
the 35 are edges the LLM had also found — MERGE matched them on the shared chunk id (a
docstring's chunk id *is* its function's) and relabelled them `ast`. So `source="ast"` now
means "parser-proven, whether or not the LLM agreed" and `source="llm"` means "LLM only".
8 edges are net new.

**th-03, end to end:** T3 from `ReadTimeoutError` now returns both true wraps. Facts-only →
`ReadTimeout` (was `ConnectionError`); facts + passages → correct; baseline → correct. The
last benchmark loss is gone. 205 tests passing.

**Two things left on the table, deliberately:**

- *43 of 81 AST wrap facts don't resolve* — `OSError` ×11, `ValueError` ×9, `ssl.SSLError`
  ×8, `OpenSSL.SSL.*`, `idna`, `socks`. Builtins and third-party names have no node and no
  resolver rung. RAISES has the same gap. A "builtin exception" rung would recover most of it;
  separate decision, since it means minting external nodes without an import to anchor them.
- *The 50 LLM-only edges include known noise* — `ReadTimeout wraps urllib3.util.timeout` (a
  module), `ReadTimeout wraps Timeout` (inheritance, not wrapping). They are now
  distinguishable by `source`; dropping LLM WRAPS edges whose endpoints aren't Exception-typed
  is the obvious next filter, and it should be measured rather than assumed to help.

**Generalisable:** the extractor's first version optimised recall ("pair everything") and
produced confident falsehoods. Reading one more level of control flow — the guard between
catch and raise — cost ~60 lines and turned it into something precise enough to write into a
graph that an answer model will trust. PyCG's stance (99% precision, 70% recall) is the right
target for a graph that feeds a model; the research note in `docs/research/` has the cites.

---

## D58 — CALLS edges come from jedi now; the resolver-based CALLS write is retired

The research note (docs/research/) put it plainly: every tool with a usable call graph
resolves receiver types, and this project resolved only `self`. Its recommendation was to
*consume* a real analysis rather than build one. Three tools were tried:

- **PyCG** (the paper's 99% precision / 70% recall) — archived, Python 3.6-era. Three
  incompatibilities in a row on 3.14 (import hook, `ast.Num`, positional-only args).
  Stopped after the third, per the three-strikes rule.
- **scip-python** (Sourcegraph's Pyright fork) — installs, then dies at startup on Windows:
  `new RegExp(path.sep)` with `\`.
- **jedi** — maintained, pure Python, pip-installable, runs. Resolves `conn.urlopen` in
  `HTTPAdapter.send` to `urllib3.connectionpool.HTTPConnectionPool.urlopen` on the first try.

**Measured before adopting.** All 2,646 call sites, jedi vs the resolver, in-corpus targets:

| | sites | share |
|---|---|---|
| both agree | 629 | 23.8% |
| jedi only | 229 | 8.7% |
| resolver only | 169 | 6.4% |
| conflict | 12 | 0.5% |
| neither (builtins, stdlib, unknowable) | 1,607 | 60.7% |

The "resolver only" bucket is mostly false: `getattr` ×34 → `LookupDict.__getattr__`,
`bool` ×23, `cast` ×18 — builtins fuzzy-matched to unrelated dunders by the normalized
rung. Those edges were in the live graph. The conflicts favour jedi: `tell` and
`idna_encode` are local closures jedi sees and the resolver mis-pointed at other classes;
`parse_url` is definition-site (`urllib3.util.url`) vs re-export (`urllib3.util`), and the
node universe holds definition sites. The "jedi only" bucket is attribute chains (97),
inherited `self.*` methods (69), closures (42), `super()` (21) — all real.

**A bug found on the way.** `Resolver.resolve(..., enclosing_class=)` — the D46
self-reference rung — was defined and **passed by nobody**. The ingestion CALLS loop never
gave it the class, so no `self.*` call had ever resolved in the live graph. D46's 30.7%
was measured somewhere that did pass it. This is the architecture review's "the resolver
is configured differently in five files" made concrete; the first head-to-head was run
against that broken baseline and over-credited jedi by ~200 sites until it was caught.

**Also on the way: `goto` vs `infer`.** The module first used `script.goto`, which answers
"where is this name bound". For `target = f if flag else A; target()` that is the two
`target = …` lines — one name, so the ambiguity vanished and a local variable was recorded
as a call target. `infer` answers "what does it evaluate to": f and A, two callables,
correctly ambiguous and correctly skipped. Caught by a test written for exactly that case.

**Design:** `graphrag/ingest/jedi_calls.py` — one jedi project per repo (jedi names
definitions relative to the project root; a urllib3 file under requests' root comes back
as `urllib3_repo.src.urllib3.…`), `infer` at the last character of each call target,
keep only targets that are a single function or class inside the corpus packages, one
edge per (function, target). Edges carry the caller's real chunk id — citable, where the
resolver-era edges carried the placeholder `"ast"` — plus `source="jedi"` and the
surface as written. 12 targets jedi finds that `iter_symbols` never yields (defs under
`if sys.platform…` / `if TYPE_CHECKING:`) are skipped and counted rather than written as
unlabeled nodes.

`extract_calls` still runs — replay's call_graph rung reads it — but produces no edges.
RAISES, INHERITS_FROM and WRAPS_EXCEPTION stay on the resolver: their surfaces are class
names, which name matching handles, and changing one path at a time keeps the benchmark
attributable.

**Result in the graph:** 691 resolver-era CALLS edges deleted (`chunk_id='ast'`), 679 jedi
edges written; 215 are `self.*`, 15 cross requests → urllib3. `HTTPAdapter.send` has 16,
all correct on inspection, including `conn.urlopen → HTTPConnectionPool.urlopen` and
`TimeoutSauce → urllib3.util.timeout.Timeout`. 12 random edges spot-checked: 12 correct.
`ast_edges_unresolved` 2,241 → 214, because unresolvable builtins no longer count as
failures. Ingestion adds ~12s. 211 tests passing. `jedi` added to requirements.

**The trade, stated:** "I built the resolver" is a smaller story now — it resolves class
names and parameters, not calls. What it buys is the ability to ask "does a better call
graph change accuracy?", which the resolver could never answer 5% at a time.

**Run 8 result.** Pooled: hybrid 36, baseline 32, delta **+4** (run 7: 39 / 30, +9). Nine
hybrid verdicts flipped between the runs — 3 up, 6 down — which is the ~12% flip rate D51
measured for *identical* code. Per category, two_hop +26.6 and single_hop −6.7, both
inside the ±13 per-category noise. **No conclusion about jedi CALLS can be drawn from this
run**, and it should not be read as "the call graph didn't help" any more than run 7's +9
meant the passages fix was worth 9 points.

What the run does establish:

- **th-03 → correct**, as D57 predicted. The one loss the passages fix couldn't reach is
  closed by the missing `ReadTimeout` edge.
- **The regressions are not a denser-graph effect.** The suspicion was that jedi's 679 edges
  drown the model: three regressed questions had 38–45 facts, and `oos-08` stopped refusing.
  Checked against the graph directly: the largest neighbourhoods are the *old* hubs —
  `concept:parameter:url` (91 edges, 0 jedi), `requests.models.Request` (79, 1 jedi), bare
  `requests` (65, 0) — and the median degree of a function with calls is 5. Only
  `HTTPConnectionPool.urlopen` is jedi-heavy (27 of 44), and it is genuinely the corpus's
  most-called function. A 45-fact context means the planner chose a hub entity (D54/D55),
  which it did before jedi and does after.
- **Two losses remain:** `sh-11` and `3h-05` (BOTH route, 38 facts — a hub plan).

**Where this leaves the measurement question.** Four runs of the current architecture (5–8)
give pooled deltas +5, +3, +9, +4. All positive, mean +5.3, spread ±3. That is the first
run of numbers whose sign holds, but it is still one architecture change per run, so
per-change attribution is impossible from these. The repeated-runs-with-spread plan
(D51) is now the only thing standing between the project and a README table.

**The next graph-quality item is not more edges — it is fewer facts per answer.** The
research note's item 5 (rank neighbours, cap them — aider's PageRank, GraphRAG's
"prioritise to fit the window") is what a 45-fact context needs, and it is measurable on
exactly the questions that regressed here.

---

## D59 — The planner: resolve first, then plan

**Why.** An audit of the planner on all 29 graph-routed benchmark questions (before any
change) found four failure patterns that together covered nearly every miss:

1. *8 surfaces the resolver couldn't map.* Four were ambiguous across packages
   (`ProxyError`, `SSLError`, `InvalidHeader`) when the question said which one it meant —
   "which **requests** exception" — and the planner threw the qualifier away. Two were
   builtins (`IOError`, `ValueError`). One was a qualified name the resolver couldn't handle
   (`HTTPAdapter.send`). One was an expression (`verify=False`).
2. *6 hub entities returning 30–96 facts.* `requests`, `urllib3`, `Session`, a 3-hop chain
   from `MaxRetryError`. The planner couldn't see that `requests` is a Module with 65 edges.
3. *Wrong template for the shape.* `ag-01` "how many exceptions inherit from
   RequestException" → corpus-wide count; `th-14`, the same shape, → the right T8. `3h-02`
   "what does `requests.get` hand off to" → T5 over DELEGATES_TO only → 0 facts.
4. *Schema looseness.* `relationship` attached to templates that don't take it.

The root cause was the order: template, entity surface and relationship chosen in one blind
call, with resolution afterwards. Every pattern above is a consequence.

**Four fixes, smallest first, each measured on the same 29 questions.**

*Fix 3 — T5 walks CALLS as well as DELEGATES_TO.* DELEGATES_TO is the LLM's sparse prose
view of hand-off; CALLS is now jedi's real chain (D58). The template returns each hop's
type and the verbalizer uses it ("calls" vs "delegates to"). `3h-02`: 0 → 25 facts.

*Fix 4 — the resolver handles qualified surfaces.* A `qualified` rung matches a dotted
surface as a whole suffix on a segment boundary. "HTTPAdapter.send" finds one id where the
bare leaf `send` tied with `Session.send` and `BaseAdapter.send`. The planner qualifies a
name exactly when the bare one would be ambiguous, so this was the case that failed most.

*Fix 2 — per-template plan schemas.* Each template gets its own pydantic model with only
the parameters it takes, built from `TEMPLATES` so the single-source discipline of D54
holds. T3 + `relationship` is now a validation error, not an ignored field.

*Fix 1 — resolve first, then plan.* Two model calls with our code between:
`extract_mentions` lists the entities the question names with the package its wording
assigns them ("urllib3's ReadTimeoutError"); `resolve_mentions` turns those into canonical
ids — using the qualifier to break a tie the resolver alone refuses to — and reads each
id's kind and degree from the graph; `plan_graph_query` picks a template and an entity
*from that list*, through a schema built per question in which `entity_id` is an enum of
exactly those ids. No candidates → no graph query → the passages carry the answer. The
model still never sees an id it wasn't handed and never writes Cypher; resolution stays a
step we own, moved before the choice instead of after.

Found on the way: pydantic serialises a discriminated union as `oneOf`, which OpenAI's
strict schema mode rejects. A plain `Union` is `anyOf` and works; each member's `Literal`
template_id still lets pydantic validate against exactly one template. Hop limits are an
int enum, not a range, for the same reason.

**Result, same 29 questions:**

| | before | after |
|---|---|---|
| resolution errors | 8 | **0** (3 name no entity → no graph query) |
| plans returning ≥ 30 facts | 6 | **1** (a genuine 2-hop call chain from `HTTPAdapter.send`) |
| aggregation on the right template | 2 / 8 | **4 / 8** (`ag-01`, `ag-03` fixed) |
| `3h-02` facts | 0 | 25 |

The three Module picks that remain (`th-05`, `th-06`, `3h-09`) are questions that name no
class at all — "which requests exception surfaces when a proxy fails" — where the only
resolvable mention was the package name. The old planner answered those by *inventing*
`ProxyError` from its own knowledge, which happened to be useful; the mention prompt now
forbids that, and the baseline's passages answer them anyway. Precision, again.

**Two things it exposed, left for later:**

- The planner picks chain templates for Parameter nodes, which have no call or wrap
  edges (`th-15`, `3h-01`, `3h-11` → 0 facts). One line of prompt guidance added;
  structural exclusion if that proves insufficient.
- `HAS_PARAMETER` is in the ontology and offered to the planner, but ingestion writes
  `Parameter -DEFINED_IN-> Function` and never `HAS_PARAMETER` (`3h-07` → 0 facts). An
  ontology/ingestion mismatch: either write it or stop offering it.

The retrieval sequence now lives in three places (pipeline, stability probe, the audit
script) and had to be edited in each. Candidate #1 of the architecture review is overdue.

224 tests passing.

**Run 9 result.** Pooled: hybrid 36, baseline 33, **+3** (run 8: 36 / 32, +4; run 7: 39 /
30, +9). Thirteen hybrid verdicts flipped, 6 up and 7 down — the D51 noise floor again, so
the pooled number says nothing about the planner. The individual flips do:

*Wins traceable to the planner:* `th-07` incorrect → correct (`InvalidHeader`, previously
an unresolvable ambiguous surface, now resolved through the qualifier). `3h-02`, `ag-03`,
`ag-08` incorrect → partial, each on the new plan (`T5` over CALLS from `requests.get`;
`T8(HTTPAdapter.send, RAISES)`).

*Losses traceable to the planner, and the lesson in them:* `th-06`, `th-15`, `3h-01` went
correct → wrong. In each, the old planner produced **no** graph facts (unresolvable
surface, or a chain template on a Parameter) so the model answered from passages alone and
got it right. The new planner produced 4–10 facts that are *true and irrelevant* — `T3`
from the `requests` Module; `T1_NEIGHBORS` of `Session.request.stream`, which lists where
the parameter is defined and which concepts mention it — and the model, told these are
GRAPH FACTS, leaned on them and got it wrong. **Four true facts about the wrong thing beat
five relevant passages.** This is D56's lesson one step further: it is not only *false*
facts that override passages; *irrelevant* ones do too, because the prompt labels them as
fact and the model has no way to weigh them.

`th-08` is the precision cost made concrete: the old planner invented `ResponseError` from
its own knowledge — the question never names it — and the resulting `T3` chain was right.
The new mention prompt forbids that, so the question gets no graph query and its passages
alone were not enough this run.

**What this means for the next step.** The planner now chooses well among the entities it
is given; the remaining damage comes from what the graph returns being trusted regardless
of relevance. Two levers, both on the answer side, not the planner: rank and cap facts by
relevance to the question (the research note's item 5), and stop presenting graph facts as
unconditionally authoritative in the synthesis prompt — "structural relationships, use
where relevant, prefer the passages when they conflict". Both are measurable on exactly
`th-06`, `th-15`, `3h-01`. Neither should be done before the repeat-runs-with-spread
measurement (D51) that everything since D56 has been waiting on.

---

## D60 — Five runs, identical code: the first number with error bars

Everything since D51 has been waiting on this. Five benchmark runs on commit `ae661ef`,
nothing changed between them, scored by the same rule (only "correct" counts).
`graphrag/eval/summarize_runs.py` produces the table from the five result files.

| category | n | hybrid | baseline | delta |
|---|---|---|---|---|
| single_hop | 15 | 81.3 [73.3 .. 93.3] | 77.3 [73.3 .. 80.0] | +4.0 [−6.7 .. 20.0] |
| two_hop | 15 | 65.3 [60.0 .. 73.3] | 46.7 [46.7 .. 46.7] | **+18.6 [13.3 .. 26.6]** |
| three_hop | 12 | 43.4 [41.7 .. 50.0] | 53.3 [50.0 .. 58.3] | **−10.0 [−16.6 .. −8.3]** |
| aggregation | 8 | 12.5 [12.5 .. 12.5] | 12.5 [12.5 .. 12.5] | +0.0 |
| out_of_scope | 10 | 86.0 [80.0 .. 90.0] | 62.0 [60.0 .. 70.0] | **+24.0 [20.0 .. 30.0]** |
| **pooled** | 60 | 61.3 [58.3 .. 63.3] | 53.7 [51.7 .. 55.0] | **+7.7 [5.0 .. 8.4]** |

**What is now supportable, because the ranges don't overlap:**

- *Pooled, hybrid beats the vector baseline by ~8 points, never less than 5.* The hybrid
  range [58.3 .. 63.3] sits entirely above the baseline range [51.7 .. 55.0].
- *On two-hop questions the graph is worth ~19 points.* Baseline is flat at 46.7 in all
  five runs; hybrid is 60–73. This is the multi-hop claim the project was built to make,
  and it holds for two hops.
- *Hybrid refuses out-of-scope questions better, by ~24 points.* Consistent in every run
  since run 3.

**What is supportable and unflattering:**

- *On three-hop questions hybrid loses, by ~10 points, in every run.* The whole gap is one
  question. `3h-01` ("what does `verify=False` do, and what does that mean downstream")
  is hybrid 0/5, baseline 5/5: the planner resolves `verify` to the Parameter node, `T1`
  returns 10 facts about where the parameter is defined and which concepts mention it —
  all true, none relevant — and the model, told they are facts, answers from them. Every
  other three-hop question is tied. Hybrid also records 22 *partial* verdicts on three-hop
  across the runs to baseline's 8: it gets part of the chain far more often, and partial
  scores zero. D59's "irrelevant facts override passages" is the mechanism, and it is
  deterministic, not noise.
- *Aggregation is dead: 1/8 for both systems, zero variance.* The questions need list and
  count operations neither system performs. Not noise — a capability gap.
- *Single-hop is noise, as it should be*: +4.0 with a range that crosses zero. The graph
  is not expected to help on a one-fact lookup.

**Stability, which is the other thing five runs buy:**

- Hybrid is always right on 33 questions, always wrong on 19, flips on 8. Baseline: 31 /
  24 / 5. Five of hybrid's eight flips are shared with the baseline (`sh-01`, `sh-11`,
  `sh-14`, `3h-11`, `oos-04`) — synthesis and router noise, nothing to do with the graph.
  Hybrid's own three (`th-06`, `th-09`, `th-15`) are D59's irrelevant-fact cases.
- Graph fact counts are identical across all five runs on 58 of 60 questions. The
  retrieval instability D53 measured at 10% is gone — resolve-first planning (D59) did
  what majority voting (D55) could not, because the instability was structural.

**The headline, stated the way the README should state it:** the hybrid system beats
plain vector retrieval by 5–8 points overall, by 13–27 points on two-hop questions and
20–30 on out-of-scope refusal, and *loses* 8–17 points on three-hop questions because true
graph facts about the wrong entity override correct passages. That last clause is the most
useful sentence in the project.

**Next, in order, each now measurable against these bars:** stop presenting graph facts as
unconditionally authoritative (synthesis prompt), then rank and cap them by relevance
(research note item 5). Both target `3h-01` and the 22 partials directly. The voting in
D55 can be removed — its 1.7-point effect was on an instability that no longer exists.

---

## D61 — Experiment 1 (dedupe + "relationships, not facts"): a confound, and a regression

Five runs on `2a3cbbc` against D60's five. The graph-side target (`3h-01`, 22 three-hop
partials) barely moved: `3h-01` 0/5 → 1/5 correct with 4 partial; partials 22 → 20.

**What moved instead was the baseline** — which never sees a graph fact:

| | D60 baseline | exp1 baseline |
|---|---|---|
| two_hop | 46.7 [46.7 .. 46.7] | 60.0 [53.3 .. 66.7] |
| out_of_scope | 62.0 [60.0 .. 70.0] | **42.0 [40.0 .. 50.0]** |

`synthesize.py` is shared by both systems. Rewriting its prompt changed how the model
answers *everything*: it hedges less and asserts more. On in-scope questions the graders
reward that (`th-01` baseline 0 → 5, `3h-08` both systems 0 → 5). On out-of-scope
questions it is a real regression: `oos-01`, `oos-07`, `oos-10` went from refused 5/5 to
answered — `oos-01`'s new answer is a Django tutorial written from the model's own
knowledge. The rewrite dropped the original first line ("using only the facts and
passages in the context") and buried "say so plainly instead of guessing" under the new
paragraph. Hybrid's out-of-scope score held (86 → 88) only because the router's REFUSE
route answers most of those questions before synthesis runs.

**The method error, stated so it isn't repeated:** a change to the shared synthesis prompt
is not a graph-side change, and the hybrid-minus-baseline delta cannot measure it. For
such a change each system has to be compared against *its own* previous runs, and even
then a grader that rewards assertion confounds "better" with "more assertive". Exp1's
pooled +8.3 [5.0 .. 15.0] against D60's +7.7 [5.0 .. 8.4] is not evidence of anything: the
baseline's out-of-scope collapse alone inflates the delta by ~3 points pooled.

**Kept:** the statement dedupe (hybrid-only, cannot hurt) and the relationships paragraph
(the mechanism it targets is real and documented in D60). **Restored:** the original
refusal wording, verbatim, at the top of the prompt — the part the rewrite weakened.
Exp1b re-measures that. Both systems' out-of-scope must return to D60's bars before
anything else in the table is read.

---

## D62 — Experiment 1b: the leak is closed, and the reframe did nothing

Five runs on `06f5439` (dedupe + relationships paragraph + original refusal wording).

**Sanity check first, as required:** baseline out_of_scope 62.0 [60.0 .. 70.0] — identical
to D60. Hybrid 86.0, identical. The refusal wording was the whole leak.

**The graph-side target did not move.** `3h-01`: still 0/5. Three-hop partials: 22 → 19.
Hybrid pooled 62.0 [58.3 .. 65.0] against D60's 61.3 [58.3 .. 63.3]. Nothing separable
from noise.

**What that says:** deduplicating identical sentences and telling the model, in general,
that graph lines are "relationships, not facts" does not change how it reads a specific
misleading one. "verify is defined in Session.request" still becomes "Session.request
applies verify". A generic disclaimer is not a definition. The next attempt says what
`DEFINED_IN` *means* — location, not behaviour — next to the line itself, and phrases a
parameter's edge as "is a parameter of" so the sentence cannot be misread in the first
place. Both are graph-side; the shared prompt is not touched again.

**A smaller thing the run shows:** even with the refusal wording restored, the baseline's
pooled range moved from [51.7 .. 55.0] to [55.0 .. 56.7]. The relationships paragraph sits
in the baseline's prompt too, and its presence alone nudges answers. The shared prompt is
sensitive to wording that has nothing to do with the baseline's own context; every word
in it is a variable for both systems.

Kept: the dedupe (cannot hurt) and the paragraph (harmless, and the legend builds on it).

---

## Open questions for the Phase 2 sweep

All of these are recall@k questions. None should be settled by argument.

| Question | Options |
|---|---|
| `max_chars` for prose | 1000 / 2000 / 4000 |
| what to embed for code | signature + docstring / docstring alone / first paragraph of docstring |
| the 33 overflowing chunks | truncate deliberately / drop signatures / accept |
| undocumented symbols | embed full body / embed signature alone |
| `ef_search` for HNSW | sweep |
| the 33 overflowing chunks | revisit only if recall@k shows they're actually hurting |
| ~~`ef_search`~~ | tuned — see D22, no effect at this corpus size, kept at default |
