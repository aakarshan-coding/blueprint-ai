# code-graph-rag (cgr): how it resolves names, and what that means for this project

*Research note, 2026-09-26. Read from primary sources only: the GitHub repository
`vitali87/code-graph-rag` at commit `636130c51b55f585f4677cb99fa16baa66affe07`
(version 0.0.996, committed 2026-09-26) and the docs it generates (docs.code-graph-rag.com is built
from `docs/` in that same repo via mkdocs). Every claim cites a file and line range at that commit;
"could not verify" is said where the sources do not settle a point. Source citations use the form
`[Sn]`, expanded in section 11. Line numbers are at the pinned commit.*

---

## 1. Summary

1. cgr is a 13-language tree-sitter indexer that writes a Memgraph/Neo4j graph with 26 node labels and
   33 relationship types, then answers questions through an LLM agent whose main graph tool has the
   model write Cypher freely, validated afterwards by text checks plus an EXPLAIN-plan read-only gate. [S1][S2][S13]
2. Its call resolver is a 4,000-line hand-written ladder (`call_resolver.py`), not a type checker. The
   ladder is: enclosing scope, cache, imports (direct, qualified, wildcard), same module, `self` sibling
   via MRO, a series of "known-external, drop" guards, and finally a name-only trie suffix match that
   picks the closest candidate by dotted-prefix distance. [S6]
3. The last rung *guesses*. Two same-named definitions are separated by `_import_distance_fast`, and the
   edge is written anyway, but with `resolution: "heuristic"` stored on it. Consumers such as dead-code
   analysis and `rename` can filter by that label (`--min-resolution`). [S6][S7][S8]
4. A same-module definition beats a same-named class elsewhere in *two* places: it is an explicit rung
   ahead of the trie for calls, and `resolve_class_name` prefers a same-module candidate for bare class
   names (base classes, receiver types). Your latest miss is a case cgr handles by ordering, not by inference. [S6][S9]
5. Python has an opt-in Jedi frontend. It runs after definitions are registered, asks Jedi to infer only
   attribute calls and import-bound bare calls, and records a fact only when Jedi returns exactly one
   function/class. Ambiguity is deliberately recorded as "no fact"; a site Jedi proves external suppresses
   the trie so no first-party edge is fabricated. [S10][S11]
6. Type inference exists but is shallow and heuristic: annotations, `self.x = Cls()` assignments,
   return-annotation chasing, and a *name-similarity guess* for unannotated parameters
   (`param_lower == class_lower` scores 100, suffix match 90). [S12]
7. Exceptions are not modelled. There is no RAISES, CATCHES or WRAPS edge. `except_clause` is used only
   to bind the `as` name for shadowing analysis and to walk taint through handlers. [S14][S15]
8. Hybrid merge is minimal: the vector store returns node ids, the graph is then asked for those ids.
   A graph-proximity reranker exists but is explicitly *not wired in* pending a measurement. [S16][S17]
9. Evaluation is unusually thorough at the graph layer (per-edge precision/recall against `ast`, `go/ast`,
   `javac`, libclang oracles). The one end-to-end A/B they committed (django, 40 questions) shows graph
   tools and grep tie exactly on F1 (0.975) with graph costing 2.8x the input tokens. [S18][S19]
10. Verdict for the reader: cgr did not end the whack-a-mole; its resolver is the whack-a-mole, four
    thousand lines in, with a GitHub issue number next to most rungs. What it *did* do that ends the
    silent-guess problem is label every edge with how it was resolved and refuse in the compiler-backed
    tier. Adopt the label and the "external proof suppresses the guess" rule; do not adopt the trie.

---

## 2. Architecture walkthrough

### What runs, in order

`GraphUpdater.run()` [S3, L1677-L1906] is the whole ingest. The passes, from the code and its log lines:

| Step | What | Deterministic or LLM |
|---|---|---|
| Pass 1 | `identify_structure()`: Project/Package/Folder/File nodes and CONTAINS_* edges [S3, L1775-L1780] | deterministic |
| pre-Pass-2 frontends | libclang (pure mode), Roslyn, `go/types`: compiler facts loaded before definitions [S3, L1791-L1803] | deterministic (external toolchain) |
| Pass 2 | `_process_files()`: tree-sitter parse, definition nodes (Class/Function/Method/...), imports into an in-memory import map, INHERITS deferred [S3, L1805-L1806] | deterministic |
| post-Pass-2 joins | Roslyn partials, Go IMPLEMENTS, **Jedi Python frontend**, javac frontend; all join on `(file, line, byte_col, name)` keys against the spans Pass 2 recorded [S3, L1822-L1837] | deterministic |
| deferred definitions | `_resolve_deferred_definitions`: INHERITS/IMPLEMENTS re-resolved against the full registry, RETURNS/ACCEPTS from annotations [S3, L1852] | deterministic |
| Pass 3 | `_process_function_calls()`: CALLS / REFERENCES / INSTANTIATES via `CallResolver` [S3, L1868-L1869] | deterministic |
| post-Pass-3 | IMPORTS flush, OVERRIDES, endpoints, ast-grep findings, flush, orphan prune [S3, L1871-L1906] | deterministic |
| Pass 4 (opt) | embeddings: a fixed Cypher lists every Function/Method with its span, the source is embedded (UniXcoder locally or an OpenAI-compatible endpoint) and stored in Qdrant/Milvus keyed by graph node id [S20, L6638-L6690][S21] | deterministic |

There is no LLM anywhere in ingest. The "Gloss" node type is an LLM-agent-written note, but it is written
at query time through an MCP tool, never by the indexer [S1, L238-L247].

### Query side

Two very different paths coexist:

- **Agentic path** (`cgr start`, MCP `query_code_graph`): a pydantic-ai orchestrator with ~13 tools
  [S22, table]. `query_graph` calls `CypherGenerator.generate()`, in which a second LLM writes Cypher from a
  schema-bearing system prompt [S13, L145-L190][S23, L410-L478]. The orchestrator is told in its tool
  description that rows are "candidates, not answers" and to verify call sites in source [S22, L80].
- **Deterministic path** (`cgr graph ...` CLI and MCP tools `resolve`, `definition`, `callers`, `callees`,
  `implementors`, `overrides`, `importers`, `tests_reaching`): fixed Cypher with parameters, sorted
  output, no model. The module docstring says why: "`query_code_graph` turns natural language into Cypher
  through an LLM, which is the wrong shape for 'go to definition' or 'find callers': those must be exact
  and repeatable." [S24, L1-L13][S25, L14-L21]

---

## 3. Ontology

Confirmed from `codebase_rag/constants/graph.py` at the pinned commit, not the docs page.

**26 node labels** (`NodeLabel` enum, [S1, L212-L252]): Project, Package, Folder, File, Module, Class,
Function, Method, Interface, Enum, Type, Union, ModuleInterface, ModuleImplementation, ExternalPackage,
ExternalModule, Resource, Section, Pattern, CodeSmell, SecurityIssue, Gloss, Parameter, Field, EnumVariant.
(The docs' "~19" is stale; the enum has 26 and a guard raises at import if any label lacks a unique-key
mapping [S1, L293-L298].)

**33 relationship types** (`RelationshipType`, [S1, L301-L350]), grouped by the `CaptureGroup` each
belongs to exactly once (a guard enforces total coverage [S1, L526-L532]):

| Capture group | Relationships | Default on? | Produced by |
|---|---|---|---|
| structure | CONTAINS_PACKAGE, CONTAINS_FOLDER, CONTAINS_FILE, CONTAINS_MODULE, CONTAINS_SECTION, DEFINES, DEFINES_METHOD | yes | parser (Pass 1/2) |
| calls | CALLS, REFERENCES, INSTANTIATES | yes | resolver (Pass 3); frontends and dynamic trace can add/confirm |
| types | INHERITS, IMPLEMENTS, IMPLEMENTS_MODULE, OVERRIDES, RETURNS, ACCEPTS | yes | parser + deferred resolution; RETURNS/ACCEPTS from annotation text |
| imports | IMPORTS, EXPORTS, EXPORTS_MODULE, DEPENDS_ON_EXTERNAL, LINKS_TO | yes | parser |
| io | READS_FROM, WRITES_TO, FLOWS_TO, EXPOSES, RESOLVES_TO | no | intra-procedural taint walk [S15] |
| findings | IMPLEMENTS_PATTERN, HAS_SMELL, HAS_VULNERABILITY | no | ast-grep rules |
| glosses | ANNOTATES, MENTIONS | no | written by an agent via MCP tool |
| parameters / fields / enum_variants | HAS_PARAMETER, OF_TYPE, HAS_FIELD, HAS_VARIANT | no | parser |

Nothing is LLM-produced except Gloss/ANNOTATES/MENTIONS, and those are agent *notes*, not extraction.
`RESOLVES_TO` is **not** a name-resolution edge; it joins a client's NETWORK `Resource` to the ENDPOINT
`Resource` its URL literal matches [S2, L152-L156]. The resolution tier lives on the CALLS edge as a
property, not as an edge type (section 5g).

Identity: `qualified_name` is the unique key for every code node [S1, L255-L291]. Duplicate definitions
of one qn in a module get an `@<start_line>` suffix and a CALLS edge to that name fans out to every
variant [S2, L253-L259][S26, L75-L99].

---

## 4. Extraction

cgr does not use tree-sitter `.scm` query files for definitions. Per language, a `LanguageSpec` lists the
node types that define functions and classes; for Python that is just `class_definition` and
`function_definition` [S2, L289]. The `queries/highlights/*.scm` files that exist are syntax highlighting
for the CLI, not extraction. Two tiny queries appear in the Python constants
(`PY_ASSIGNMENT_QUERY = "(assignment) @assignment"`, `PY_RETURN_QUERY`) and feed type inference [S14, L28-L29].

Where the Python-specific logic lives:

- `codebase_rag/constants/ast_python.py`: node-type names and Python-only tables (dunder map for
  operator dispatch, `PROPERTY_DECORATORS`, `Protocol` handling) [S14].
- `codebase_rag/parsers/import_processor.py`: `_handle_python_import_from_statement` builds the import
  map `{local_name: "<resolved module qn>.<original>"}`; wildcards are stored under a `*<module>` key;
  relative imports are resolved to project-prefixed qns [S27, L3129-L3238].
- `codebase_rag/parsers/class_ingest/parent_extraction.py`: `extract_python_superclasses` reads the
  `superclasses` field, unwraps `Generic[...]` subscripts, and resolves each base name (section 5b) [S28, L534-L580].
- `codebase_rag/parsers/py/`: `type_inference.py`, `variable_analyzer.py`, `expression_analyzer.py`,
  `ast_analyzer.py` (local-binding/shadowing analysis), `utils.py` (`resolve_class_name`) [S12][S9].
- Calls: `CallProcessor._get_call_target_name` reads the `function` field of a `call` node; the name
  string (`obj.method`, `pkg.f`, `self.x.y`) is what the resolver receives [S29, L2710].

Everything cross-file is done in memory during the run: a `FunctionRegistryTrie` of every definition qn
with a simple-name index and an "ending with" cache [S26], the import map, and `class_inheritance`. On
incremental runs these are rehydrated from the graph [S1, L744-L775].

---

## 5. Name resolution

### 5a. The tiers and their order

Entry point: `CallResolver._resolve_function_call` [S6, L1226-L1573]. Stripping the Rust/C#/JS/Dart
branches, the Python path is:

1. **Inline receiver** `(a / b).m()`; then an *untyped shadow* check: if the receiver name is bound by
   the caller itself (parameter, assignment, loop target) and has no inferred type, return `None`
   immediately, never reaching import probe, cache or trie (issue #1907) [S6, L1237-L1252].
2. **Enclosing scope** (Python LEGB through nested defs; class scope stops the walk) [S6, L1300-L1303; L840-L867].
3. **Cache** keyed `(call_name, module_qn, constructing)`, bypassed whenever local types are in play [S6, L1317-L1352].
4. **IIFE**, **super()** [S6, L1354-L1358].
5. **Method chains** `a().b()`: resolved *only* by return-type inference, "does NOT fall through to the
   trie fallback" [S6, L1360-L1377].
6. **Typed receiver lacks method** guard: receiver has a known first-party type that defines no such
   method (own or inherited) -> `None` (issue #1897) [S6, L1424-L1436].
7. **Imports** `_try_resolve_via_imports`: direct import (`f` in import map), qualified call
   (`obj.m` via local type, then via import, then module-level function), wildcard imports [S6, L1461-L1466; L1926-L1955].
8. **Same module** `f"{module_qn}.{call_name}"` in registry [S6, L1468-L1471; L2769-L2782].
9. **`self.m()` sibling via MRO** when the method is not on the class or its bases [S6, L1493-L1496; L3798-L3829].
10. **Drop guards**: bare name imported from outside the project; dotted call on a slash-path import;
    receiver with a known non-first-party type. Each writes `None` to the cache [S6, L1498-L1531].
11. **Trie suffix fallback** `_try_resolve_via_trie` [S6, L1568-L1573; L2857-L2922].

Note what is *not* in the list: the cache is consulted before imports, so the same `(name, module)`
resolves identically for every caller in a module unless local types differ. The docs' one-line summary
("scope, import, type or signature (exact), then overload matching, then name-only trie suffix and
wildcard imports (heuristic)") [S2, L103] is accurate as a grouping, but the actual ordering above has
eleven interleaved rungs plus roughly a dozen language-specific ones.

### 5b. A bare class name as a base class

`_resolve_python_base` [S28, L567-L580]:

```python
head, sep, tail = parent_name.partition(cs.SEPARATOR_DOT)
if import_map and head in import_map:
    resolved_head = import_map[head]
elif import_map:
    resolved_head = resolve_to_qn(head, module_qn)
else:
    resolved_head = f"{module_qn}.{head}"
```

So: import map first; otherwise `resolve_to_qn`, which is `resolve_class_name` [S9, L12-L31]:

```python
mapped = _import_mapped_class(...)          # import map
if mapped is not None: return mapped
return _class_in_module_or_enclosing_package(...)   # module.Name, then parent packages
    or _class_by_simple_name(...)           # registry suffix match
```

`_class_by_simple_name` prefers a candidate under the *current module's* prefix, else "the first
full-segment match" in sorted order [S9, L71-L95]. That last branch is a guess with no tie-break beyond sort order.

The edge itself is **deferred** until every file is parsed (`resolve_deferred_inherits`) and
re-resolved against the full registry [S30, L521-L606]. If the parent still resolves to nothing, cgr
does *not* drop it: `_externalize_written_base` writes an INHERITS edge to an `ExternalModule` node
named by the written base name (e.g. `object`, `Exception`) [S30, L882-L894; L972-L988]. Ambiguous
suffix matches produce no edge ("ambiguity means no edge") [S30, L836-L881].

**Except clauses are not resolved at all** — there is no code path that takes the name in `except X:`
and binds it (section 6).

### 5c. The same name defined in two packages

Three different answers depending on where the name appears:

- **Calls** reaching the trie: all candidates ending in `.name` are gathered, filtered to callable-compatible
  languages and to non-nested definitions, then the minimum of
  `(is_abstract, import_distance, qn)` wins [S6, L2897-L2919]. `_import_distance_fast` is
  `max(len(caller_parts), len(candidate_parts)) - common_prefix_len`, minus one if the candidate is in the
  caller's parent package [S6, L3920-L3942]. Ties are broken by the qn string. **The edge is emitted**,
  labelled heuristic.
- **Base classes / receiver types** via `resolve_class_name`: same-module candidate wins, else first
  sorted match [S9, L71-L95].
- **Return/parameter annotations** (RETURNS/ACCEPTS edges) via `TypeReferenceResolver`: imports, then
  module and enclosing modules, then "a unique project type of that name, preferring the nearest package;
  two equally near candidates stay unresolved rather than guessed" [S31, L300-L356].

So the same ambiguity is guessed for calls, sort-ordered for classes, and refused for annotations.
Your specific miss ("a name defined in the same module loses to a same-named class in another package")
cannot happen in cgr for calls or classes because the same-module rung runs before any cross-package
lookup [S6, L1468][S9, L59-L61], but it is ordering, not evidence.

### 5d. `self.method()` and attribute chains

- `self.m()` (two parts): `_resolve_two_part_call` first tries the local type map, where `self` is
  seeded to the enclosing class for the duration of the walk [S12, L343-L402], then
  `_try_method_on_class` looks up `Class.m` and falls back to a BFS over `class_inheritance`
  (`_resolve_inherited_method`), following package re-exports one hop at a time [S6, L3027-L3056; L3867-L3896; L398-L401][S32, L47-L70].
- If that fails, `_resolve_self_sibling_method` collects every concrete subclass's MRO hit for `m`,
  prefers non-abstract, and resolves **only when exactly one** remains [S6, L3798-L3829].
- `self.attr.m()` (three parts): `_resolve_self_attribute_call` requires `self.attr` in the local type map,
  which the `__init__`-assignment and class-annotation passes populate [S6, L3160-L3199][S12, L375-L379].
- `a.b.c.m()` with typed `a`: `_resolve_field_hop_method` walks each middle segment through recorded field
  types and "resolves ONLY when every middle segment is a known field ... never a name-only fallback" [S6, L3277-L3321].
- Chains with calls (`x.f().g()`): return-type inference only, else dropped [S6, L1360-L1377].

### 5e. Type inference

Per-caller, built in `build_local_variable_type_map` [S12, L343-L407]. Sources, in order:
parameter annotations; **untyped parameters guessed from name similarity to visible classes**
(`_calculate_match_score`: exact 100, either is a suffix of the other 90, contains scaled from 80)
[S12, L187-L245]; `self`/`cls` seed; single-pass assignment walk with `_infer_type_from_expression`
(constructor calls, method return annotations, module-level aliases, operator dunders) [S33];
`self.x = ...` in `__init__`; `@property` return annotations; class-level annotations; loop variables
from `list[T]` annotations; local aliases. `Optional[X]`/`X | None` is stripped to `X`; real unions stay
unresolved [S6, L385-L396]. There is no flow sensitivity, no inter-procedural propagation beyond
return-annotation chasing, and the parameter-name guess is exactly the kind of heuristic that produces
plausible wrong edges.

### 5f. When resolution fails

Three outcomes, by rung:

- **Dropped** (`None`): the drop guards in 5a step 10, the untyped-shadow rule, typed-receiver-lacks-method,
  chain with unknown return, ambiguous `self` sibling, Jedi-external site. Logged at debug level
  (`CALL_UNRESOLVED`) [S6, L2893-L2895]. The written name is also appended to the module's
  `unresolved_references` property so that a file added later triggers a re-parse of the waiting module
  [S1, L115-L121][S30, L544-L549].
- **Guessed**: the trie fallback, wildcard-import hit, ambiguous Go package member; `last_resolution`
  is set to `HEURISTIC` [S6, L2921; L2314; L3111; L2977].
- **Fanned out**: a name with several registered variants gets one edge per variant, labelled `OVERLOAD`
  [S29, L4288-L4306].

### 5g. Is the tier stored on the edge?

Yes, as an edge property, not an edge type. `EdgeResolution` is `exact | overload | heuristic |
trace_confirmed | dynamic` [S1, L69-L95]. The resolver keeps `self.last_resolution`, reset to `EXACT`
at the top of every call site [S29, L3535-L3543]; `_emit_rel` copies `self._resolution` onto every
CALLS/REFERENCES/INSTANTIATES edge under key `resolution` [S29, L1313-L1337]. The label is queryable
(`r.resolution` is returned by the fixed callers/callees Cypher [S34, L983-L989]), filterable in dead-code
analysis via `--min-resolution` [S7, L57-L72], and `rename` treats heuristic/overload/dynamic sites as
"ambiguous" rather than rewriting them [S8, L58-L63]. Dynamic tracing upgrades confirmed edges to
`trace_confirmed` in place [S2, L110].

### 5h. The Jedi tier (Python only, opt-in via `PYTHON_FRONTEND`)

`run_python_frontend` [S10]: stdlib `ast` collects call sites that are attribute calls or bare calls
bound by an import ("module-local bare calls are the heuristics' home turf"); `jedi.Script.infer` is
called per site with a 10 s per-file budget; a file that blows the budget contributes nothing. The
decisive rule:

```python
def _single_resolvable_target(names):
    # Exactly one inferred function/class is a usable fact; anything else
    # (ambiguity, modules, instances) is the ceiling: no fact, never a guess.
    if len(names) != 1:
        return None
```
[S10, L109-L117]. A target outside the repo goes into `external_sites`. In Pass 3,
`resolve_python_call_site` is consulted **before** the heuristic ladder: a fact wins; an external
sentinel returns `None` so the trie cannot fabricate a first-party edge; a miss falls through to the
ladder [S6, L4024-L4064][S29, L3797-L3809]. The frontend docs make the three-state contract explicit:
"'Found nothing' and 'did not look' are different facts" [S11, L107-L117], and "Omit the fact. Do not
guess." [S11, L119-L121]. The docs also admit the known limit: a frontend that knows three candidates
cannot say so [S11, L142-L145].

---

## 6. Exception flow

Not modelled. Searching the parsers for `raise_statement` / `except_clause` / `throw_statement` /
`catch_clause` finds node-type constants in `ast_python.py` and `ast_js.py` [S14, L63; L89] and three
consumers, none of which creates an edge:

- `py/ast_analyzer.py`: `except X as e` is one of the binding forms that make `e` a local (shadowing
  analysis), and a `try: import x / except ImportError: x = None` handler is recognised so the fallback
  binding does not shadow the import [S35, L140-L161; L245-L276].
- `flow_access/processor.py` (opt-in `io` group): `_walk_try` seeds each `except` block with the union
  of the pre-try and post-body taint states; this is data-flow over handlers, not an exception edge [S15, L3378-L3409].
- `import_processor.py`: `_is_conditional_import_node` marks an import nested under `if`/`try` as
  conditional, so a same-named local `def` is treated as a mutually-exclusive fallback variant rather
  than shadowed code; the `try` is read for binding semantics, not for the exception [S27, L450-L459].

There is no RAISES, no CATCHES, no wrapping, and `raise_statement` is never read. The roadmap and
language-support pages do not mention exceptions. This is the largest thing the reader's graph has that
cgr's does not.

---

## 7. Retrieval and Q&A

### The LLM writes Cypher freely; validation is after the fact

`CypherGenerator.generate` runs a pydantic-ai agent whose system prompt contains the schema, rules
("ALWAYS return specific properties with aliases", "Use ENDS WITH for qualified_name", "NEVER use
unbounded variable-length paths"), MAGE procedure catalogue for Memgraph, and worked patterns
[S23, L69-L116; L410-L478]. The output is cleaned of markdown fences, must contain `MATCH`, then passes
three regex validators: no write keywords (on literal-masked text), no unbounded `[*]`, only allow-listed
procedures [S13, L116-L190]. Execution goes through `fetch_read_only`, which EXPLAINs the query and
refuses any plan operator that is not a known read operator [S36, L1-L18; L164-L198].

Name resolution at query time is therefore delegated to the model: the prompt tells it to match short
names with `ENDS WITH '.VatManager'` [S23, L73]. There is no server-side surface-name to canonical-id
step in the agentic path.

Additional project-scope guard: `requires_project_evidence` refuses a generated query that does not
project `qualified_name` for every entity, and `scope_rows_to_project` post-filters rows by prefix;
the docstrings describe four review rounds finding bypasses [S37, L141-L291].

### Deterministic tools do have a resolver

`graph_query.resolve(target)` runs one fixed Cypher (`n.qualified_name = $qn OR ENDS WITH $suffix OR
n.name = $name`) and buckets results *exact, then dotted-suffix, then by-name*, each sorted
[S24, L222-L286][S34, L531-L537]. It returns all candidates; it does not pick. `definition`, `callers`,
`callees`, etc. take a full qn.

### Hybrid merge

`semantic_code_search`: embed the query, `search_embeddings(top_k)` returns `(node_id, score)` pairs
from Qdrant, then one Cypher `build_nodes_by_ids_query(node_ids)` fetches labels/qn for those ids
[S16, L32-L93]. That is the entire merge: vector picks, graph decorates. The `get_function_source` tool
reads the span off the graph node and slices the file [S16, L108-L169]. The orchestrator prompt calls
the system "hybrid retrieval" [S23, L169], but the hybridisation is the agent choosing between tools.

`graph_rerank.py` blends `similarity + 0.25 * in_set_proximity` (normalised count of distinct edge
types between hits) and is "DELIBERATELY NOT INTEGRATED. ... no labelled retrieval corpus exists in
this repository yet" [S17, L1-L18; L27-L34].

### Fallback when the graph returns nothing

None in code. `query_codebase_knowledge_graph` returns an empty `results` list with a summary string;
translation failure, timeout and DB error each return an empty result with a different summary
[S37, L697-L814]. Whether to retry, grep, or search semantically is left to the orchestrator LLM, which
is prompted to "cross-check suspiciously short result lists with a text search" [S22, L80].

---

## 8. Evaluation

The `evals/` harness is the most rigorous part of the project [S18]:

- L1 structure vs stdlib `ast`: P/R 1.0 on every label for `codebase_rag` [S18, L1198-L1211].
- L3 CALLS recall vs `sys.settrace` execution trace: 634/634 [S18, L1213-L1220].
- Retrieval (file-level "which files call X") vs `ast` oracle: graph P 0.846 / R 0.989 / F1 0.912 against
  grep_call F1 0.536 on `codebase_rag`; on django P 0.977 / R 0.938 / F1 0.957 [S18, L1222-L1235; L139-L151].
  Note the unit is `(caller_file, callee_simple_name)`, which cannot see a call bound to the *wrong*
  same-named symbol in the *right* file.
- Inheritance resolved-qn (Python only): 31/31; instantiation 378/378 [S18, L1276-L1298].
- Incremental vs clean re-index: CALLS F1 0.9988 over 25 neutral edits; 10/25 exact [S18, L1237-L1260].
- Per-language CALLS retrieval against compiler oracles, each with its own recall (Java 0.52, TS 0.75,
  JS 0.83, Rust 0.95, C 0.93) and a documented residual [S18, L505-L930].
- **Agentic A/B** (django, 40 questions, seed 0, same model, tools differ): graph mean F1 0.975 / exact
  0.925 / 23,777 mean input tokens / 19.6 s; grep mean F1 0.975 / exact 0.925 / 8,625 tokens / 16.2 s.
  Multihop: 0.9777 both, graph 35,205 vs 17,865 tokens [S19]. The graph tools did not change accuracy on
  this question type and cost 2-2.8x the tokens.

What is absent: any measurement of resolution *precision at the qn level* for calls (the retrieval
metric is file+simple-name), any error bars or repeated runs on the A/B, and any comparison against a
vector-only baseline (the semantic eval is a curated recall@3 fixture [S18, L1013-L1032]).

---

## 9. Incremental updates and scale

- Incremental runs re-parse changed files plus one level of dependents found through the graph's own
  edges, restore inbound edges verbatim ("re-resolving the callers instead would diverge ... because
  cgr's call resolution is context-sensitive"), and rehydrate the registry, `class_inheritance`
  (ordered by persisted `base_index`), property flags and function locations from the graph
  [S1, L785-L848][S18, L178-L242].
- Unresolved names are persisted per module so a later-added file re-triggers exactly the modules that
  waited for it [S1, L115-L121; L889-L918].
- Scoped re-ingest of one typical file: p50 194 ms; a hub imported by 54 files: 3.5 s [S38, L38-L48].
- Edges are one-per-site with `line`/`col` in the MERGE key, so re-indexing is idempotent [S2, L106].
- The trie's `find_ending_with` was 48% of CPU before a full simple-name index was added [S39].

---

## 10. Comparison: cgr vs this project

| Row | code-graph-rag | This project |
|---|---|---|
| Extraction | tree-sitter node-type lists per language; in-memory registry + import map; opt-in Jedi/javac/Roslyn/go-types/libclang facts joined by position [S3][S11] | Python `ast` for structure, jedi `infer` for calls (`graphrag/ingest/jedi_calls.py`), LLM for prose |
| Resolution tiers | 11 Python rungs: scope, cache, imports, same-module, self/MRO, drop guards, trie by import distance [S6, L1226-L1573] | 8 rungs: exact, `self.`, import alias, re-export, qualified suffix, exact bare, normalized, public+documented, call-graph tie-break, builtin (`resolve.py` L100-L172) |
| On ambiguity | calls: pick nearest by dotted-prefix distance, label `heuristic`; classes: same-module else first sorted; annotations: refuse [S6][S9][S31] | refuse, return `candidates` tuple; planner narrows by package mention (`graph_query.py` L138-L173) |
| Type inference | per-caller map: annotations, assignments, `__init__` attrs, return annotations, **parameter-name similarity** [S12] | none in resolver; jedi does inference for CALLS only |
| Exceptions | none [S14][S15] | RAISES, WRAPS_EXCEPTION, Exception nodes |
| Query authoring | LLM writes Cypher from schema prompt [S13][S23]; separate fixed-Cypher tools for exact lookups [S24] | LLM picks a template and fills validated blanks (`cypher_templates.py`) |
| Validation | post-hoc regex (write keywords, unbounded paths, procedures) + EXPLAIN plan gate + project-scope evidence check [S13][S36][S37] | pre-execution: pydantic model per template, entity ids as enum of resolved candidates |
| Hybrid merge | vector ids -> graph fetch by id; proximity reranker written but not wired [S16][S17] | shared chunk id joins Neo4j and pgvector |
| Evaluation | per-edge P/R vs compiler oracles; agentic A/B, one run, no error bars [S18][S19] | 90-question benchmark, five runs, error bars (D66) |

---

## 11. What to adopt, what not to, ranked

### Adopt

1. **A `resolution` label on every CALLS/INHERITS/RAISES edge, and a `method` on every `Resolution`.**
   You already return `Resolution.method` (`resolve.py` L32-L43); cgr's step is to write it onto the edge
   and let consumers filter [S1, L69-L95][S29, L1330-L1331]. Touches `graphrag/ingest/resolve.py`
   (rename `method` values into a rank: exact > jedi > import > suffix > heuristic) and the edge writer;
   `graphrag/retrieval/cypher_templates.py` gains an optional `min_resolution` param so a template can
   ask for exact-only edges when answering "does X call Y" and any-tier when answering "what might".
   Effort: small (one property, one filter). Payoff: a wrong heuristic edge stops being able to override
   a correct passage silently, which is the D56 failure shape.
2. **"External proof suppresses the guess."** cgr's Jedi tier records `external_sites` and returns a
   sentinel so no later rung fabricates a first-party edge [S10, L143-L147][S6, L4062-L4063]. Your
   `jedi_calls.resolve_calls` filters targets to `packages` and *drops* everything else (L125-L127), so
   a call jedi proves is `stdlib.x` looks identical to a call jedi could not resolve. Record the negative:
   when jedi returns exactly one target outside the corpus, emit a `Resolution(method="external")`
   and make `Resolver.resolve` return it before the normalized/suffix rungs. Touches
   `graphrag/ingest/jedi_calls.py` (return `CallEdge` with `target_id="external:<full_name>"` or a
   separate set) and `graphrag/ingest/resolve.py`. Effort: small. This is the single cgr mechanism
   most directly aimed at your "new edge case every run" symptom, because a large class of those cases
   are stdlib/third-party names colliding with corpus names.
3. **Deferred base-class resolution with an explicit external target.** cgr resolves INHERITS after
   all files are parsed and, when the base is not in the corpus, keeps the fact as an edge to an
   `ExternalModule` node named by the written name rather than dropping it [S30, L521-L606; L972-L988].
   For `requests`/`urllib3` this matters for `class ConnectionError(RequestException)` vs
   `class SSLError(ConnectionError)` chains that cross into `builtins`/`ssl`. You already have
   `BUILTIN_EXCEPTIONS` (`resolve.py` L25-L28); extend it to a general "external, named" resolution
   rather than `unresolved`. Touches `graphrag/ingest/resolve.py`. Effort: small.
4. **A same-module rung before any cross-package rung, for classes too.** Your ladder has no
   `module_context + "." + surface` check before `_by_exact_name` (L149-L151), which is exactly the miss
   you described. cgr's `_class_in_module_or_enclosing_package` walks module, then each enclosing
   package [S9, L56-L68]. Add it between `import_alias` and `reexport`. Effort: ten lines.
5. **Per-rung regression fixtures with a control.** cgr's tests pair each shadowing case with the same
   body minus the shadow, "so a green means the binding was honoured rather than the harness seeing
   nothing" [S40, L1-L16]. Cheap discipline that turns each whack-a-mole fix into a permanent test.
6. **The deterministic `resolve` shape at query time**: return exact, then suffix, then by-name buckets,
   never pick [S24, L222-L286]. You already do this in `resolve_mentions`; keep it. Nothing to change.

### Do not adopt

- **The trie suffix fallback with import-distance tie-break** [S6, L2857-L2922]. It is a guess with a
  distance heuristic, and the evals README concedes precision "is bounded by the trie suffix-match
  fallback occasionally resolving a module-level call to an unrelated first-party name" [S18, L58-L59].
  Your `candidates` tuple plus refusal is strictly safer for an answer model.
- **Parameter-name-to-class similarity typing** [S12, L187-L245]. `def send(request)` binding `request`
  to `Request` is right often enough to be dangerous; in `requests` the same name means `PreparedRequest`
  in half the code. This is a heuristic that produces confident wrong receiver types.
- **Free-form LLM Cypher with post-hoc validation** [S13][S37]. cgr needed an EXPLAIN gate, a
  literal-masking pass, and a 600-line project-scope guard whose docstrings record four rounds of
  bypasses. Your typed-template approach avoids the entire class.
- **Their retrieval-eval unit** `(caller_file, callee_simple_name)` [S18, L107-L110]. It cannot detect a
  call resolved to the wrong same-named symbol, which is precisely your failure mode. Keep grading at
  canonical-id level.

### Does cgr end the whack-a-mole?

No. `_resolve_function_call` cites issues #1009, #1011, #1026, #1061, #1082, #1093, #1526, #1897,
#1907, #2002 inline as the reason each rung exists [S6, L1237-L1573]; the resolver is 4,078 lines and the
call processor 8,576. What cgr changed is *where the misses land*: the compiler-backed tier refuses
rather than guesses, the heuristic tier labels its output, and consumers can choose a floor. The
honest reading is that a hand-written ladder never converges; the structural fix is (a) make the
exact tier (jedi) cover more sites and prove externals, (b) label everything else, (c) let the answer
path prefer labelled-exact edges and show the passage when only heuristic edges exist.

---

## 12. Sources

All GitHub links are pinned to commit `636130c51b55f585f4677cb99fa16baa66affe07`. Base:
`https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/`.

- [S1] `codebase_rag/constants/graph.py` — schema enums, `EdgeResolution`, capture groups, rehydration Cypher. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/constants/graph.py
- [S2] `docs/architecture/graph-schema.md` (published at https://docs.code-graph-rag.com/architecture/graph-schema/). https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/docs/architecture/graph-schema.md
- [S3] `codebase_rag/graph_updater.py` — `run()` pass order L1677-L1906; `_run_python_frontend` L1298-L1327. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/graph_updater.py
- [S6] `codebase_rag/parsers/call_resolver.py` — the ladder. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/call_resolver.py
- [S7] `codebase_rag/dead_code.py` — `resolution_at_least`, `--min-resolution`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/dead_code.py
- [S8] `codebase_rag/editing/rename.py` — heuristic/overload/dynamic sites treated as ambiguous. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/editing/rename.py
- [S9] `codebase_rag/parsers/py/utils.py` — `resolve_class_name`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/py/utils.py
- [S10] `codebase_rag/parsers/py_frontend/frontend.py` — Jedi frontend. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/py_frontend/frontend.py
- [S11] `docs/architecture/language-frontends.md` (https://docs.code-graph-rag.com/architecture/language-frontends/). https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/docs/architecture/language-frontends.md
- [S12] `codebase_rag/parsers/py/type_inference.py` (L343-L407) and `codebase_rag/parsers/py/variable_analyzer.py` (L116-L245). https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/py/type_inference.py , https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/py/variable_analyzer.py
- [S13] `codebase_rag/services/llm.py` — `CypherGenerator`, validators. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/services/llm.py
- [S14] `codebase_rag/constants/ast_python.py`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/constants/ast_python.py
- [S15] `codebase_rag/parsers/flow_access/processor.py` — `_walk_try` L3378-L3409. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/flow_access/processor.py
- [S16] `codebase_rag/tools/semantic_search.py`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/tools/semantic_search.py
- [S17] `codebase_rag/tools/graph_rerank.py`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/tools/graph_rerank.py
- [S18] `evals/README.md` — harness and "Latest results". https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/evals/README.md
- [S19] `evals/results/agentic_qa.json` and `evals/results/agentic_qa_multihop.json` (summary blocks). https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/evals/results/agentic_qa.json , https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/evals/results/agentic_qa_multihop.json
- [S20] `codebase_rag/graph_updater.py` L6630-L6690 — embedding pass (same file as S3).
- [S21] `docs/sdk/semantic-search.md` (https://docs.code-graph-rag.com/sdk/semantic-search/). https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/docs/sdk/semantic-search.md
- [S22] `docs/guide/interactive-querying.md` (https://docs.code-graph-rag.com/guide/interactive-querying/). https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/docs/guide/interactive-querying.md
- [S23] `codebase_rag/prompts.py` — Cypher system prompt. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/prompts.py
- [S24] `codebase_rag/graph_query.py` — deterministic `resolve`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/graph_query.py
- [S25] `codebase_rag/constants/mcp.py` — MCP tool names. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/constants/mcp.py
- [S26] `codebase_rag/function_registry.py` — `FunctionRegistryTrie`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/function_registry.py
- [S27] `codebase_rag/parsers/import_processor.py` — Python from-imports L3129-L3238. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/import_processor.py
- [S28] `codebase_rag/parsers/class_ingest/parent_extraction.py` — Python bases L534-L580. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/class_ingest/parent_extraction.py
- [S29] `codebase_rag/parsers/call_processor.py` — `_emit_rel` L1313-L1337; per-site reset L3535-L3543; Jedi join L3797-L3809; overload fan-out L4288-L4306. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/call_processor.py
- [S30] `codebase_rag/parsers/class_ingest/mixin.py` — `resolve_deferred_inherits` L521-L606, `_resolve_deferred_parent_qn` L783-L894, `_externalize_written_base` L972-L988. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/class_ingest/mixin.py
- [S31] `codebase_rag/parsers/type_facts.py` — `TypeReferenceResolver` L283-L368. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/type_facts.py
- [S32] `codebase_rag/parsers/utils.py` — `follow_reexports` L47-L70. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/utils.py
- [S33] `codebase_rag/parsers/py/expression_analyzer.py`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/py/expression_analyzer.py
- [S34] `codebase_rag/cypher_queries.py` — `CYPHER_GRAPH_RESOLVE_NAME` L531-L537; callers/callees with `r.resolution` L975-L989. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/cypher_queries.py
- [S35] `codebase_rag/parsers/py/ast_analyzer.py` — bindings incl. `except ... as` L140-L161, optional-import idiom L245-L276. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/parsers/py/ast_analyzer.py
- [S36] `codebase_rag/services/cypher_guard.py` — EXPLAIN plan gate. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/services/cypher_guard.py
- [S37] `codebase_rag/tools/codebase_query.py` — project-scope guard and `query_codebase_knowledge_graph`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/tools/codebase_query.py
- [S38] `docs/reports/REINGEST_BENCHMARK.md`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/docs/reports/REINGEST_BENCHMARK.md
- [S39] `docs/reports/BENCHMARK_REPORT.md` — Finding 1. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/docs/reports/BENCHMARK_REPORT.md
- [S40] `codebase_rag/tests/test_local_binding_shadows_import.py`. https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/codebase_rag/tests/test_local_binding_shadows_import.py
- README: https://github.com/vitali87/code-graph-rag/blob/636130c51b55f585f4677cb99fa16baa66affe07/README.md ("How It Works", L73-L86). Docs site root: https://docs.code-graph-rag.com/ (mkdocs `site_url`, built from `docs/`).

Reader-side files referenced: `graphrag/ingest/resolve.py` (L25-L172), `graphrag/ingest/jedi_calls.py`
(L89-L144), `graphrag/retrieval/graph_query.py` (L138-L173), `graphrag/retrieval/cypher_templates.py`.

Could not verify: whether docs.code-graph-rag.com is currently deployed from this exact commit (the
site is generated from `docs/` by mkdocs per `mkdocs.yml`, but the live build was not fetched);
the model used for the committed agentic A/B run (the JSON records corpus, commit, seed and sample
but not the model id).
