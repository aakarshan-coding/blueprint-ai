# How other tools build a knowledge graph from a codebase — and how this project compares

*Research note, 2026-09-24. Every claim about an external tool cites the primary source it was read from (official
docs, source on GitHub, a spec, or a first-party paper/blog). Where a page did not state something, that is said
explicitly. Claims about this project cite the file or `docs/DECISIONS.md` entry.*

---

## 1. Summary

1. **There are two families, and this project is a hybrid of both.** Compiler/parser tools (SCIP, CodeQL, Joern, PyCG,
   pyan, aider) never use an LLM and resolve references from code semantics. LLM tools (Microsoft GraphRAG,
   LlamaIndex, LangChain) never parse code and resolve entities by string-merging model output. This project runs
   Python's `ast` for structure and an LLM only for prose, then pushes *both* through one resolver
   (`graphrag/ingest/resolve.py`). No surveyed tool does that split.

2. **Everyone serious about call graphs resolves receiver types; this project only resolves `self`.** scip-python is a
   Pyright fork, CodeQL uses points-to plus type tracking, PyCG builds an inter-procedural assignment graph, and even
   pyan (a small AST tool) tracks `self` and simple bindings. This project's 30.7% call-surface resolution (D46) is
   the direct cost of that gap.

3. **Exception flow is a control-flow problem, and nobody in the survey solves it with a flat AST pass.** CodeQL and
   Joern carry a control-flow graph; the LLM tools don't model exceptions at all. This project's
   `isinstance`-narrowing failure (D56) is what happens when a handler/raise pairing ignores the control flow between
   them.

4. **Precision beats recall for a graph that feeds an answer model.** PyCG reports ~99.2% precision at ~69.9% recall
   and treats that as the right trade. D56 reached the same conclusion the hard way: one wrong `WRAPS_EXCEPTION` edge
   overrode a correct passage.

5. **Graph facts are always a supplement, never the whole context, in every retrieval system surveyed.** GraphRAG's
   local search fetches text units alongside entities; LlamaIndex retrievers return paths plus source nodes; aider's
   map sits beside whatever files are already in chat. D56's "GRAPH route withheld passages" bug was a departure from
   the field norm, and its fix put the project back in line.

6. **Template-only Cypher is a recognised option, not an oddity.** LlamaIndex ships both `TextToCypherRetriever`
   (model writes Cypher) and `CypherTemplateRetriever` (model fills blanks). This project sits at the template end,
   with stricter typed validation than LlamaIndex documents.

---

## 2. Per-tool sections

### 2.1 Sourcegraph SCIP (+ scip-python)

**Ontology.** SCIP is "a language-agnostic protocol for indexing source code, which can be used to power code
navigation functionality such as Go to definition, Find references, and Find implementations" ([SCIP
README](https://github.com/sourcegraph/scip)). The schema is a Protobuf file with `Index`, `Document`, `Occurrence`,
`SymbolInformation`, and `Relationship` messages
([scip.proto](https://github.com/sourcegraph/scip/blob/main/scip.proto)). There are no free-form node labels: a symbol
string is `<scheme> <manager> <package-name> <version> (<descriptor>)+`, and descriptor suffixes distinguish
`Namespace` (`/`), `Type` (`#`), `Term` (`.`), `Method` (`().`), `Parameter` (`()`), `TypeParameter` (`[]`), `Meta`
(`:`), `Macro` (`!`) and `Local` ([scip.proto](https://github.com/sourcegraph/scip/blob/main/scip.proto)). Edges are
`Relationship` records with boolean flags: `is_reference`, `is_implementation`, `is_type_definition`, `is_definition`;
`SymbolRole` is a bitset with `Definition`, `Import`, `WriteAccess`, `ReadAccess`, `Generated`, `Test`,
`ForwardDefinition` ([scip.proto](https://github.com/sourcegraph/scip/blob/main/scip.proto)).

**Extraction.** Per-language indexers built on real compilers or language servers. scip-python "is a Sourcegraph fork
of pyright focused on generating SCIP for python projects" ([scip-python
README](https://github.com/sourcegraph/scip-python)). Deterministic.

**Name resolution.** Full: Pyright's type checker does it. Cross-package resolution rides on package metadata:
scip-python "uses `pip` to attempt to determine the versions and names of the packages available in your environment,"
and `--project-namespace` puts "a namespace before all the generated symbols" for "cross repository navigation"
([scip-python README](https://github.com/sourcegraph/scip-python)). Every symbol string carries package and version,
so `requests.adapters.HTTPAdapter.send` and `urllib3` symbols are globally distinct by construction.

**Call graph / inheritance / exceptions.** SCIP models *occurrences* of symbols with roles, plus `is_implementation`
and `is_definition` relationships; it does not define a call edge type, an inheritance edge type, or anything about
exceptions ([scip.proto](https://github.com/sourcegraph/scip/blob/main/scip.proto)). Call-graph-like queries are
derived from "find references" over occurrences. Whether scip-python emits `is_implementation` for Python subclass
methods: **could not verify** from the README.

**Retrieval.** No vector search in the protocol. Sourcegraph's product: "Sourcegraph automatically uses Precise Code
Navigation whenever available, and Search-based Code Navigation is used as a fallback when precise navigation is not
available" ([Sourcegraph docs](https://sourcegraph.com/docs/code_navigation/explanations/precise_code_navigation)).
Queries are issued by the IDE/UI, not a model.

**Relevance to this project.** The `Parameter` descriptor `()` means every parameter symbol is scoped to its enclosing
function, exactly like this project's `Session.request.verify` ids (D33). SCIP never hits the "9 candidates for
`verify`" problem because it only ever resolves *code* references, which come with a scope; the problem exists here
only because prose references have no scope. That is the real source of D33/D36/D47, not weak matching.

### 2.2 GitHub CodeQL

**Ontology.** A CodeQL database is "a full, hierarchical representation of the code, including a representation of the
abstract syntax tree, the data flow graph, and the control flow graph," stored relationally; "CodeQL libraries define
classes to provide a layer of abstraction over the database tables" ([About
CodeQL](https://codeql.github.com/docs/codeql-overview/about-codeql/)). For Python, `Module`, `Class`, `Function` are
subclasses of `Scope`; `Stmt` has ~20 subclasses including `Raise`, `ExceptStmt`, `Try`; `Expr` has ~30 including
`Call`, `Attribute`, `Name` ([CodeQL library for
Python](https://codeql.github.com/docs/codeql-language-guides/codeql-library-for-python/)). Control flow is exposed as
`ControlFlowNode` and `BasicBlock` ([same
page](https://codeql.github.com/docs/codeql-language-guides/codeql-library-for-python/)).

**Extraction.** Extractors. "For interpreted languages, the extractor runs directly on the source code, resolving
dependencies to give an accurate representation of the codebase" ([About
CodeQL](https://codeql.github.com/docs/codeql-overview/about-codeql/)). Deterministic.

**Name resolution.** Two layers. The older points-to layer: a `Value` is "a static approximation to a set of one or
more real objects," with `ClassValue`, `FunctionValue`, `ModuleValue`
([ObjectAPI](https://codeql.github.com/codeql-standard-libraries/python/semmle/python/objects/ObjectAPI.qll/module.ObjectAPI.html)).
The newer data-flow layer resolves calls with type trackers: `resolveCall` "Holds if `call` is a call to the `target`,
with call-type `type`"; `viableCallable` "Gets a viable run-time target for the call `call`"; `selfTracker` "Gets a
reference to the `self` argument of a method on class `classWithMethod`"; method lookup follows an MRO approximation
(`findFunctionAccordingToMro`)
([DataFlowDispatch](https://codeql.github.com/codeql-standard-libraries/python/semmle/python/dataflow/new/internal/DataFlowDispatch.qll/module.DataFlowDispatch.html)).
The docs are candid: "Due to the dynamic nature of Python, such syntactic queries can be inaccurate" ([Functions in
Python](https://codeql.github.com/docs/codeql-language-guides/functions-in-python/)). Library calls are matched via
API graphs, e.g. `API::moduleImport("os").getMember("open").getACall()` ([Analyzing data flow in
Python](https://codeql.github.com/docs/codeql-language-guides/analyzing-data-flow-in-python/)).

**Call graph / inheritance / exceptions.** Calls: resolved as above, including `self.method()`, `cls.method()`,
classmethod/staticmethod, and class instantiation (`resolveClassCall`, `resolveClassInstanceCall`)
([DataFlowDispatch](https://codeql.github.com/codeql-standard-libraries/python/semmle/python/dataflow/new/internal/DataFlowDispatch.qll/module.DataFlowDispatch.html)).
Inheritance: `ClassValue` exists and MRO is approximated
([ObjectAPI](https://codeql.github.com/codeql-standard-libraries/python/semmle/python/objects/ObjectAPI.qll/module.ObjectAPI.html),
[DataFlowDispatch](https://codeql.github.com/codeql-standard-libraries/python/semmle/python/dataflow/new/internal/DataFlowDispatch.qll/module.DataFlowDispatch.html)).
Exceptions: the CFG guide shows "one path where an exception is raised by `might_raise()`" as a distinct control-flow
path ([Analyzing control flow in
Python](https://codeql.github.com/docs/codeql-language-guides/analyzing-control-flow-in-python/)), and
`Raise`/`ExceptStmt`/`Try` are first-class AST nodes. Whether the standard library ships a predicate that pairs a
raise inside a handler with the caught type (i.e. exception *wrapping*): **could not verify** from the pages fetched;
it is expressible as a query over `ExceptStmt` containing a `Raise`, which is what this project's
`extract_exception_wrapping` does on the AST.

**Retrieval.** None. Humans write QL; there is no vector component and no model in the loop.

**Relevance.** CodeQL's `selfTracker` is the industrial version of this project's `self_reference` rung (D46). Its
explicit CFG is what would have caught the `isinstance(e, ReadTimeoutError)` narrowing in D56: on a CFG, the `raise
ReadTimeout` sits on a path guarded by that test, so "what was caught" is a path condition, not the handler's tuple.

### 2.3 Joern Code Property Graph

**Ontology.** The CPG is "a directed, edge-labeled, attributed multigraph" organised in layers; "Different classic
program representations are merged into a property graph into a single data structure that holds information about the
program's syntax, control- and intra-procedural data-flow" ([Joern docs](https://docs.joern.io/code-property-graph/)).
Node types include `METHOD`, `METHOD_PARAMETER_IN`, `METHOD_RETURN`, `TYPE_DECL`, `TYPE`, `MEMBER`, `CALL`,
`IDENTIFIER`, `FIELD_IDENTIFIER`, `CONTROL_STRUCTURE`, `RETURN`, `LOCAL`, `BLOCK`, `LITERAL`, `NAMESPACE_BLOCK`,
`FILE`. Edge types include `AST`, `CFG`, `CDG`, `REACHING_DEF`, `CALL`, `ARGUMENT`, `RECEIVER`, `REF`, `EVAL_TYPE`,
`INHERITS_FROM`, `ALIAS_OF`, `BINDS_TO`, `DOMINATE`, `POST_DOMINATE`, `CONTAINS`, `SOURCE_FILE`. `CALL` nodes carry
`METHOD_FULL_NAME` and `DISPATCH_TYPE` ([CPG spec](https://cpg.joern.io/)).

**Extraction.** Language frontends emit the base AST layer; the spec "allows for hosting of graphs generated by 8
different language frontends" ([Joern docs](https://docs.joern.io/code-property-graph/)). Deterministic. The Python
frontend is `pysrc2cpg`; its README lists known gaps: "No named parameter support", "Incorrect instance argument for
call like x.func", "No handling of `__getattr__`, `__setattr__`, etc." ([pysrc2cpg
README](https://github.com/joernio/joern/blob/master/joern-cli/frontends/pysrc2cpg/README.md)).

**Name resolution.** Type *recovery*, not full inference: the Python frontend exposes `--type-prop-iterations`
("Maximum iterations of type propagation") and `--no-dummyTypes` ("Disables the generation of dummy types during type
propagation") ([Joern Python frontend docs](https://docs.joern.io/frontends/python/)). Calls on dynamic receivers are
modelled differently from static languages: "In dynamic languages `arg[0]` is no longer the receiver of the call, but
instead is the object that holds the property which is the receiver of the call" ([same
page](https://docs.joern.io/frontends/python/)).

**Call graph / inheritance / exceptions.** Calls: `CALL` edges with `DISPATCH_TYPE`, plus `RECEIVER`/`ARGUMENT` edges
([CPG spec](https://cpg.joern.io/)). Inheritance: `INHERITS_FROM` edges on `TYPE_DECL` ([CPG
spec](https://cpg.joern.io/)). Exceptions: `CONTROL_STRUCTURE` nodes with a `CONTROL_STRUCTURE_TYPE` property, and
`CFG` edges ([CPG spec](https://cpg.joern.io/)); a dedicated exception-wrapping relation: **could not verify**;
nothing in the spec names one.

**Retrieval.** None. Queries are Scala DSL written by humans. No vectors.

**Relevance.** The CPG's `REACHING_DEF` and `CFG` edges are what "which exception reaches this `raise`" needs. This
project stores no intra-function structure at all — a function is one node and its raises are one list (D10) — so the
information required to fix D56 is discarded at extraction time.

### 2.4 Aider repo map

**Ontology.** Two kinds of "tags" from tree-sitter queries: definitions (`name.definition.*`) and references
(`name.reference.*`); a graph whose nodes are *files* and whose edges connect a referencing file to a defining file
per identifier ([repomap.py](https://raw.githubusercontent.com/Aider-AI/aider/main/aider/repomap.py)). "Using the AST,
we can identify where functions, classes, variables, types and other definitions occur in the source code. We can also
identify where else in the code these things are used or referenced" ([aider
blog](https://aider.chat/2023/10/22/repomap.html)).

**Extraction.** tree-sitter, deterministic; "The tree-sitter repository map replaces the ctags based map that aider
originally used" ([aider blog](https://aider.chat/2023/10/22/repomap.html)). Languages without reference queries fall
back: "Use pygments to backfill refs"
([repomap.py](https://raw.githubusercontent.com/Aider-AI/aider/main/aider/repomap.py)).

**Name resolution.** None beyond identifier-string equality: a reference to `send` links to *every* file that defines
something named `send`. The code compensates with weights rather than resolution: `if ident in mentioned_idents: mul
*= 10`; `if (is_snake or is_kebab or is_camel) and len(ident) >= 8: mul *= 10`; `if ident.startswith("_"): mul *=
0.1`; `if len(defines[ident]) > 5: mul *= 0.1`; references from files in chat get `use_mul *= 50`; counts are dampened
with `num_refs = math.sqrt(num_refs)`
([repomap.py](https://raw.githubusercontent.com/Aider-AI/aider/main/aider/repomap.py)). Note the `len(defines[ident])
> 5` rule: a name defined in more than five places is *penalised*, which is the same ambiguity this project hit with
`verify` (9 candidates, D33), handled by demotion instead of resolution.

**Call graph / inheritance / exceptions.** None of the three is modelled as such. Everything is "reference" edges
between files.

**Retrieval.** `nx.pagerank` with a personalisation vector boosting chat files and mentioned identifiers, rank
distributed over out-edges to `(file, identifier)` pairs, then binary search to fit a token budget
([repomap.py](https://raw.githubusercontent.com/Aider-AI/aider/main/aider/repomap.py)). Budget defaults to
`--map-tokens` = 1k and "adjusts the size of the repo map dynamically based on the state of the chat" ([aider
docs](https://aider.chat/docs/repomap.html)). No vector search. No model writes a query; the map is computed and
prepended.

**Relevance.** Aider proves that a ranked, budgeted *subset* of the graph is more useful to a model than an unranked
dump. This project's `T1_NEIGHBORS` returns every neighbour unordered (`cypher_templates.py`), and D56 found the
answer model drowning in 11–38 mostly-irrelevant facts.

### 2.5 Microsoft GraphRAG

**Ontology.** Entities, relationships, claims ("covariates"), text units, communities, community reports ([indexing
overview](https://microsoft.github.io/graphrag/index/overview/), [default
dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/)). Entity and relationship *types* are whatever
the LLM emits, guided by the prompt; there is no fixed code ontology.

**Extraction.** LLM over every text unit: the pipeline exists to "extract entities, relationships and claims from raw
text" ([indexing overview](https://microsoft.github.io/graphrag/index/overview/)). Inferred, non-deterministic.

**Name resolution.** String merge: "any entities with the same _title_ and _type_ are merged by creating an array of
their descriptions," which are then summarised "into a single description per entity and relationship" ([default
dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/)). No scoping, no alias table.

**Call graph / inheritance / exceptions.** Not modelled; GraphRAG is not a code tool. A code call would only appear if
the LLM chose to emit a relationship for it.

**Retrieval.** Three modes. Global search runs "map-reduce" over community reports ([query
overview](https://microsoft.github.io/graphrag/query/overview/)). Local search "identifies a set of entities from the
knowledge graph that are semantically-related to the user input" via "Entity Description Embedding," then pulls
"connected entities, relationships, entity covariates, and community reports" plus "relevant text chunks from the raw
input documents that are associated with the identified entities," and these "are then prioritized and filtered to fit
within a single context window of pre-defined size" ([local
search](https://microsoft.github.io/graphrag/query/local_search/)). The model never writes a graph query; entry-point
selection is an embedding lookup and expansion is a fixed neighbourhood walk. Community detection uses a "Hierarchical
Leiden Algorithm" ([default dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/)).

**Relevance.** Two direct lessons. (a) Entity entry via *embeddings of entity descriptions* is what this project's
resolver lacks at query time (`Resolver.resolve` in `resolve.py` is string matching only; D14 deferred embeddings),
and D55 shows the planner's `entity_surface` failing to resolve on `3h-08`/`oos-04` for exactly that reason. (b)
GraphRAG's local search *always* includes text units next to graph facts. D56's graph-only route was the outlier.

### 2.6 LlamaIndex PropertyGraphIndex / KnowledgeGraphIndex

**Ontology.** Labelled property graph: nodes with labels and properties, paths (triples). Extractors decide the labels
([PropertyGraphIndex
guide](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/)).

**Extraction.** Four extractors. `SimpleLLMPathExtractor` (default): "Extract short statements using an LLM to prompt
and parse single-hop paths in the format (entity1, relation, entity2)." `ImplicitPathExtractor` (default): uses
existing node relationship attributes without an LLM. `DynamicLLMPathExtractor`: optional type lists, not enforced.
`SchemaLLMPathExtractor`: "Extract paths following a strict schema of allowed entities, relationships, and which
entities can be connected to which relationships" ([PropertyGraphIndex
guide](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/)). The older
`KnowledgeGraphIndex` extracts triplets with `max_triplets_per_chunk` and can `include_embeddings=True`
([KnowledgeGraphIndex
demo](https://developers.llamaindex.ai/python/examples/index_structs/knowledge_graph/knowledgegraphdemo/)).

**Name resolution.** By entity string as emitted by the LLM. No alias table documented on the guide page.

**Call graph / inheritance / exceptions.** Not modelled as such; only whatever an extractor emits for the text.

**Retrieval.** Five retrievers. `LLMSynonymRetriever` (default): "Takes the query, and tries to generate keywords and
synonyms to retrieve nodes (and therefore the paths connected to those nodes)." `VectorContextRetriever` (default when
embeddings are on): vector similarity to nodes, then connected paths. `TextToCypherRetriever`: uses the graph schema
to generate Cypher. `CypherTemplateRetriever`: "Rather than letting the LLM have free-range of generating any cypher
statement, we can instead provide a cypher template and have the LLM fill in the blanks." `CustomPGRetriever`: user
code. "If no sub-retrievers are provided, the defaults are LLMSynonymRetriever and VectorContextRetriever (if
embeddings are enabled)" and results are merged ([PropertyGraphIndex
guide](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/)).
`KnowledgeGraphIndex` uses keyword-based querying with an optional `embedding_mode="hybrid"` ([KnowledgeGraphIndex
demo](https://developers.llamaindex.ai/python/examples/index_structs/knowledge_graph/knowledgegraphdemo/)).

**Relevance.** `CypherTemplateRetriever` is the same design as `cypher_templates.py` + `graph_query.py`, so "the model
never writes Cypher" is a mainstream option, not a contrarian one. The difference: LlamaIndex documents one template
per retriever; this project has one planner choosing among several templates with a single shared `GraphQueryPlan`
schema, which is what lets the model return `T3 + relationship=RAISES` (D54) — a combination no single-template schema
could produce.

### 2.7 LangChain LLMGraphTransformer

**Ontology.** `GraphDocument` with nodes and relationships; `allowed_nodes: List[str]`, `allowed_relationships` as
strings or 3-tuples `(source_type, rel, target_type)`, `node_properties` and `relationship_properties` as `True` or a
list ([llm.py
source](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py)).

**Extraction.** LLM per document; `convert_to_graph_documents` runs the chain on each document's text ([llm.py
source](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py)).
Inferred.

**Name resolution.** Prompt-based: "When extracting entities, it's vital to ensure consistency... always use the most
complete identifier for that entity throughout the knowledge graph"; labels should be "basic or elementary types";
relationships "general and timeless" ([llm.py
source](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py)).
No post-hoc alias table. `strict_mode=True` (default) filters nodes whose type is not in `allowed_nodes`
(case-insensitive) and relationships whose type, or `(source, rel, target)` triple, is not allowed ([llm.py
source](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py)).

**Call graph / inheritance / exceptions.** Not modelled.

**Retrieval.** The transformer is construction-only; retrieval is left to the graph store's chains. The current docs
URL redirects to a general overview page, so the retrieval side: **could not verify** from a current first-party doc
page.

**Relevance.** This project's `llm_extract.py` does the same ontology-constrained extraction with a Pydantic `Literal`
schema, plus two things LangChain's transformer does not: the model is told to emit "the exact surface name as
written" and never a qualified id, and every emitted edge carries the caller-supplied `chunk_id`
([`graphrag/ingest/llm_extract.py`](../../graphrag/ingest/llm_extract.py)). `strict_mode` in LangChain drops
out-of-schema output silently; this project's `ontology.py` makes an out-of-ontology write fail loudly at load time.

### 2.8 code-graph-rag (tree-sitter + Memgraph)

**Ontology.** Node labels: Project, Package, Folder, File, Module, Class, Function, Method, Interface, Enum, Type,
Union, Parameter, Field, EnumVariant, ExternalPackage, ExternalModule, plus doc/analysis labels (Section, Pattern,
CodeSmell, SecurityIssue, Resource). Relationships include DEFINES, CONTAINS_*, IMPORTS, EXPORTS, INHERITS,
IMPLEMENTS, OVERRIDES, RETURNS, ACCEPTS, CALLS, REFERENCES, INSTANTIATES, READS_FROM, WRITES_TO, FLOWS_TO,
HAS_PARAMETER, OF_TYPE, RESOLVES_TO, MENTIONS ([graph
schema](https://docs.code-graph-rag.com/architecture/graph-schema/)).

**Extraction.** "Uses Tree-sitter for robust, language-agnostic AST parsing" across 13 languages with "a unified graph
schema" ([architecture overview](https://docs.code-graph-rag.com/architecture/overview/)). Deterministic.

**Name resolution.** Layered heuristics. CALLS resolve "through scope, import, type or signature" (exact), then
overload matching, then "name-only trie suffix and wildcard imports (heuristic)," with optional runtime-trace
confirmation; IMPORTS edges carry `alias` and `imported_name`; INHERITS in C# can use a compiler-backed "hybrid mode"
versus "syntactic heuristics" elsewhere ([graph schema](https://docs.code-graph-rag.com/architecture/graph-schema/)).
Type inference for Python: not documented on the pages fetched.

**Call graph / inheritance / exceptions.** CALLS, INHERITS, OVERRIDES, INSTANTIATES present ([graph
schema](https://docs.code-graph-rag.com/architecture/graph-schema/)). Exception flow: no relationship type on the
schema page; **not modelled** as far as the schema shows.

**Retrieval.** "AI-Powered Cypher Generation" — the LLM writes Cypher — plus "Semantic Code Search using UniXcoder
embeddings to find functions by intent" ([homepage](https://docs.code-graph-rag.com/)). How the two combine: not
documented on the overview page.

**Relevance.** This is the closest open-source analogue in *shape* (parser → property graph → LLM query) and the
opposite choice on the query side: model-written Cypher versus this project's templates. Its schema also has a
`RESOLVES_TO` edge and explicit "heuristic" tiers, i.e. it records resolution confidence in the graph, which is what
`Resolution.method` does here but in a log, not the graph.

### 2.9 Neo4j's own codebase-graph example

Neo4j's developer blog builds a C# codebase graph with Roslyn ("Roslyn is a powerful .NET compiler that provides APIs
for code analysis"), with node layers for projects/packages, files/folders, types, and methods, and relationships
including `:HAVE`, `:IMPLEMENTED_AS`, `:INVOKE`, `:INSTANTIATE`; analysis is "by using a Cypher queries," written by
hand, no LLM and no embeddings ([Neo4j blog](https://neo4j.com/blog/developer/codebase-knowledge-graph/)). Included
only to show Neo4j's own reference design uses a compiler's semantic model for resolution, not a parser.

### 2.10 Python call-graph tools: PyCG and pyan

**PyCG.** "PyCG generates call graphs for Python code using static analysis" and supports "Higher order functions,"
"Twisted class inheritance schemes," "Automatic discovery of imported modules for further analysis," "Nested
definitions" ([PyCG README](https://raw.githubusercontent.com/vitsalis/PyCG/master/README.md)). Method: "We compute
all assignment relations between program identifiers of functions, variables, classes, and modules through an
inter-procedural analysis," producing an assignment graph; it handles "modules, generators, function closures, and
multiple inheritance" and reports "high rates of precision ~99.2%, and adequate recall ~69.9%" at "0.38 seconds for 1k
LoC on average" ([arXiv 2103.00587](https://arxiv.org/abs/2103.00587)). Output is a JSON adjacency list ([PyCG
README](https://raw.githubusercontent.com/vitsalis/PyCG/master/README.md)). The repo is archived: "PyCG is archived.
Due to limited availability, no further development improvements are planned" ([PyCG
GitHub](https://github.com/vitsalis/PyCG)). No inheritance or exception edges in the output; no retrieval.

**pyan.** Static AST analysis: it "reads through the source code, and makes deductions from its structure." Two edge
kinds: *defines* and *uses*. `self` handling: "the literal name representing `self` is captured from the argument
list, as Python does; then in the lexical scope of that method, that name points to the current class." It supports
"inherited attributes" and "Resolution of `super()` based on the static type at the call site." Limits: it "cannot
correctly track cases where the current binding of `self.f` depends on the order in which the methods of the class are
executed," instead it "uses the most recent binding that is currently in scope"; "A string annotation, `Optional[X]`,
or a union resolves to nothing" ([pyan README](https://github.com/Technologicat/pyan)). No exception edges; no
retrieval.

**Relevance.** pyan's `self` rule is word-for-word this project's `self_reference` rung (D46), which suggests the rung
was table stakes rather than an insight. PyCG's precision/recall numbers are the field's baseline for "static Python
call graph without type annotations": ~70% recall is what a well-built assignment-graph analysis gets, so this
project's 30.7% has headroom but the ceiling is not 100%.

### 2.11 Greptile, Cursor, Devin (first-party pages only)

**Greptile.** First-party page: the index "Parses every file to extract directories, files, functions, classes,
variables," "Connects all elements: function calls, imports, dependencies, variable usage," and "Stores the complete
graph for instant querying during code reviews"; at review time it "queries the pre-built graph" for dependencies and
call sites ([Greptile docs](https://www.greptile.com/docs/how-greptile-works/graph-based-codebase-context)). The page
does **not** say whether parsing is AST, LSP or LLM, how names are resolved, or whether embeddings are used. **No
primary source** for those questions.

**Cursor.** First-party blog: "When a file changes, Cursor splits it into syntactic chunks. These chunks are converted
into the embeddings that enable semantic search," synced via a Merkle tree of file hashes; "During search, the server
filters results by checking those hashes against the client's tree" ([Cursor
blog](https://cursor.com/blog/secure-codebase-indexing)). The post "does not mention graphs, ASTs, symbol resolution,
or call graphs." The current docs URL for codebase indexing now describes "Instant Grep," which "builds and queries
its index on your machine" and "does not store embeddings of your codebase for search" ([Cursor
docs](https://cursor.com/docs/context/codebase-indexing), as of 2026-09-24). Either way: **no code graph documented**;
it is chunk embeddings and grep.

**Devin.** First-party docs describe DeepWiki output ("architecture diagrams, documentation, and source links") and
configuration via `.devin/wiki.json`, but nothing about the indexing mechanism ([Devin
docs](https://docs.devin.ai/work-with-devin/deepwiki)). **No primary source** on graph construction.

---

## 3. Comparison table

| Tool | Extraction | Name resolution | Call graph | Inheritance | Exception flow | Retrieval combination | Who writes the query |
|---|---|---|---|---|---|---|---|
| SCIP / scip-python | Pyright fork, deterministic | Full type checker; package+version in symbol | Derived from references (no call edge type) | `is_implementation` relationship | No | None (precise nav, search fallback) | IDE/UI, fixed operations |
| CodeQL | Extractor → AST+CFG+DFG, deterministic | Points-to + type trackers, MRO approx | `resolveCall`, incl. `self`/`cls`/instantiation | `ClassValue`, MRO | CFG has exceptional paths; wrapping = user query | None | Human (QL) |
| Joern CPG | Frontend → layered CPG, deterministic | Type propagation with dummy types | `CALL` edges w/ `DISPATCH_TYPE` | `INHERITS_FROM` | `CONTROL_STRUCTURE` + `CFG`; no wrap relation | None | Human (Scala DSL) |
| Aider repo map | tree-sitter tags, deterministic | None (string equality, weighted) | File-level reference edges only | No | No | PageRank subset + chat files; no vectors | Nobody; map is computed |
| MS GraphRAG | LLM over all text | Merge by (title, type) | No | No | No | Entity embeddings → neighbourhood + text units + reports | Nobody; fixed walk |
| LlamaIndex PGI | LLM (schema-guided optional) + implicit | LLM string consistency | No | No | No | Synonym + vector retrievers merged; text-to-Cypher or template optional | Model (text-to-Cypher) or model fills template |
| LangChain LLMGraphTransformer | LLM, `strict_mode` filter | Prompt: "most complete identifier" | No | No | No | Construction only | n/a |
| code-graph-rag | tree-sitter, deterministic | Scope/import/type/signature, then name heuristics | `CALLS`, `INSTANTIATES` | `INHERITS`, `OVERRIDES` | No | LLM Cypher + UniXcoder embeddings (combination undocumented) | Model writes Cypher |
| PyCG | AST → assignment graph, inter-procedural | Assignment-flow, no annotations needed | Yes, 99.2% P / 69.9% R | Handled internally, not output | No | None | n/a |
| pyan | AST, linear two-pass | Lexical bindings; `self` → class; `super()` | Yes (`uses`) | `inherited attributes` | No | None | n/a |
| Greptile / Cursor / Devin | Not documented / chunk embeddings / not documented | Not documented | Claimed (Greptile) / no / n.d. | n.d. / no / n.d. | n.d. / no / n.d. | Graph query at review (Greptile); vector (Cursor) | n.d. |
| **This project** | Python `ast` (structure) + LLM (prose only) | 8-rung ladder: exact → self → import alias → re-export → normalized → public API → call graph → unresolved; no types | `CALLS` per function, surfaces; `self.*` only | `INHERITS_FROM` per class, full base list | `RAISES` per function; `WRAPS_EXCEPTION` LLM-only (AST extractor unwired) | Router → template facts + vector passages, always both | Model picks template id + params; code validates; never Cypher |

---

## 4. Where this project is unusual

**Two extraction paths through one resolver, with the LLM forbidden from emitting ids.** GraphRAG, LlamaIndex and
LangChain let the model name entities and merge on strings; the compiler tools never involve a model. Here
`llm_extract.py` demands "the exact surface name as written," and `resolve.py` treats a code-derived `except
MaxRetryError` and a prose-derived "MaxRetryError" identically. No surveyed tool separates "what the text said" from
"what it refers to" this cleanly for LLM output. It is what made D33/D47/D48 possible: 4.5x the facts from replaying a
log, zero re-extraction.

**Refusal to guess, recorded as data.** `Resolution.method` and `candidates` make the resolution rate a number (D14,
D33). code-graph-rag comes closest with `RESOLVES_TO` and labelled "heuristic" tiers; the LLM frameworks have nothing
equivalent; aider *penalises* ambiguity (`len(defines[ident]) > 5: mul *= 0.1`) rather than reporting it.

**One edge per function, aggregated, with no positions (D10, D12).** SCIP records an `Occurrence` with a range for
every reference; the CPG gives every `CALL` a `LINE_NUMBER` and CFG edges. This project's
`RaisesEdge.exceptions_raised` and `CallsEdge.calls` are lists with no location and no control-flow context. That is
unusual and it is a cost: it is why `extract_exception_wrapping` cannot tell that `raise ReadTimeout` is guarded by
`isinstance(e, ReadTimeoutError)` — the guard was thrown away one level up.

**Typed template parameters validated against the ontology and the resolved-entity set**
(`cypher_templates.py::_validate_param`). LlamaIndex's `CypherTemplateRetriever` fills blanks; this project
additionally rejects an entity id the model did not obtain from the resolver. Good, and rare.

**Prose-only LLM extraction.** Every LLM graph tool surveyed runs the model over all text. Restricting it to
docstrings, docs and changelogs (D2) is unique in the survey and is the reason the AST/LLM boundary left a gap:
exception wrapping is *in code* but is a *relationship*, so neither pass owned it until D44.

**The benchmark discipline** (noise floor, controlled single-variable tests, predictions checked one-for-one in
D45/D56) is not something any surveyed tool's docs describe. Not a graph-design point, but it is the part of this
project a reader is least likely to find elsewhere.

---

## 5. Where this project is behind

1. **Receiver resolution beyond `self`.** Every code tool in the survey resolves more: Pyright (scip-python),
   points-to/type-tracking (CodeQL), type propagation (Joern), assignment graph (PyCG), lexical bindings (pyan).
   `conn.urlopen` in `HTTPAdapter.send` — the single most important call in the corpus (D12) — is unresolved here.
   30.7% of call surfaces resolve (D46) against PyCG's ~70% recall.

2. **No positions or intra-function structure on edges.** See §4. SCIP and CPG keep occurrence ranges; this project
   keeps a `chunk_id` per function. Fine for citation, useless for control-flow-sensitive facts.

3. **Exception wrapping is LLM-only in the live graph.** All 77 `WRAPS_EXCEPTION` edges are from the prose pass; the
   deterministic extractor exists but is never called (D56). CodeQL and Joern treat exceptions as control flow; the
   LLM tools ignore them; this project has the right extractor sitting unwired.

4. **Reference edges are calls only.** SCIP, aider and Greptile record *any* reference to a symbol. `isinstance(e,
   ReadTimeoutError)`, a type annotation, or a bases tuple are references this project's `CALLS` pass never sees, so
   `T1_NEIGHBORS` cannot find "where is `ReadTimeoutError` mentioned in code."

5. **No override edges.** code-graph-rag has `OVERRIDES`; CodeQL approximates MRO. `BaseAdapter.send` vs
   `HTTPAdapter.send` (the D10 bug) is exactly an override relationship, and the planner's
   `RetryError`/`MaxRetryError` confusion (D55) is a sibling problem: the graph has no "same-name, different-owner"
   link to consult.

6. **Query-time entity linking is string-only.** GraphRAG uses entity-description embeddings; LlamaIndex defaults to a
   synonym retriever plus a vector retriever. This project's planner hands a surface string to the same
   exact/normalized matcher used at ingestion (`graph_query.py::build_template_values`), and D55's two "resolver error
   vs bare `requests`" flips are the result. D14 deferred exactly this.

7. **No ranking of graph results.** Aider ranks with PageRank and fits a token budget; GraphRAG "prioritized and
   filtered to fit within a single context window." `T1_NEIGHBORS` returns everything, unordered, and D56 shows the
   model saying "not enough information" with 36 facts in front of it.

8. **No incremental re-index.** Cursor's Merkle tree and SCIP's per-document indexes are incremental. D56 ends with
   "needs re-ingestion" for a single extractor change.

---

## 6. What it could adopt (ranked)

Each item names the tool whose approach it borrows, the file here it would touch, and honest effort. "Would not help"
items are at the end.

**1. Wire `extract_exception_wrapping`, add `isinstance` narrowing, and let AST wrap edges override LLM ones.**
*(CodeQL/Joern: exceptions are control flow; PyCG: precision over recall.)* Files: `graphrag/ingest/ast_extract.py`
(treat `isinstance(<handler var>, X)` in a handler as narrowing the caught set to X on that branch), the ingestion
runner (call it), and the write path (when an AST and an LLM edge disagree on the same `(raised, caught)` pair, keep
the AST one). Effort: small; this is the project's own next-thread plan in D56. It directly fixes the only remaining
benchmark loss (th-03) and it operationalises D56's lesson that a wrong fact is worse than none.

**2. Check the aliased-import path in the wrapping pass.** *(SCIP/code-graph-rag: import aliases resolve by binding
name.)* D44 reports "aliased imports like `_SSLError`" among the 50 unresolved wrap facts, but `extract_imports`
already records `asname` as `local_name`, and `Resolver.resolve` consults `import_aliases[module_context]`. Worth
confirming that the wrapping pass passes `module_context=` to `resolve()`; if it does not, that is a one-line fix
worth ~a dozen edges, including `ReadTimeoutError → ReadTimeout`. Effort: trivial to check. (Hypothesis, not verified
here.)

**3. Per-template Pydantic plans instead of one shared `GraphQueryPlan`.** *(LlamaIndex `CypherTemplateRetriever`: one
schema per template.)* File: `graphrag/retrieval/graph_query.py`. Make the model return a discriminated union where
`T3_EXCEPTION_WRAP_CHAIN` has no `relationship` field and `T6_COUNT_BY_REL` has no `entity_surface`, so D54's "T3 +
rel=RAISES" and "T6 with a named entity" become unrepresentable rather than merely discouraged in prose. Effort:
small. D54 asked for exactly this ("the next attempt should be structural").

**4. Query-time entity linking by embedding, as a rung after `normalized`.** *(GraphRAG entity-description embeddings;
LlamaIndex `VectorContextRetriever`/`LLMSynonymRetriever`.)* Files: `graphrag/ingest/resolve.py` (the deferred D14
step 4) and `graph_query.py`. Embed each node's `embed_text` (already in Postgres) and, when the planner's surface
resolves to nothing or to a bare package, take the nearest node above a threshold and *record the method as
`embedding`* so the benchmark can count how often it fires and how often it is wrong. Effort: medium. Targets D55's
`3h-08`/`oos-04` flips. Risk: it reintroduces guessing; the `method` field is what keeps it honest.

**5. Rank neighbourhood results and cap them.** *(Aider: PageRank + token budget; GraphRAG: prioritise to fit the
window.)* Files: `graphrag/retrieval/merge.py` (order facts), possibly a precomputed in-degree per node at load time.
Cheapest version: sort `T1_NEIGHBORS` rows by the neighbour's in-degree, cap at N, and put the AST-derived
relationship types before LLM-derived ones. Effort: small. Measure on the 16 GRAPH-route questions from D56.

**6. Add reference edges for non-call mentions.** *(SCIP `Occurrence` roles; aider `name.reference`; Greptile
"variable usage".)* File: `ast_extract.py`: a `REFERENCES` edge per function listing every `Name`/`Attribute` surface
that is not already a call target, resolved through the same ladder. This requires adding a relationship type to the
frozen ontology (`ontology.py`), which the design treats as a real decision. Effort: small-medium. It is what makes
"where is X mentioned" answerable and gives item 4 more anchors.

**7. Resolve one more receiver class: locals bound from a constructor or a `self` call in the same function.**
*(pyan's "most recent binding in scope"; PyCG's assignment graph, in miniature.)* File: `ast_extract.py` +
`resolve.py`. Track `x = ClassName(...)` and `x = self.method(...)` inside a function; when `x.attr(...)` follows,
resolve `attr` against `ClassName` or against the return annotation of `method` if present. Effort: medium. Would
resolve `conn = self.get_connection_with_tls_context(...)` → `conn.urlopen` only if `get_connection...` has a return
annotation, which it may not; check the corpus first. This is the point at which building further starts to look like
reimplementing PyCG.

**8. Consume a real call graph instead of building one.** *(scip-python / PyCG output as an input edge set.)* Run
scip-python (Pyright) over the two packages and import its resolved references as `CALLS` edges tagged
`method="scip"`; or run PyCG (archived, but functional) for the adjacency list. Effort: medium, mostly plumbing.
Trade: the "I built the resolver" story weakens, but the benchmark can then answer "does a better call graph change
accuracy?" — which is the question that matters for AI-engineering intuition, and it cannot be answered by improving
`resolve.py` 5% at a time.

**9. `OVERRIDES` edges.** *(code-graph-rag; CodeQL MRO.)* Derivable at load time from `INHERITS_FROM` plus same-named
methods, like `DEFINED_IN` (D15). Effort: small, needs an ontology addition. Gives the planner a link between
`BaseAdapter.send` and `HTTPAdapter.send`.

**What would NOT help, and why.**

- **Full type inference built in-house.** The corpus is two libraries; PyCG-grade analysis is a research project and
  its own authors report ~70% recall. Item 8 gets the result without the build.
- **Community detection and summaries (GraphRAG).** Leiden communities exist to answer "what is this dataset about"
  over corpora too large to read. This graph is 3.3k nodes with a known module structure; module-level summaries would
  be a cheaper substitute if global questions ever matter, and no benchmark category asks one.
- **Model-written Cypher (code-graph-rag, LlamaIndex `TextToCypherRetriever`).** D54 shows the planner already
  mis-selects among six templates with prose guidance; free Cypher would widen the space of wrong plans and remove the
  validation that catches invented ids. Item 3 tightens the current design instead.
- **Switching to tree-sitter.** Python's `ast` *is* the reference parser for the one language in scope, and it gives
  `ast.unparse`, relative-import levels and `ExceptHandler` structure for free. tree-sitter buys multi-language
  support this project does not need.
- **More router/planner voting.** D55 measured it: 1.7 points for 3x cost, because the residual variance is structural
  (resolver failures, sibling-exception confusion), which items 3, 4 and 9 attack directly.

---

## 7. Sources

All fetched 2026-09-24.

- SCIP protocol schema: https://github.com/sourcegraph/scip/blob/main/scip.proto
- SCIP README: https://github.com/sourcegraph/scip
- scip-python README: https://github.com/sourcegraph/scip-python
- Sourcegraph precise code navigation:
  https://sourcegraph.com/docs/code_navigation/explanations/precise_code_navigation
- CodeQL, About CodeQL: https://codeql.github.com/docs/codeql-overview/about-codeql/
- CodeQL library for Python: https://codeql.github.com/docs/codeql-language-guides/codeql-library-for-python/
- CodeQL, Functions in Python: https://codeql.github.com/docs/codeql-language-guides/functions-in-python/
- CodeQL, Analyzing control flow in Python:
  https://codeql.github.com/docs/codeql-language-guides/analyzing-control-flow-in-python/
- CodeQL, Analyzing data flow in Python:
  https://codeql.github.com/docs/codeql-language-guides/analyzing-data-flow-in-python/
- CodeQL `DataFlowDispatch` module docs:
  https://codeql.github.com/codeql-standard-libraries/python/semmle/python/dataflow/new/internal/DataFlowDispatch.qll/module.DataFlowDispatch.html
- CodeQL `ObjectAPI` module docs:
  https://codeql.github.com/codeql-standard-libraries/python/semmle/python/objects/ObjectAPI.qll/module.ObjectAPI.html
- Joern Code Property Graph docs: https://docs.joern.io/code-property-graph/
- Joern CPG specification: https://cpg.joern.io/
- Joern Python frontend docs: https://docs.joern.io/frontends/python/
- pysrc2cpg README: https://github.com/joernio/joern/blob/master/joern-cli/frontends/pysrc2cpg/README.md
- Aider repo map docs: https://aider.chat/docs/repomap.html
- Aider tree-sitter repo map post: https://aider.chat/2023/10/22/repomap.html
- Aider `repomap.py`: https://raw.githubusercontent.com/Aider-AI/aider/main/aider/repomap.py
- Microsoft GraphRAG indexing overview: https://microsoft.github.io/graphrag/index/overview/
- Microsoft GraphRAG default dataflow: https://microsoft.github.io/graphrag/index/default_dataflow/
- Microsoft GraphRAG query overview: https://microsoft.github.io/graphrag/query/overview/
- Microsoft GraphRAG local search: https://microsoft.github.io/graphrag/query/local_search/
- LlamaIndex PropertyGraphIndex guide:
  https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/
- LlamaIndex KnowledgeGraphIndex demo:
  https://developers.llamaindex.ai/python/examples/index_structs/knowledge_graph/knowledgegraphdemo/
- LangChain `LLMGraphTransformer` source:
  https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/graph_transformers/llm.py
- code-graph-rag homepage: https://docs.code-graph-rag.com/
- code-graph-rag architecture overview: https://docs.code-graph-rag.com/architecture/overview/
- code-graph-rag graph schema: https://docs.code-graph-rag.com/architecture/graph-schema/
- Neo4j developer blog, codebase knowledge graph (C#/Roslyn):
  https://neo4j.com/blog/developer/codebase-knowledge-graph/
- PyCG README: https://raw.githubusercontent.com/vitsalis/PyCG/master/README.md
- PyCG repository (archived notice): https://github.com/vitsalis/PyCG
- PyCG paper, ICSE 2021: https://arxiv.org/abs/2103.00587
- pyan README: https://github.com/Technologicat/pyan
- Greptile, graph-based codebase context:
  https://www.greptile.com/docs/how-greptile-works/graph-based-codebase-context
- Cursor blog, securely indexing large codebases: https://cursor.com/blog/secure-codebase-indexing
- Cursor docs, codebase indexing (currently "Instant Grep"): https://cursor.com/docs/context/codebase-indexing
- Devin docs, DeepWiki: https://docs.devin.ai/work-with-devin/deepwiki
- This project: `docs/DECISIONS.md` (D1–D12, D14, D33–D36, D44–D48, D54–D56); `graphrag/ontology.py`;
  `graphrag/ingest/ast_extract.py`; `graphrag/ingest/resolve.py`; `graphrag/ingest/llm_extract.py`;
  `graphrag/retrieval/cypher_templates.py`; `graphrag/retrieval/graph_query.py`; `graphrag/retrieval/router.py`;
  `graphrag/retrieval/merge.py`; `graphrag/answer/pipeline.py`.
