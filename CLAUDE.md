# Working on this repository

Graph + vector RAG over Python codebases, with a benchmark that measures the graph's
contribution. Read `README.md` first; it explains the system in the order it runs. The
decisions log, `docs/DECISIONS.md`, records every change with its evidence, and is the
place to look before proposing something that may have been tried.

## Rules that were learned the hard way

1. **Change the system, re-run the benchmark; change the grading, re-score.** Any change
   to retrieval, ingestion or the synthesis prompt alters the answers and needs a fresh
   five-run benchmark (`scripts/reproduce.sh`, about 2.5 hours on 220 questions). A
   grader or reference change never needs a re-run: re-score the recorded answers in
   `results/<experiment>/` offline. Never compare a single run to a five-run mean; the
   noise floor is about 12% of verdicts.
2. **Rebuild the graph from scratch before claiming a number.** `graphrag ingest --corpus
   corpora/requests-urllib3.yaml` then `python -m graphrag.ingest.replay_llm_edges`. A
   graph that accumulated incremental re-applies hid a parser gap for 38 decisions (D94).
3. **A graph-side change may not touch anything the baseline shares.** The vector-only
   baseline uses the same chunks, passages, answer model and synthesis prompt. Editing the
   shared prompt to help the graph side once collapsed the baseline's refusals (D61).
4. **Repairs act on what the graph proves, never on labels.** A plan repair may use a fact
   the graph guarantees (a Module has no RAISES edges; a Parameter has no CALLS edges).
   It may not key on a node label ingestion assigns inconsistently (D77). When unsure,
   run the planned query and fall back only if it returns nothing.
5. **The model never writes a query.** It picks a template from
   `graphrag/retrieval/cypher_templates.py` and fills typed parameters, each validated
   before anything runs. New question shapes get a new template or a new expansion entry
   in `graphrag/retrieval/retrieve.py`, not model-written Cypher.
6. **Parser for structure, model for prose.** Anything the code states directly is
   extracted with `ast` or jedi and tagged `source="ast"`/`"jedi"`; the model pass reads
   documentation only. The runtime oracle (`python -m graphrag.eval.runtime_oracle`)
   scores parser-derived edges against real execution; run it after any extractor change.
7. **Log every decision.** Append a `D<n>` entry to `docs/DECISIONS.md` with what was
   tried, what was measured and what was kept or reversed, at each checkpoint, without
   being asked. Side findings go in the log, not into the current thread.
8. **No secrets in files.** `OPENAI_API_KEY` is read from the environment only. Scan a
   diff for keys before every commit.

## Commands

```bash
pip install -e ".[embed,dev]"          # embed = sentence-transformers/torch; dev = pytest
pytest                                 # 389 unit tests, no database, no model calls
docker compose up -d                   # Neo4j 5 and Postgres with pgvector
graphrag ingest <repo> [<repo>...]     # any Python repo: parser pass + embeddings, no model calls
graphrag ingest <repo> --with-llm      # also the model pass over docs (caller's key, logged for replay)
graphrag ask "question" --show-plan    # one answer with citations
graphrag ask "question" --compare      # also the vector-only answer, side by side
scripts/demo.sh                        # five-step demo on itsdangerous (replaces the stored corpus)
python -m graphrag.eval.run_benchmark  # one run on the 220 questions, ~28 min, ~$3
python -m graphrag.eval.summarize_runs results/exp10/*.json
python -m graphrag.eval.runtime_oracle --no-run   # re-score the graph against the saved trace
```

Tools write scratch output under `results/` (gitignored); kept runs live under
`results/<experiment>/` and are indexed in `results/README.md`.

## Where things are

- `graphrag/ontology.py`: node and relationship types with the meaning shown to the model.
- `graphrag/corpus.py`: which repositories the graph is over; detected or from `corpora/`.
- `graphrag/ingest/`: chunking, `ast` and jedi extraction, name resolution, store loading.
- `graphrag/retrieval/`: router, mentions, planner and `repair_plan`, templates,
  expansion table `_EXPANSIONS`, context assembly in `merge.py`.
- `graphrag/eval/`: grader, benchmark, five-run summaries, runtime oracle.
- `tests/fakes.py`: the shared fakes; tests never need a database or a key.

## Communication style

When explaining work you performed on this project, optimize for clarity and continuity, not brevity.

Assume I understand the project technically, but I do NOT necessarily remember every experiment ID, implementation detail, previous debugging decision, or piece of terminology from earlier in the conversation.

For any substantial change, investigation, experiment, or debugging result, explain it in this order:

1. **What we were trying to figure out**
   - Briefly remind me what question or problem motivated the work.
   - Do not start with an experiment ID or metric without first explaining what the experiment was testing.

2. **What happened**
   - State the important result in plain English first.
   - Then give the relevant numbers.

3. **Why it happened**
   - Walk through the causal chain.
   - Explicitly connect components instead of assuming I remember how they interact.
   - If a specific edge, parser behavior, retrieval step, or data structure caused the issue, explain what it represents and why its absence/presence affects the final result.

4. **What you changed**
   - Explain the actual code/system change and where it fits into the architecture.
   - Distinguish clearly between:
     - what was already true,
     - what was broken or missing,
     - what you changed now.

5. **How we know the change is correct**
   - Explain what test, benchmark, trace, or measurement supports the conclusion.
   - Give metrics after explaining what they measure.

6. **Current state**
   - Clearly state what is now fixed, what is still uncertain, and what has not yet been tested.

7. **Next step**
   - State the most logical next action and why it matters.

### Writing rules

- Prefer complete, conversational technical explanations over compressed changelog prose.
- Do not write sentences like "Three-hop fell to 62.2. That drop is real, and it was the point of the run" without explaining what the run was designed to test.
- Do not introduce experiment IDs such as D94 or D95 before explaining what those experiments actually did.
- Do not assume terms like "historical graph," "oracle," "envelope," or "model edge" are self-explanatory. Briefly remind me what they mean in this specific context.
- Make causal connections explicit using language like:
  - "This matters because…"
  - "The reason this affected three-hop retrieval is…"
  - "Previously, the system…"
  - "After this change…"
- Avoid vague references such as "that shape," "the gap," "the envelope," "those questions," or "that number" when a specific noun would be clearer.
- Do not optimize for token efficiency at the expense of understanding.
- Use paragraphs rather than terse bullet fragments when explaining reasoning.
- It is fine to repeat a small amount of context if it makes the explanation understandable without rereading earlier messages.

The goal is that I should be able to read your explanation several days later and reconstruct:
**what problem we had → what caused it → what changed → what evidence supports it → what remains to do.**
