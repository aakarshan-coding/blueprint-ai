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
graphrag status                        # which repos are ingested; graph and text-index counts
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

These rules apply when you explain your work on this project. Clarity and continuity are more important than a short text.

Write all responses in Simplified Technical English (ASD-STE100). Use one topic in each sentence. Use active voice and simple words. Use numbered lists for sequences.

I know the project technically. I possibly do not remember the experiment IDs, the implementation details, the past debugging decisions or the terms from earlier in the conversation.

For each large change, investigation, experiment or debugging result, give the explanation in this sequence:

1. **The question**
   - Tell me again which question or problem started the work.
   - Do not start with an experiment ID or a metric.
   - First, tell me what the experiment tested.

2. **The result**
   - Give the important result in plain words first.
   - Then give the numbers.

3. **The cause**
   - Give each step of the cause, in sequence.
   - Tell me how the components connect. Do not think that I remember this.
   - Sometimes an edge, a parser behavior, a retrieval step or a data structure causes the problem. Tell me what that item is.
   - Tell me how its presence or its absence changes the final result.

4. **The change**
   - Tell me what you changed in the code or in the system.
   - Tell me where the change is in the architecture.
   - Keep these three items separate:
     - what was already true
     - what was broken or missing
     - what you changed now

5. **The evidence**
   - Tell me which test, benchmark, trace or measurement shows that the change is correct.
   - Tell me what a metric measures before you give its value.

6. **The current state**
   - Tell me what is now fixed.
   - Tell me what is not certain yet.
   - Tell me what is not tested yet.

7. **The next step**
   - Tell me the best next action.
   - Tell me why it is important.

### Rules for the text

- Write full explanations in a conversational, technical style. Do not write short changelog text.
- Do not give a result without its context. This is a bad example: "Three-hop fell to 62.2. That drop is real, and it was the point of the run." First, tell me what the run tested.
- Do not use an experiment ID, for example D94 or D95, before you tell me what that experiment did.
- Do not think that project terms are clear to me. Examples are "historical graph", "oracle", "envelope" and "model edge". Each time, tell me what the term means in this project.
- Show each cause and its effect clearly. Use phrases like these:
  - "This is important because..."
  - "This changed three-hop retrieval because..."
  - "Before this change, the system..."
  - "After this change..."
- Use a specific noun. Do not use a vague reference, for example "that shape", "the gap", "the envelope", "those questions" or "that number".
- Do not make an explanation shorter if it then becomes less clear.
- Use paragraphs to explain why something occurred. Do not use short bullet fragments for this.
- You can repeat a small quantity of context. Do this when it lets me understand the explanation without the earlier messages.

### The objective

Some days later, I must be able to read your explanation and find these five items:
**the problem → the cause → the change → the evidence → the work that remains.**
