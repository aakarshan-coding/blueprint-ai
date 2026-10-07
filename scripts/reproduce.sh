#!/usr/bin/env bash
# From a clean clone to the README table. Needs Docker, Python 3.12+, and
# OPENAI_API_KEY in the environment (see .env.example). The LLM extraction
# pass is replayed from data/llm_edges_log.jsonl, so ingestion makes no
# model calls; the benchmark does (about $3 per run).
set -euo pipefail
cd "$(dirname "$0")/.."

scripts/fetch_corpus.sh
docker compose up -d
python -m graphrag.ingest.run_ingestion --ast-only      # parser-derived graph + vector store
python -m graphrag.ingest.replay_llm_edges              # the model-derived edges, from the log
for i in 1 2 3 4 5; do
  python -m graphrag.eval.run_benchmark
  cp results/latest_benchmark_results.json "results/local/$i.json"
done
python -m graphrag.eval.summarize_runs results/local/*.json
