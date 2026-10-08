#!/usr/bin/env bash
# A five-minute demo on a repository the project was never tuned on.
#
# itsdangerous is the Pallets library Flask uses to sign session cookies: small
# (about 180 graph nodes, a two-minute ingest), with its own exception hierarchy,
# and nothing to do with HTTP. Needs Docker and OPENAI_API_KEY.
#
# Note: ingesting replaces whatever corpus the stores held. To go back to the
# measured requests + urllib3 corpus afterwards:
#   graphrag ingest --corpus corpora/requests-urllib3.yaml
#   python -m graphrag.ingest.replay_llm_edges
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?set OPENAI_API_KEY first}"
DEMO_DIR=.demo/itsdangerous
COMMIT=672971d66a

step() { printf '\n\033[1m### %s\033[0m\n' "$1"; [ -n "${PAUSE:-}" ] && read -r -p "(enter) " _ || true; }

if [ ! -d "$DEMO_DIR/.git" ]; then
  git clone --quiet https://github.com/pallets/itsdangerous.git "$DEMO_DIR"
fi
git -C "$DEMO_DIR" checkout --quiet "$COMMIT"
docker compose up -d >/dev/null

step "1. Ingest a repository the system has never seen (no model calls)"
graphrag ingest "$DEMO_DIR"

step "2. A structural question: the plan is a fixed template, not model-written Cypher"
graphrag ask "Which classes in itsdangerous.exc inherit from BadSignature?" --show-plan

step "3. The same kind of question, graph + text against text only"
graphrag ask "Which exceptions does itsdangerous.exc define, and what does each inherit from?" --compare

step "4. The context the answer was written from: every fact tagged with where it came from"
# The raw text passages are left out here; they follow the graph lines in the real context.
graphrag ask "What does Signer.unsign raise?" --show-plan --show-context \
  | sed '/=== RETRIEVED PASSAGES ===/,/^---$/d'

step "5. Out of scope: an empty graph result is a refusal signal"
graphrag ask "How do I configure nginx as a reverse proxy?" --show-plan
