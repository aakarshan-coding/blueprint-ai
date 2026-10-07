#!/usr/bin/env bash
# Fetch the two libraries the graph is built from, at the exact commits the
# benchmark numbers were measured on. Idempotent: re-running checks the commits.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {
  local dir="$1" url="$2" commit="$3"
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$commit"
  echo "$dir at $(git -C "$dir" rev-parse --short HEAD)"
}

fetch requests_repo https://github.com/psf/requests.git dae7ef63
fetch urllib3_repo  https://github.com/urllib3/urllib3.git b1d30ab
