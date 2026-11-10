#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/.venv"
test -x "$VENV/bin/python" || { echo "Expected virtual environment at $VENV" >&2; exit 1; }
test -f "$HOME/.config/remote-ollama/env" || { echo "Missing ~/.config/remote-ollama/env" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$HOME/.config/remote-ollama/env"
set +a
export CODING_WORKSPACE="${CODING_WORKSPACE:-$HOME/HopeLinkCancer-AI}"
export CODING_AGENT_DIR="${CODING_AGENT_DIR:-$HOME/ml-project/agent-workspace}"
export WEB_AGENT_COOKIE_SECURE="${WEB_AGENT_COOKIE_SECURE:-1}"
: "${WEB_AGENT_PASSWORD:?Set WEB_AGENT_PASSWORD to a unique password (14+ characters)}"
: "${WEB_AGENT_SESSION_SECRET:?Set WEB_AGENT_SESSION_SECRET to a random secret (32+ characters)}"
cd "$ROOT"
exec "$VENV/bin/uvicorn" app:app --host 127.0.0.1 --port 8765
