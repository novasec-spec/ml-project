#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/.venv"
AGENT="$HOME/ml-project/agent-workspace"
if [[ ! -x "$VENV/bin/python" ]]; then echo "Expected Python venv at $VENV" >&2; exit 1; fi
if [[ ! -f "$HOME/.config/remote-ollama/env" ]]; then echo "Missing $HOME/.config/remote-ollama/env" >&2; exit 1; fi
# Back up existing agent files before any installation. Nothing in the agent is overwritten.
if [[ -d "$AGENT" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  backup="$AGENT/backups/web-ui-$stamp"
  mkdir -p "$backup"
  copied=0
  for file in server.py client.py ai.py; do
    if [[ -f "$AGENT/$file" ]]; then cp -a "$AGENT/$file" "$backup/"; copied=$((copied+1)); fi
  done
  if [[ "$copied" -gt 0 ]]; then echo "Backed up $copied existing agent file(s) to: $backup"; else rmdir "$backup"; echo "No server.py/client.py/ai.py files found to back up."; fi
else
  echo "Warning: existing MCP agent directory not found at $AGENT; no agent files were changed." >&2
fi
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
echo "Setup complete. This web app is separate from the existing agent and HopeLinkCancer web app."
echo "Before starting, export WEB_AGENT_PASSWORD and WEB_AGENT_SESSION_SECRET as described in README.md."
