# Remote Ollama + MCP Web Agent

A separate browser interface for your existing remote Ollama + MCP coding agent. It does not overwrite `~/ml-project/agent-workspace/server.py`, `client.py`, or `ai.py`, and does not touch `~/HopeLinkCancer/hopelinkcancer-web`.

## Requirements and defaults
- Python virtual environment: `~/.venv` (Python 3.13)
- Existing credentials: `~/.config/remote-ollama/env`
- MCP server: `~/ml-project/agent-workspace/server.py`
- Workspace: `~/HopeLinkCancer-AI`
- History: `~/.local/share/remote-ollama-web/history`

## Install in WSL Ubuntu
Extract this folder to `~/ml-project/remote-ollama-web`. Then run the setup script. It first creates a timestamped backup of any existing `server.py`, `client.py`, and `ai.py` files in `~/ml-project/agent-workspace/backups/`, then installs dependencies in your existing venv. It does not overwrite those files or modify the HopeLinkCancer web app:

```bash
~/ml-project/remote-ollama-web/setup.sh
```

Create a strong app password and session secret in your terminal. The password is not your Ollama API key:

```bash
export WEB_AGENT_PASSWORD='choose-a-long-random-password'
export WEB_AGENT_SESSION_SECRET="$(~/.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

Check that `~/.config/remote-ollama/env` contains the variable names `OLLAMA_REMOTE_URL` and `OLLAMA_REMOTE_API_KEY`. Do not paste the API key into browser code or share it in a URL.

## Local smoke test
For localhost HTTP only, disable Secure cookies for this one test session:

```bash
export WEB_AGENT_COOKIE_SECURE=0
~/ml-project/remote-ollama-web/run-local.sh
```

Open `http://127.0.0.1:8765`. Sign in, verify model listing, browse files, and try `pwd` from the terminal panel. Stop with Ctrl+C. Never use `WEB_AGENT_COOKIE_SECURE=0` over an untrusted network.

## Public internet deployment
The server intentionally binds only to `127.0.0.1`; do not expose port 8765 directly or open it on your router. Keep `WEB_AGENT_COOKIE_SECURE=1` and put an HTTPS access layer in front, preferably Cloudflare Tunnel plus Cloudflare Access configured to allow only your identity. Point the tunnel at `http://127.0.0.1:8765` and keep this app's password as a second layer. Test that Access blocks an unauthorised account before publishing.

Your laptop and WSL must be awake and connected for remote access. Do not deploy publicly until you have inspected the MCP tools in `server.py`: the app requires confirmation before every MCP call and terminal command, but this is not a complete OS sandbox. The command runner is allowlisted and rejects shell operators, but allowed commands can still be powerful.

## Behavior
- Chat requests go to remote Ollama using server-side credentials.
- Model-generated MCP tool calls are presented as proposals, not auto-executed.
- All direct MCP tool calls and terminal commands require explicit approval.
- File browsing is confined to `~/HopeLinkCancer-AI`, rejects path traversal and limits text reads to 1 MB.
- Conversations are stored locally under `~/.local/share/remote-ollama-web/history`.
- The backend sends both `Authorization: Bearer` and `X-API-Key` headers. If your remote Ollama gateway uses a different authentication scheme, adjust `ollama_headers()` in `app.py`.
