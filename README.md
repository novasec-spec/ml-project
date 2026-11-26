# ml-project

A workspace of machine-learning and AI-agent experiments: local LLM agents
over MCP (Model Context Protocol), a web front end for a remote Ollama
agent, and small ML scripts.

## What's here

| Path | What it is |
|---|---|
| `remote-ollama-web/` | A browser interface for the remote Ollama + MCP coding agent. See its own [README](remote-ollama-web/README.md) |
| `agent-workspace/` | The MCP coding agent: `server.py` (tools), `client.py` and `ai.py` (Ollama-driven agent) |
| `mcp-agent/`, `workspace/` | Earlier versions of the MCP agent |
| `ai.py`, `client.py`, `server.py`, `m.py`, `mcpi.py` | Stand-alone MCP server and client experiments |
| `hug.py` | Hugging Face experiments |
| `jaxy.py` | A JAX / Flax NNX sharding experiment |
| `tabpfn.py` | A TabPFN tabular-model experiment |
| `1st/` | First steps: scikit-learn datasets and a small neural net |

## Setup

```bash
python -m venv ~/.venv && source ~/.venv/bin/activate
pip install mcp ollama numpy scikit-learn
```

Each experiment has its own dependencies, so install the ones its imports
need. For the web agent, follow `remote-ollama-web/README.md`.

Credentials (Ollama URL and API key, web-agent password) are read from
environment variables and `~/.config/remote-ollama/env`. They are never
stored in this repo.
