#!/usr/bin/env python3
"""
ollama_coding_mcp.py
A local, Ollama-powered coding MCP server AND headless autonomous agent.

RUN AS AN MCP SERVER (stdio) -- for Claude Desktop / Cursor / any MCP host:
    python ollama_coding_mcp.py

RUN AS A HEADLESS AGENT (no host, no human, unattended):
    python ollama_coding_mcp.py --agent "Add input validation to api.py and run the tests"
    python ollama_coding_mcp.py --agent "Fix the failing tests" --repo myproject --steps 15

Requirements:
    pip install mcp          # the MCP Python SDK
    ollama serve             # Ollama running locally

NOTE: In stdio mode, stdout is the MCP wire protocol -- nothing is ever printed
there. All diagnostics go to stderr. Do not add print() to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


# --------------------------------------------------------------------------- #
#  Logging (stderr only -- keeps the stdio MCP stream clean)
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[ollama-coding-mcp] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Every file/command operation is confined inside this directory.
WORKSPACE = Path(os.environ.get("DEV_MCP_WORKSPACE", os.getcwd())).expanduser().resolve()

# Set DEV_MCP_ALLOW_ALL=1 to skip the command allowlist (deny-list + sandbox stay on).
ALLOW_ALL = os.environ.get("DEV_MCP_ALLOW_ALL", "0") == "1"

# Model routing -- point each ROLE at one of YOUR installed models (`ollama list`).
MODEL_ROUTES = {
    "chat":   os.environ.get("MODEL_CHAT",   "qwen2.5:3b"),      # general Q&A
    "code":   os.environ.get("MODEL_CODE",   "qwen2.5:3b"),      # recommend: qwen2.5-coder:3b
    "reason": os.environ.get("MODEL_REASON", "deepseek-r1:8b"),   # planning / debugging
    "tiny":   os.environ.get("MODEL_TINY",   "qwen2.5:0.5b"),     # fast throwaway calls
}
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

# Command allowlist (first token must match). Extend deliberately.
ALLOWED_COMMANDS = {
    "python", "python3", "pytest", "pip", "pip3", "make", "cargo", "rustc",
    "go", "npm", "npx", "node", "yarn", "pnpm", "bun", "tsc", "deno",
    "ruff", "black", "mypy", "flake8", "isort", "eslint", "prettier",
    "gcc", "g++", "clang", "clang++", "cmake", "git", "ls", "cat", "find",
    "grep", "rg", "head", "tail", "wc", "sed", "awk", "sort", "diff",
    "echo", "mkdir", "cp", "mv", "touch",
}

# Always-blocked patterns, even with ALLOW_ALL.
DENY_PATTERNS = [
    "rm -rf /", ":(){", "mkfs", "dd if=", "sudo ", "shutdown", "reboot",
    "chmod -R 777 /", "> /dev/sda", "curl | sh", "curl|sh", "wget | sh",
    "wget|sh", "git push", ":> /", "fork bomb",
]

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
               "dist", "build", ".mypy_cache", ".pytest_cache", ".next", "target"}


# --------------------------------------------------------------------------- #
#  Ollama client (stdlib only -- no extra dependency)
# --------------------------------------------------------------------------- #
def _ollama_chat(model: str, messages: list, *, system: str | None = None,
                 temperature: float = 0.2, num_ctx: int | None = None,
                 timeout: int = 900) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    payload = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx or NUM_CTX},
    }
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running? ({exc})"
        ) from exc
    return (data.get("message", {}).get("content") or "").strip()


def _ask(role: str, prompt: str, *, system: str | None = None,
         temperature: float = 0.2) -> str:
    model = MODEL_ROUTES.get(role, MODEL_ROUTES["chat"])
    return _ollama_chat(model, [{"role": "user", "content": prompt}],
                        system=system, temperature=temperature)


def _list_models() -> list[str]:
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}") from exc
    return [m.get("name", "?") for m in data.get("models", [])]


# --------------------------------------------------------------------------- #
#  Sandbox + safety
# --------------------------------------------------------------------------- #
def _safe_path(rel: str) -> Path:
    """Resolve `rel` inside WORKSPACE; refuse anything that escapes it."""
    p = (WORKSPACE / rel).resolve()
    if p != WORKSPACE and WORKSPACE not in p.parents:
        raise ValueError(f"Refusing to touch a path outside the sandbox: {rel!r}")
    return p


def _guard_command(command: str) -> None:
    low = " ".join(command.lower().split())
    for bad in DENY_PATTERNS:
        if bad in low:
            raise ValueError(f"Blocked by safety policy: {bad!r}")
    if ALLOW_ALL:
        return
    stripped = command.strip()
    if not stripped:
        raise ValueError("Empty command.")
    base = os.path.basename(stripped.split()[0])
    if base not in ALLOWED_COMMANDS:
        raise ValueError(
            f"Command {base!r} is not in ALLOWED_COMMANDS. Add it, or run with "
            f"DEV_MCP_ALLOW_ALL=1 (deny-list + sandbox still apply)."
        )


# --------------------------------------------------------------------------- #
#  Core operations (shared by MCP tools AND the headless agent)
# --------------------------------------------------------------------------- #
def op_list_files(subdir: str = ".", pattern: str = "*") -> str:
    root = _safe_path(subdir)
    if not root.exists():
        return f"(no such path: {subdir})"
    out = []
    for p in sorted(root.rglob("*")):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        rel = p.relative_to(WORKSPACE).as_posix()
        if p.is_dir():
            out.append(f"{rel}/")
        elif fnmatch(p.name, pattern):
            out.append(f"{rel}  ({p.stat().st_size} B)")
    return "\n".join(out[:400]) or "(no matches)"


def op_read_file(path: str, max_bytes: int = 200_000) -> str:
    p = _safe_path(path)
    if not p.is_file():
        return f"(not a file: {path})"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_bytes:
        return text[:max_bytes] + f"\n... [truncated at {max_bytes} bytes]"
    return text


def op_write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {p.relative_to(WORKSPACE)}"


def op_search_code(query: str, subdir: str = ".", pattern: str = "*") -> str:
    root = _safe_path(subdir)
    hits = []
    for p in sorted(root.rglob(pattern)):
        if any(part in IGNORE_DIRS for part in p.parts) or not p.is_file():
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if query in line:
                    hits.append(f"{p.relative_to(WORKSPACE)}:{i}: {line.strip()[:200]}")
        except Exception:
            continue
        if len(hits) > 200:
            hits.append("... (truncated)")
            break
    return "\n".join(hits) or f"(no matches for {query!r})"


def op_run_command(command: str, cwd: str = ".", timeout: int = 300) -> str:
    _guard_command(command)
    workdir = _safe_path(cwd)
    try:
        proc = subprocess.run(command, shell=True, cwd=str(workdir),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s: {command}"
    out = proc.stdout or ""
    if proc.stderr:
        out += f"\n[stderr]\n{proc.stderr}"
    return f"exit={proc.returncode}\n{out.strip()[:12000]}"


def op_run_tests(command: str | None = None, cwd: str = ".") -> str:
    if command:
        return op_run_command(command, cwd)
    root = _safe_path(cwd)
    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() \
            or (root / "tests").is_dir() or any(root.glob("test_*.py")):
        return op_run_command("python -m pytest -q", cwd)
    if (root / "package.json").exists():
        return op_run_command("npm test --silent", cwd)
    if (root / "Cargo.toml").exists():
        return op_run_command("cargo test", cwd)
    if (root / "go.mod").exists():
        return op_run_command("go test ./...", cwd)
    return "No test runner auto-detected; call run_command with an explicit command."


def op_git_checkpoint(message: str = "agent checkpoint") -> str:
    status = op_run_command("git status --porcelain")
    if "exit=0" not in status or not status.split("\n", 1)[-1].strip():
        return "Nothing to commit (or not a git repo)."
    op_run_command("git add -A")
    safe_msg = re.sub(r'["\n]', " ", message)[:120]
    return op_run_command(f'git commit -m "{safe_msg}"')


def op_repo_snapshot(subdir: str = ".", max_files: int = 80) -> str:
    root = _safe_path(subdir)
    lines, count = [], 0
    for p in sorted(root.rglob("*")):
        if any(part in IGNORE_DIRS for part in p.parts) or p.is_dir():
            continue
        count += 1
        if count > max_files:
            lines.append(f"... ({count}+ files, truncated)")
            break
        lines.append(f"  {p.relative_to(WORKSPACE)}  ({p.stat().st_size} B)")
    return "\n".join(lines) or "(empty workspace)"


# --------------------------------------------------------------------------- #
#  Prompt templates for the coding roles
# --------------------------------------------------------------------------- #
SYS_CODE = ("You are a precise senior software engineer. Output only the "
            "requested code or answer. No pleasantries.")

PLAN_PROMPT = """Produce a short, concrete plan (max 8 bullets) to accomplish the task.
No code. Name the exact files to change and the verification step.

TASK: {task}

REPO FILES:
{context}
"""

AGENT_SYSTEM = """You are an autonomous coding agent inside a sandboxed repository.
You operate in a loop: choose exactly ONE action, it is executed, you see the
result, then choose the next action.

Respond with a SINGLE JSON object and NOTHING else. No prose, no markdown fences.

{"thought": "<one sentence>", "action": "<action>", "args": {...}}

Actions and their args:
  read_file    {"path": "<relative path>"}
  write_file   {"path": "<relative path>", "content": "<full new file content>"}
  list_files   {"subdir": ".", "pattern": "*"}
  search_code  {"query": "<text>"}
  run_command  {"command": "<shell command>", "cwd": "."}
  run_tests    {"cwd": "."}
  finish       {"summary": "<what you did and the result>"}

Rules:
- Read a file before you rewrite it.
- Make small, verifiable changes.
- Run the tests after changing code.
- Only use "finish" when the task is genuinely complete and verified.
"""


# --------------------------------------------------------------------------- #
#  JSON extraction tolerant of small-model sloppiness
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _execute_action(action: str, args: dict) -> str:
    if action == "read_file":
        return op_read_file(args.get("path", ""))
    if action == "write_file":
        return op_write_file(args.get("path", ""), args.get("content", ""))
    if action == "list_files":
        return op_list_files(args.get("subdir", "."), args.get("pattern", "*"))
    if action == "search_code":
        return op_search_code(args.get("query", ""))
    if action == "run_command":
        return op_run_command(args.get("command", ""), args.get("cwd", "."))
    if action == "run_tests":
        return op_run_tests(cwd=args.get("cwd", "."))
    return f"Unknown action: {action!r}"


# --------------------------------------------------------------------------- #
#  The autonomous loop (one call does the whole job -- no human in the middle)
# --------------------------------------------------------------------------- #
def op_dev_agent(task: str, subdir: str = ".", max_steps: int = 12,
                 checkpoint: bool = True) -> str:
    log: list[str] = []
    if checkpoint:
        log.append("checkpoint: " + op_git_checkpoint(f"agent: {task[:60]}"))

    context = op_repo_snapshot(subdir)
    try:
        plan = _ollama_chat(
            MODEL_ROUTES["reason"],
            [{"role": "user", "content": PLAN_PROMPT.format(task=task, context=context)}],
            system="You are a pragmatic senior software engineer.",
            temperature=0.2,
        )
    except RuntimeError as exc:
        return f"Agent aborted: {exc}"
    log.append(f"== PLAN ({MODEL_ROUTES['reason']}) ==\n{plan}")

    history = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": f"TASK: {task}\n\nPLAN:\n{plan}\n\n"
                                    f"Begin. Respond with ONE JSON action."},
    ]

    for step in range(1, max_steps + 1):
        raw = _ollama_chat(MODEL_ROUTES["code"], history, temperature=0.1)
        parsed = _extract_json(raw)
        if parsed is None:
            history.append({"role": "assistant", "content": raw})
            history.append({"role": "user",
                            "content": "Invalid output. Reply with ONE JSON object only."})
            log.append(f"[{step}] (unparseable output, retrying)")
            continue

        action = parsed.get("action", "")
        args = parsed.get("args") or {}
        log.append(f"[{step}] {parsed.get('thought', '').strip()}\n"
                   f"      -> {action} {json.dumps(args)[:300]}")

        if action == "finish":
            log.append(f"\n== DONE ==\n{args.get('summary', '(no summary)')}")
            return "\n".join(log)

        try:
            observation = _execute_action(action, args)
        except Exception as exc:                       # noqa: BLE001
            observation = f"ERROR: {exc}"
        log.append("      <- " + observation.replace("\n", "\n         ")[:1500])

        history.append({"role": "assistant", "content": json.dumps(parsed)})
        history.append({"role": "user",
                        "content": f"OBSERVATION:\n{observation}\n\nNext action (one JSON):"})
        # Keep the context bounded: always keep system + first user, then last 8.
        if len(history) > 11:
            history = history[:2] + history[-8:]

    log.append(f"\n== STOPPED: reached max_steps={max_steps} without finish ==")
    return "\n".join(log)


# --------------------------------------------------------------------------- #
#  MCP wiring
# --------------------------------------------------------------------------- #
try:
    from mcp.server.fastmcp import FastMCP   # SDK import path (verify if it moved)
except ImportError:                           # pragma: no cover
    _log("The 'mcp' package is not installed. Run: pip install mcp")
    sys.exit(1)

mcp = FastMCP("ollama-coding")


@mcp.tool()
def list_local_models() -> str:
    """List the Ollama models available on this machine."""
    return "\n".join(_list_models())


@mcp.tool()
def ask(prompt: str, role: str = "chat") -> str:
    """Ask a local model directly. role is one of: chat, code, reason, tiny."""
    return _ask(role, prompt)


@mcp.tool()
def generate_code(task: str, language: str = "python", context: str = "") -> str:
    """Generate code for a task. `context` can hold existing code or constraints."""
    prompt = f"Language: {language}\nTask: {task}\n"
    if context:
        prompt += f"\nExisting context:\n{context}\n"
    prompt += "\nReturn only the code, in a single fenced block."
    return _ask("code", prompt, system=SYS_CODE, temperature=0.2)


@mcp.tool()
def review_code(code: str, language: str = "python") -> str:
    """Review code and return a prioritized list of concrete issues."""
    prompt = (f"Review this {language} code. List bugs, security issues, and "
              f"edge cases by severity. Be specific.\n\n{code}")
    return _ask("reason", prompt)


@mcp.tool()
def explain_code(code: str, language: str = "python") -> str:
    """Explain what a piece of code does, step by step."""
    return _ask("reason", f"Explain this {language} code clearly:\n\n{code}")


@mcp.tool()
def refactor_code(code: str, instructions: str, language: str = "python") -> str:
    """Refactor code according to instructions; returns the rewritten code."""
    prompt = (f"Refactor this {language} code per the instructions. "
              f"Return only the new code.\n\nInstructions: {instructions}\n\n{code}")
    return _ask("code", prompt, system=SYS_CODE)


@mcp.tool()
def plan_task(task: str, subdir: str = ".") -> str:
    """Produce an implementation plan for a task, given the repo file tree."""
    return _ask("reason", PLAN_PROMPT.format(task=task, context=op_repo_snapshot(subdir)))


@mcp.tool()
def list_files(subdir: str = ".", pattern: str = "*") -> str:
    """List files in the sandbox workspace (relative paths)."""
    return op_list_files(subdir, pattern)


@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file inside the sandbox workspace."""
    return op_read_file(path)


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write full content to a file in the sandbox (creates parent dirs)."""
    return op_write_file(path, content)


@mcp.tool()
def search_code(query: str, subdir: str = ".", pattern: str = "*") -> str:
    """Grep for a substring across the sandbox workspace."""
    return op_search_code(query, subdir, pattern)


@mcp.tool()
def run_command(command: str, cwd: str = ".", timeout: int = 300) -> str:
    """Run a shell command inside the sandbox, subject to the safety policy."""
    return op_run_command(command, cwd, timeout)


@mcp.tool()
def run_tests(command: str = "", cwd: str = ".") -> str:
    """Run the project's tests (auto-detects pytest/npm/cargo/go if command empty)."""
    return op_run_tests(command or None, cwd)


@mcp.tool()
def git_status() -> str:
    """Show git status --porcelain for the sandbox workspace."""
    return op_run_command("git status --porcelain")


@mcp.tool()
def git_diff(args: str = "") -> str:
    """Show a git diff. `args` may add options, e.g. '--staged' or 'HEAD~1'."""
    return op_run_command(f"git diff {args}".strip())


@mcp.tool()
def git_checkpoint(message: str = "agent checkpoint") -> str:
    """Commit all current changes as a safety checkpoint before agent work."""
    return op_git_checkpoint(message)


@mcp.tool()
def dev_agent(task: str, subdir: str = ".", max_steps: int = 12) -> str:
    """Run the AUTONOMOUS coding agent on `task`: plans, edits files, runs tests,
    and iterates until done -- all within a single call, no human in the loop.
    Takes a git checkpoint first. Returns a full transcript of the run."""
    return op_dev_agent(task, subdir, max_steps)


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def _main() -> None:
    parser = argparse.ArgumentParser(description="Ollama coding MCP server / agent")
    parser.add_argument("--agent", metavar="TASK",
                        help="Run the autonomous agent on TASK headlessly, then exit.")
    parser.add_argument("--repo", default=".", help="Subdir within the workspace.")
    parser.add_argument("--steps", type=int, default=12, help="Max agent steps.")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Skip the pre-run git commit.")
    args = parser.parse_args()

    if args.agent:
        result = op_dev_agent(args.agent, args.repo, args.steps,
                              checkpoint=not args.no_checkpoint)
        print(result)
        return

    _log(f"workspace = {WORKSPACE}")
    _log(f"models    = {MODEL_ROUTES}")
    _log("serving MCP over stdio")
    mcp.run()


if __name__ == "__main__":
    _main()
