#!/usr/bin/env python3
"""Local Coding Agent - bounded Ollama + MCP coding agent.

Usage:
    python client.py                     # pick a model from a list
    python client.py --model qwen2.5:7b  # skip the menu
    python client.py --lite              # fewer tools: better for small models
    python client.py --ctx 16384         # context window (default 8192)

Chat commands:  /help
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import time

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).resolve().with_name("server.py")
MAX_TOOL_ROUNDS = 12
MODEL_TIMEOUT_SECONDS = 180
MAX_SAME_TOOL_CALLS = 2
MAX_TOOL_CHARS = 8000          # what the model sees from one tool result
DESTRUCTIVE = {"delete_file", "delete_directory", "git_commit", "restore_checkpoint"}
RISKY_CMD = re.compile(r"\b(rm|mv|git\s+(reset|clean|checkout)|truncate|drop\s+table|pip\s+uninstall|npm\s+uninstall)\b")

LITE_TOOLS = {
    "create_project", "switch_project", "list_projects", "project_overview", "list_files",
    "project_tree", "read_file", "write_file", "replace_text", "search_files", "run_command",
    "run_tests", "run_build", "install_dependencies", "start_process", "process_output",
    "stop_process", "http_request", "git_status", "git_diff", "git_commit", "manage_tasks",
    "check_syntax", "delete_file",
}

CORE_TOOLS = {
    # project/workspace
    "create_project", "switch_project", "list_projects", "project_overview",
    # inspect/edit
    "list_files", "project_tree", "read_file", "write_file", "append_file",
    "replace_text", "replace_lines", "insert_lines", "search_files",
    # execute/verify
    "run_command", "run_python", "run_tests", "run_build", "check_syntax",
    "install_dependencies",
    # servers / APIs
    "start_process", "process_output", "stop_process", "http_request",
    # source control / planning
    "git_status", "git_diff", "git_create_branch", "git_commit", "manage_tasks",
    # packaging
    "package_project",
}

USE_COLOR = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


SYSTEM_PROMPT = """You are a careful, autonomous software engineer running on the user's own machine.
You work ONLY through the provided tools. Every project lives in its own folder in the agent workspace;
all file tools operate inside the ACTIVE project. Use relative paths (src/app.py), never absolute paths.

STEP 0 - UNDERSTAND THE REQUEST
- Restate the goal in 1-3 lines. Decide the stack and the main features yourself using sensible defaults.
- Ask the user questions ONLY when you cannot proceed without the answer (max 3 short questions, with suggested defaults).
- Vague request ("make a todo app")? Pick the simplest good design and say your assumptions in one line.
- Questions about code or concepts: answer directly; use tools only when you need to look at files.

HOW TO WORK
1. Starting from scratch: create_project (templates: python, flask, node, web, cpp, empty), then manage_tasks to add
   3-8 short tasks. Existing work: switch_project, then project_overview.
2. Before editing a file: read_file it. Small edits: replace_text / replace_lines / insert_lines. New or fully
   rewritten files: write_file (always give COMPLETE content, never placeholders like "...rest of code").
3. Python needs packages: install_dependencies(packages="flask requests") - it uses the project's own .venv.
   Node: install_dependencies(packages="express").
4. After changes: run_tests. Write tests when a project has none. Then run_build when the project has a build.
5. Servers/watchers (flask, npm start, http.server): start_process, then http_request (use wait_seconds=10),
   process_output to read logs, and ALWAYS stop_process when finished.
6. If something fails: read the error, find the root cause (read_file / search_files / process_output), fix it,
   run the same check again. Repeat until it passes or you are truly blocked. Never repeat the exact same failing
   call - change something.
7. Before risky or large refactors: create_checkpoint. Undo with restore_checkpoint.
8. When the user wants the result delivered: run_build, then package_project (zip in the workspace 'exports' folder).
9. git_commit only when the user asks. Never git push.

HONESTY
- Never claim something works unless a tool result shows it (RESULT: SUCCESS, passing tests, HTTP 200).
- Never invent file contents, command output or test results.
- Finish with a short summary: what you built/changed, which checks passed or failed, and how to run it.

TOOL TIPS
- Tool results start with RESULT: SUCCESS or RESULT: FAILED for commands.
- Tool arguments must match the schema exactly. Numbers as numbers, booleans as true/false.
"""

NEW_TEMPLATE = """NEW PROJECT REQUEST:
{idea}

Plan: (1) restate the requirement in 2-4 lines and list your assumptions - choose sensible defaults, ask only if truly blocked.
(2) create_project with the best template. (3) manage_tasks add a short task list.
(4) Implement step by step; install_dependencies; write tests; run_tests; run_build if applicable; fix until green.
(5) If it is a server or web app, start_process and verify with http_request, then stop_process.
(6) Finish with a summary and exact commands to run it."""

SHORTCUTS = {
    "/test": "Run the tests. If anything fails, find the root cause, fix it, and rerun until they pass.",
    "/build": "Run the build. Fix any errors and rebuild until it succeeds, then tell me where the output is.",
    "/review": "Review this project: read the key files, run the linter and tests, list real problems, and fix the important ones.",
    "/package": "Run the build if the project has one, then package the project with package_project and tell me the zip path.",
}

HELP = """Commands
  /new <idea>      build a new project end to end from an idea
  /fix <problem>   debug a problem: reproduce, find the cause, fix, verify
  /test /build /review /package    run those workflows
  /project         show the active project overview     /projects   list projects
  /model           switch model                          /tools      list tools
  /confirm [destructive|all|none]  when to ask before running actions (default: destructive)
  /ctx <tokens>    set context window (current: {ctx})    /think on|off|auto
  /clear           forget the conversation               /exit       quit
Multi-line input: end a line with \\ to continue on the next line."""


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def list_models():
    try:
        resp = ollama.list()
    except Exception as e:  # noqa: BLE001
        print(f"Cannot reach Ollama: {e}\nStart it with:  ollama serve")
        sys.exit(1)
    rows = []
    for m in resp.models:
        name = getattr(m, "model", None) or getattr(m, "name", "")
        caps = None
        try:
            caps = getattr(ollama.show(name), "capabilities", None)
        except Exception:  # noqa: BLE001
            pass
        supports = None if caps is None else ("tools" in caps)
        size = getattr(m, "size", 0) or 0
        rows.append((name, supports, size))
    return rows


def choose_model(preselected=None):
    rows = list_models()
    if not rows:
        print("No Ollama models found. Try:  ollama pull qwen2.5:7b")
        sys.exit(1)
    if preselected:
        match = [r for r in rows if r[0] == preselected or r[0].split(":")[0] == preselected]
        if not match:
            print(f"Model '{preselected}' not installed. Installed: {', '.join(r[0] for r in rows)}")
            sys.exit(1)
        return match[0][0]
    print(c("\n==== SELECT MODEL ====", "1"))
    for i, (name, supports, size) in enumerate(rows, 1):
        tag = {True: c("tools ok", "32"), False: c("no tool support", "31"), None: "tools ?"}[supports]
        print(f"{i:>2}. {name:<22} {size / 1e9:>5.1f} GB   {tag}")
    print("Tip: 7-8B tool-capable models (qwen2.5:7b, qwen3:8b, llama3.1:8b) work far better than 1B models.")
    while True:
        raw = input("\nModel number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            name, supports, _ = rows[int(raw) - 1]
            if supports is False:
                if input(c(f"{name} does not support tools; the agent cannot work. Use anyway? [y/N] ", "33")).lower() != "y":
                    continue
            return name
        print("Enter one of the numbers above.")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def slim_schema(node):
    """Drop 'title' noise from JSON schemas to save context tokens."""
    if isinstance(node, dict):
        return {k: slim_schema(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [slim_schema(v) for v in node]
    return node


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.4)
    return text[:head] + f"\n...[{len(text) - limit} chars omitted]...\n" + text[-(limit - head):]


def msg_size(m: dict) -> int:
    return len(str(m.get("content", ""))) + len(str(m.get("tool_calls", "")))


def trim_history(messages: list, budget_chars: int, keep_tail: int = 8) -> None:
    """Keep the conversation inside the context window."""
    def total():
        return sum(msg_size(m) for m in messages)

    if total() <= budget_chars:
        return
    for m in messages[1:-keep_tail]:
        if m["role"] == "tool" and len(m["content"]) > 400:
            m["content"] = m["content"][:300] + "\n...[older output trimmed]"
    while total() > budget_chars:
        users = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(users) < 2:
            break
        del messages[users[0]:users[1]]  # drop the oldest complete turn


def extract_text_call(content: str, names: set):
    """Some small models print the tool call as JSON text instead of using tool_calls."""
    s = content.strip()
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", s, re.S)
    if m:
        s = m.group(1)
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    if not s.startswith("{"):
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("tool")
    args = obj.get("arguments") or obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if name in names and isinstance(args, dict):
        return name, args
    return None


def clean_args(schema: dict, args) -> tuple:
    """Keep only known args; report missing required ones."""
    props = (schema or {}).get("properties", {})
    required = (schema or {}).get("required", [])
    args = dict(args or {})
    cleaned = {k: v for k, v in args.items() if k in props and v is not None}
    missing = [k for k in required if k not in cleaned]
    return cleaned, missing


async def call_tool(session: ClientSession, name: str, args: dict) -> str:
    try:
        res = await session.call_tool(name, args)
    except Exception as e:  # noqa: BLE001
        return f"Tool execution error: {type(e).__name__}: {e}"
    parts = [getattr(part, "text", "") for part in res.content if getattr(part, "text", "")]
    out = "\n".join(parts).strip() or "Tool completed with no output."
    is_error = getattr(res, "is_error", getattr(res, "isError", False))
    return ("TOOL ERROR: " + out) if is_error else out


def needs_confirm(mode: str, name: str, args: dict) -> bool:
    if mode == "none":
        return False
    if mode == "all":
        return True
    if name in DESTRUCTIVE:
        return True
    return name in ("run_command", "start_process") and bool(RISKY_CMD.search(str(args.get("command", ""))))


def ask_yes(prompt: str) -> bool:
    try:
        return input(c(prompt + " [y/N] ", "33")).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def read_input() -> str:
    lines = []
    prompt = c("\nYou: ", "36;1")
    while True:
        line = input(prompt if not lines else "...  ")
        if line.endswith("\\"):
            lines.append(line[:-1])
            continue
        lines.append(line)
        return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Model streaming
# --------------------------------------------------------------------------
def stream_chat(model, messages, tools, options, think):
    """Run one Ollama turn with a hard wall-clock timeout.

    The old implementation could wait forever on a slow local model.  This
    version uses a bounded Ollama HTTP timeout and stops consuming the stream
    after MODEL_TIMEOUT_SECONDS.
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "options": options,
        "stream": True,
    }
    if think is not None:
        kwargs["think"] = think

    client = ollama.Client(timeout=MODEL_TIMEOUT_SECONDS)
    content, calls = "", []
    printed_header = False
    thinking_shown = False
    started = time.monotonic()

    try:
        stream = client.chat(**kwargs)
        for chunk in stream:
            if time.monotonic() - started > MODEL_TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"Ollama turn exceeded {MODEL_TIMEOUT_SECONDS}s. "
                    "Use a smaller model, lower context, or send /think off."
                )
            m = chunk.message
            thought = getattr(m, "thinking", None)
            if thought:
                if think is True:
                    sys.stdout.write(c(thought, "2"))
                    sys.stdout.flush()
                elif not thinking_shown:
                    sys.stdout.write(c("\n(thinking...)", "2"))
                    sys.stdout.flush()
                    thinking_shown = True
            if m.content:
                if not printed_header:
                    sys.stdout.write("\n" + c("AI: ", "35;1"))
                    printed_header = True
                sys.stdout.write(m.content)
                sys.stdout.flush()
                content += m.content
            for tc in (m.tool_calls or []):
                raw = getattr(tc.function, "arguments", None)
                if isinstance(raw, dict):
                    args = raw
                elif isinstance(raw, str):
                    try:
                        args = json.loads(raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    try:
                        args = dict(raw or {})
                    except (TypeError, ValueError):
                        args = {}
                calls.append((tc.function.name, args))
    finally:
        if printed_header:
            print()
    return content, calls


# --------------------------------------------------------------------------
# Main agent loop
# --------------------------------------------------------------------------
class Agent:
    def __init__(self, session, tools, model, ctx, confirm, think):
        self.session = session
        self.model = model
        self.ctx = ctx
        self.confirm = confirm
        self.think = think
        self.schemas = {t.name: slim_schema(getattr(t, "input_schema", getattr(t, "inputSchema", {}))) for t in tools}
        self.ollama_tools = [
            {"type": "function", "function": {
                "name": t.name, "description": t.description or "", "parameters": self.schemas[t.name]}}
            for t in tools
        ]
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def refresh_state(self):
        overview = await call_tool(self.session, "project_overview", {})
        self.messages[0]["content"] = (
            SYSTEM_PROMPT + "\n\nCURRENT WORKSPACE STATE (auto-refreshed each turn):\n" + clip(overview, 3500)
        )

    def reset(self):
        self.messages = self.messages[:1]

    async def turn(self, user_text: str):
        await self.refresh_state()
        self.messages.append({"role": "user", "content": user_text})
        names = set(self.schemas)
        last_sig, repeats = None, 0

        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            trim_history(self.messages, int(self.ctx * 2.4))
            options = {
                "num_ctx": self.ctx,
                "temperature": 0.1,
                "num_predict": 1536,
            }
            print(c(f"\n[agent round {round_no}/{MAX_TOOL_ROUNDS}]", "90"))
            try:
                content, calls = await asyncio.to_thread(
                    stream_chat, self.model, self.messages, self.ollama_tools, options, self.think
                )
            except TimeoutError as e:
                print(c(f"\n[STOPPED: {e}]", "31"))
                self.messages.append({
                    "role": "user",
                    "content": "The previous model turn timed out. Do not repeat the same action. Continue with a smaller, concrete next step or report the blocker."
                })
                return
            except ollama.ResponseError as e:
                text = str(e)
                print(c(f"\nOllama error: {text}", "31"))
                return
            except KeyboardInterrupt:
                print(c("\n[interrupted]", "33"))
                return
            except Exception as e:
                print(c(f"\nModel error: {type(e).__name__}: {e}", "31"))
                return

            if not calls and content:
                fallback = extract_text_call(content, names)
                if fallback:
                    calls, content = [fallback], ""

            assistant = {"role": "assistant", "content": content}
            if calls:
                assistant["tool_calls"] = [
                    {"function": {"name": n, "arguments": a}} for n, a in calls
                ]
            self.messages.append(assistant)

            if not calls:
                return

            # Execute calls sequentially. Never let the model silently repeat
            # an identical tool invocation forever.
            for name, raw_args in calls[:3]:
                sig = (name, json.dumps(raw_args, sort_keys=True, default=str))
                repeats = repeats + 1 if sig == last_sig else 1
                last_sig = sig
                if repeats > MAX_SAME_TOOL_CALLS:
                    print(c(f"\n[STOPPED: repeated identical tool call: {name}]", "31"))
                    self.messages.append({
                        "role": "tool",
                        "tool_name": name,
                        "content": "STOPPED: this exact tool call was repeated. Read the previous result and choose a different action."
                    })
                    return
                result = await self.run_one(name, raw_args)
                self.messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": clip(result, MAX_TOOL_CHARS),
                })

        print(c("\n[STOPPED: maximum agent rounds reached. No infinite loop.]", "31"))

    async def run_one(self, name: str, raw_args) -> str:
        if name not in self.schemas:
            return f"Unknown tool '{name}'. Available tools: {', '.join(sorted(self.schemas))}"
        args, missing = clean_args(self.schemas[name], raw_args)
        short = json.dumps(args, ensure_ascii=False, default=str)
        print(c(f"\n[tool] {name} {clip(short, 300)}", "33"))
        if missing:
            msg = f"Missing required argument(s): {', '.join(missing)}. Expected: {json.dumps(self.schemas[name].get('properties', {}))}"
            print(c(f"  -> {msg}", "31"))
            return msg
        if needs_confirm(self.confirm, name, args) and not ask_yes(f"  Allow {name}?"):
            print(c("  -> denied", "31"))
            return "The user DENIED this action. Do not retry it; explain what you wanted and ask how to proceed."
        result = await call_tool(self.session, name, args)
        preview = clip(result.strip(), 700).replace("\n", "\n     ")
        failed = "RESULT: FAILED" in result or result.startswith(("ERROR", "TOOL ERROR", "BLOCKED"))
        print(c("  -> " + preview, "31" if failed else "2"))
        return result


async def amain(args):
    model = choose_model(args.model)
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},  # pass the full env (default env drops PATH/venv vars)
    )
    print("\nConnecting to MCP server...")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            all_tools = (await session.list_tools()).tools
            if args.all_tools:
                tools = list(all_tools)
                mode = "all"
            elif args.lite:
                tools = [t for t in all_tools if t.name in LITE_TOOLS]
                mode = "lite"
            else:
                tools = [t for t in all_tools if t.name in CORE_TOOLS]
                mode = "core"
            agent = Agent(session, tools, model, args.ctx, args.confirm, {"on": True, "off": False}.get(args.think))
            print(c("\n==== LOCAL CODING AGENT ====", "1"))
            print(f"Model: {model}   Tools: {len(tools)}/{len(all_tools)} ({mode})   Context: {args.ctx}   Confirm: {args.confirm}   Think: {args.think}")
            print("Type /help for commands, /exit to quit.\n")
            print(await call_tool(session, "list_projects", {}))

            while True:
                try:
                    text = read_input()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not text:
                    continue
                low = text.lower()
                if low in ("exit", "quit", "/exit", "/quit"):
                    break
                if low == "/help":
                    print(HELP.format(ctx=agent.ctx))
                    continue
                if low == "/tools":
                    print(", ".join(sorted(agent.schemas)))
                    continue
                if low == "/clear":
                    agent.reset()
                    print("Conversation cleared.")
                    continue
                if low == "/project":
                    print(await call_tool(session, "project_overview", {}))
                    continue
                if low == "/projects":
                    print(await call_tool(session, "list_projects", {}))
                    continue
                if low == "/model":
                    agent.model = choose_model()
                    print(f"Now using {agent.model}")
                    continue
                if low.startswith("/ctx"):
                    parts = text.split()
                    if len(parts) == 2 and parts[1].isdigit() and 2048 <= int(parts[1]) <= 131072:
                        agent.ctx = int(parts[1])
                        print(f"Context window set to {agent.ctx}")
                    else:
                        print("Usage: /ctx 16384   (2048 to 131072)")
                    continue
                if low.startswith("/confirm"):
                    parts = text.split()
                    if len(parts) == 2 and parts[1] in ("destructive", "all", "none"):
                        agent.confirm = parts[1]
                        print(f"Confirmation mode: {agent.confirm}")
                    else:
                        print(f"Usage: /confirm destructive|all|none   (now: {agent.confirm})")
                    continue
                if low.startswith("/think"):
                    parts = text.split()
                    mapping = {"on": True, "off": False, "auto": None}
                    if len(parts) == 2 and parts[1] in mapping:
                        agent.think = mapping[parts[1]]
                        print(f"Thinking mode: {parts[1]}")
                    else:
                        print("Usage: /think on|off|auto")
                    continue
                if low.startswith("/new"):
                    idea = text[4:].strip()
                    if not idea:
                        print("Usage: /new <describe what you want built>")
                        continue
                    text = NEW_TEMPLATE.format(idea=idea)
                elif low.startswith("/fix"):
                    problem = text[4:].strip() or "the current failure in this project"
                    text = (f"Debug this: {problem}\nReproduce it first, find the root cause, fix it, then "
                            "rerun to verify and run the tests.")
                elif low.split()[0] in SHORTCUTS:
                    text = SHORTCUTS[low.split()[0]]
                elif low.startswith("/"):
                    print("Unknown command. Type /help.")
                    continue
                await agent.turn(text)


def main():
    ap = argparse.ArgumentParser(description="Local coding agent (Ollama + MCP)")
    ap.add_argument("--model", help="Ollama model name, skips the menu")
    ap.add_argument("--lite", action="store_true", help="expose the smallest core tool set")
    ap.add_argument("--all-tools", action="store_true", help="expose all MCP tools; default uses a focused coding set")
    ap.add_argument("--ctx", type=int, default=8192, help="context window tokens (default 8192)")
    ap.add_argument("--confirm", choices=["destructive", "all", "none"], default="destructive",
                    help="when to ask before running a tool")
    ap.add_argument("--think", choices=["auto", "on", "off"], default="off",
                    help="thinking mode for reasoning models")
    args = ap.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
