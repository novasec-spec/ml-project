#!/usr/bin/env python3
"""Local Coding Agent - MCP server (stdio transport).

Works with mcp 1.x (FastMCP). Falls back to the mcp 2.x MCPServer class.
NEVER print to stdout in this file: stdout is the MCP protocol channel.
Use sys.stderr for any debug output.

Layout:  AGENT_WORKSPACE (default ~/ml-project/agent-workspace)
            <project-a>/   <- one folder per project
            <project-b>/
            .agent/        <- server state (active project, tasks, checkpoints)
            exports/       <- zip packages made by package_project
All file tools work inside the ACTIVE project only.
"""

import ast
import atexit
import difflib
import fnmatch
import functools
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# MCP 2.x is the primary supported line. Keep a v1 fallback for older installs.
try:
    from mcp.server import MCPServer as _Server  # mcp 2.x
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x

mcp = _Server("Local Coding Agent")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ROOT = Path(
    os.environ.get("AGENT_WORKSPACE", str(Path.home() / "ml-project" / "agent-workspace"))
).resolve()
META = ROOT / ".agent"
EXPORTS = ROOT / "exports"
STATE_FILE = META / "state.json"
ROOT.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", "dist", "build", ".next",
    ".expo", "target", ".gradle", "coverage", ".cache", ".agent",
}
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar",
    ".so", ".o", ".a", ".pyc", ".class", ".jar", ".apk", ".aab", ".exe", ".dll",
    ".ttf", ".otf", ".woff", ".woff2", ".mp3", ".mp4", ".wav", ".db", ".sqlite",
}

MAX_READ_CHARS = 60_000
MAX_FILE_BYTES = 2_000_000
MAX_SEARCH_RESULTS = 200
MAX_OUTPUT = 20_000
MAX_TIMEOUT = 900
MAX_TREE = 400
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class ToolError(Exception):
    """A problem the model can understand and fix."""


def tool(fn):
    """Register fn as an MCP tool; turn exceptions into readable error text."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError as e:
            return f"ERROR: {e}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"

    return mcp.tool()(wrapper)


# --------------------------------------------------------------------------
# State / paths
# --------------------------------------------------------------------------
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def project_name() -> str:
    name = _read_json(STATE_FILE, {}).get("project", "")
    if name and NAME_RE.fullmatch(name) and (ROOT / name).is_dir():
        return name
    return ""


def workdir() -> Path:
    name = project_name()
    return ROOT / name if name else ROOT


def safe_path(path: str) -> Path:
    base = workdir()
    path = (path or ".").strip()
    if path.startswith("~"):
        raise ToolError("Use paths relative to the project (like 'src/app.py'), not '~'.")
    target = (base / path).resolve()
    if target != base and base not in target.parents:
        raise ToolError(
            f"Access denied: '{path}' is outside the active project "
            f"({base.name}). Use relative paths like 'src/app.py'."
        )
    return target


def rel(p: Path) -> str:
    try:
        r = str(p.relative_to(workdir()))
        return r if r != "." else "."
    except ValueError:
        return str(p)


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{n}B"


# --------------------------------------------------------------------------
# Command execution
# --------------------------------------------------------------------------
BLOCKED_PATTERNS = [
    (r"\bsudo\b", "sudo is not allowed"),
    (r"\bmkfs(\.|\s)", "formatting disks is not allowed"),
    (r"\b(shutdown|reboot|poweroff|halt)\b", "power commands are not allowed"),
    (r":\(\)\s*\{", "fork bombs are not allowed"),
    (r"\bdd\s+.*\bof=/dev/", "writing to devices is not allowed"),
    (r">\s*/dev/sd", "writing to devices is not allowed"),
    (r"\bchmod\s+-R\s+\S+\s+/(\s|$)", "recursive chmod on / is not allowed"),
    (r"(curl|wget)[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", "piping downloads into a shell is not allowed"),
    (r"\bgit\s+push\b", "git push is not allowed - ask the user to push"),
]
_RM_RE = re.compile(r"\brm\s+((?:-\S+\s+)+)(\S+)")


def command_is_blocked(command: str) -> str:
    for pattern, reason in BLOCKED_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return reason
    for flags, target in _RM_RE.findall(command):
        if "r" not in flags.lower():
            continue
        t = target.strip("'\"")
        if t in ("/", "/*") or t.startswith(("~", "$HOME", "..", "${HOME")):
            return "recursive delete outside the project is not allowed"
        if t.startswith("/") and not t.startswith(str(ROOT)):
            return "recursive delete outside the agent workspace is not allowed"
    return ""


def truncate_output(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.35)
    tail = limit - head
    cut = len(text) - limit
    return text[:head] + f"\n\n...[{cut} characters cut from the middle]...\n\n" + text[-tail:]


def build_env() -> dict:
    env = os.environ.copy()
    env.update(
        CI="1", PYTHONUNBUFFERED="1", NO_COLOR="1", TERM="dumb", GIT_TERMINAL_PROMPT="0",
        GIT_PAGER="cat", PAGER="cat", PIP_DISABLE_PIP_VERSION_CHECK="1",
        npm_config_yes="true", npm_config_update_notifier="false", npm_config_fund="false",
        DEBIAN_FRONTEND="noninteractive",
    )
    env.pop("PYTHONHOME", None)
    venv = workdir() / ".venv"
    if (venv / "bin" / "python").exists():  # prefer the project's own virtualenv
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    return env


def _kill_group(proc: subprocess.Popen, sig=signal.SIGTERM) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_raw(command: str, timeout: int = 120, cwd=None):
    """Run a shell command. Returns (code, stdout, stderr, seconds, timed_out)."""
    timeout = max(1, min(int(timeout), MAX_TIMEOUT))
    started = time.time()
    proc = subprocess.Popen(
        ["/bin/bash", "-c", command],
        cwd=str(cwd or workdir()),
        env=build_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc, signal.SIGKILL)
        out, err = proc.communicate()
    return proc.returncode, out or "", err or "", time.time() - started, timed_out


def run_process(command: str, timeout: int = 120) -> str:
    reason = command_is_blocked(command)
    if reason:
        return (
            f"BLOCKED: {reason}.\nCommand: {command}\n"
            "Do not retry this. If it is really needed, tell the user to run it manually."
        )
    code, out, err, elapsed, timed_out = _run_raw(command, timeout)
    parts = [f"COMMAND: {command}"]
    if timed_out:
        parts.append(f"RESULT: TIMED OUT after {timeout}s (process killed). "
                     "For servers or long-running programs use start_process instead.")
    else:
        parts.append(f"RESULT: {'SUCCESS' if code == 0 else 'FAILED'} (exit code {code})")
    parts.append(f"TIME: {elapsed:.2f}s")
    if out.strip():
        parts.append("STDOUT:\n" + truncate_output(out.rstrip()))
    if err.strip():
        parts.append("STDERR:\n" + truncate_output(err.rstrip()))
    if not out.strip() and not err.strip():
        parts.append("(no output)")
    return "\n\n".join(parts)


def _which(name: str):
    return shutil.which(name, path=build_env().get("PATH"))


def _version(cmd: str) -> str:
    exe = cmd.split()[0]
    if not shutil.which(exe):
        return "not installed"
    code, out, err, _, _ = _run_raw(cmd, 15, cwd=ROOT)
    text = (out or err).strip().splitlines()
    return text[0] if text else "?"


# --------------------------------------------------------------------------
# File helpers
# --------------------------------------------------------------------------
def read_text_file(p: Path) -> str:
    if not p.exists():
        raise ToolError(f"File does not exist: {rel(p)}")
    if not p.is_file():
        raise ToolError(f"Not a file: {rel(p)}")
    if p.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"File is too large ({_human(p.stat().st_size)}).")
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError("File is not UTF-8 text (binary file?).")


def syntax_problem(p: Path, text: str) -> str:
    suffix = p.suffix.lower()
    try:
        if suffix == ".py":
            ast.parse(text, filename=p.name)
        elif suffix == ".json" and not p.name.startswith(("tsconfig", "jsconfig")):
            json.loads(text)
    except SyntaxError as e:
        return f"{p.name} line {e.lineno}: {e.msg}"
    except json.JSONDecodeError as e:
        return f"{p.name} line {e.lineno}: {e.msg}"
    return ""


def _after_edit(p: Path, text: str) -> str:
    problem = syntax_problem(p, text)
    return f"\nWARNING: syntax error -> {problem}. Fix it." if problem else ""


def _snippet(lines, start: int, end: int, pad: int = 2) -> str:
    a = max(1, start - pad)
    b = min(len(lines), end + pad)
    return "\n".join(f"{n:>6}: {lines[n - 1].rstrip()}" for n in range(a, b + 1))


def _closest_lines(text: str, old_text: str) -> str:
    first = next((ln.strip() for ln in old_text.splitlines() if ln.strip()), "")
    if not first:
        return ""
    lines = text.splitlines()
    stripped = [ln.strip() for ln in lines]
    close = difflib.get_close_matches(first, stripped, n=3, cutoff=0.55)
    if not close:
        return "\nNo similar lines found. Use read_file to see the current content."
    shown = []
    for c in close:
        idx = stripped.index(c)
        shown.append(f"  line {idx + 1}: {lines[idx].rstrip()}")
    return "\nClosest existing lines (check spacing/indentation):\n" + "\n".join(shown)


def _tree_lines(base: Path, max_depth: int, limit: int):
    lines = []

    def walk(d: Path, depth: int) -> bool:
        try:
            items = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return True
        for it in items:
            if it.name in IGNORED_DIRS:
                continue
            if len(lines) >= limit:
                lines.append("... (truncated)")
                return False
            indent = "  " * depth
            if it.is_dir():
                lines.append(f"{indent}{it.name}/")
                if depth + 1 < max_depth and walk(it, depth + 1) is False:
                    return False
            else:
                lines.append(f"{indent}{it.name}")
        return True

    walk(base, 0)
    return lines


# --------------------------------------------------------------------------
# Project detection
# --------------------------------------------------------------------------
def _pkg_json(d: Path) -> dict:
    return _read_json(d / "package.json", {}) if (d / "package.json").exists() else {}


def _text(d: Path, name: str) -> str:
    try:
        return (d / name).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def detect_kinds(d: Path):
    kinds = []
    pkg = _pkg_json(d)
    if (d / "package.json").exists():
        kinds.append("node")
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        for key, label in [("expo", "expo"), ("react-native", "react-native"), ("react", "react"),
                           ("next", "next.js"), ("express", "express"), ("vite", "vite"),
                           ("typescript", "typescript")]:
            if key in deps:
                kinds.append(label)
    py_text = (_text(d, "requirements.txt") + _text(d, "pyproject.toml")).lower()
    if py_text or (d / "setup.py").exists() or any(d.glob("*.py")):
        kinds.append("python")
        for key in ("flask", "fastapi", "django"):
            if key in py_text:
                kinds.append(key)
    if (d / "Cargo.toml").exists():
        kinds.append("rust")
    if (d / "go.mod").exists():
        kinds.append("go")
    if (d / "CMakeLists.txt").exists() or any(d.glob("**/*.cpp")):
        kinds.append("c++")
    if (d / "gradlew").exists():
        kinds.append("gradle/android")
    if list(d.glob("*.csproj")):
        kinds.append(".net")
    if (d / "index.html").exists():
        kinds.append("web")
    return kinds or ["unknown/empty"]


def _make_has_target(d: Path, target: str) -> bool:
    return bool(re.search(rf"^{re.escape(target)}\s*:", _text(d, "Makefile"), re.M))


def detect_test(d: Path):
    scripts = _pkg_json(d).get("scripts", {})
    if scripts.get("test") and "no test specified" not in scripts["test"]:
        return "npm test", "package.json test script"
    py_tests = (d / "tests").is_dir() and any((d / "tests").rglob("*.py"))
    if (any((d / n).exists() for n in ("pytest.ini", "tox.ini", "conftest.py")) or py_tests
            or "pytest" in _text(d, "pyproject.toml").lower()
            or any(d.glob("test_*.py")) or any(d.glob("*_test.py"))):
        return "python -m pytest -q", "pytest project"
    if (d / "Cargo.toml").exists():
        return "cargo test", "Rust project"
    if (d / "go.mod").exists():
        return "go test ./...", "Go module"
    if (d / "gradlew").exists():
        return "./gradlew test", "Gradle wrapper"
    if list(d.glob("*.csproj")):
        return "dotnet test", ".NET project"
    if _make_has_target(d, "test"):
        return "make test", "Makefile test target"
    return None, "no tests detected (create tests first)"


def detect_build(d: Path):
    pkg = _pkg_json(d)
    scripts = pkg.get("scripts", {})
    if scripts.get("build"):
        return "npm run build", "package.json build script"
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    if "expo" in deps:
        return "npx expo export", "Expo project (bundle check; use EAS for a real APK)"
    if (d / "tsconfig.json").exists():
        return "npx tsc", "TypeScript project"
    if (d / "Cargo.toml").exists():
        return "cargo build", "Rust project"
    if (d / "go.mod").exists():
        return "go build ./...", "Go module"
    if (d / "CMakeLists.txt").exists():
        return "cmake -S . -B build && cmake --build build", "CMake project"
    if (d / "Makefile").exists():
        return "make", "Makefile"
    if (d / "gradlew").exists():
        return "./gradlew build", "Gradle wrapper"
    if list(d.glob("*.csproj")):
        return "dotnet build", ".NET project"
    if any(d.glob("**/*.py")):
        return "python -m compileall -q -x '(\\.venv|node_modules)' .", "Python byte-compile check"
    if (d / "index.html").exists():
        return None, "static website - no build step needed"
    return None, "no build system detected"


def detect_run(d: Path):
    scripts = _pkg_json(d).get("scripts", {})
    if "expo" in {**_pkg_json(d).get("dependencies", {})}:
        return "npx expo start", "Expo dev server"
    if scripts.get("start"):
        return "npm start", "package.json start script"
    if scripts.get("dev"):
        return "npm run dev", "package.json dev script"
    for name in ("manage.py",):
        if (d / name).exists():
            return "python manage.py runserver", "Django"
    for name in ("app.py", "main.py", "server.py"):
        if (d / name).exists():
            return f"python {name}", f"Python entry {name}"
    if (d / "Cargo.toml").exists():
        return "cargo run", "Rust project"
    if (d / "go.mod").exists():
        return "go run .", "Go module"
    if _make_has_target(d, "run"):
        return "make run", "Makefile run target"
    if (d / "index.html").exists():
        return "python -m http.server 8000", "static site server"
    return None, "unknown"


# --------------------------------------------------------------------------
# Project templates
# --------------------------------------------------------------------------
GITIGNORE = ".venv/\n__pycache__/\n*.pyc\n.pytest_cache/\nnode_modules/\ndist/\nbuild/\n.env\n.expo/\n"

TEMPLATES = {
    "python": {
        "main.py": (
            "# __NAME__ - entry point\n\n\n"
            "def greet(name: str = \"world\") -> str:\n"
            "    return f\"Hello, {name}!\"\n\n\n"
            "def main() -> None:\n"
            "    print(greet())\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    main()\n"
        ),
        "tests/test_main.py": (
            "from main import greet\n\n\n"
            "def test_greet_default():\n"
            "    assert greet() == \"Hello, world!\"\n\n\n"
            "def test_greet_name():\n"
            "    assert greet(\"Ada\") == \"Hello, Ada!\"\n"
        ),
        "requirements.txt": "pytest\n",
        "README.md": "# __NAME__\n\n__DESC__\n\n## Run\n\n    python main.py\n\n## Test\n\n    python -m pytest -q\n",
    },
    "flask": {
        "app.py": (
            "from flask import Flask, jsonify\n\n"
            "app = Flask(__name__)\n\n\n"
            "@app.get(\"/\")\n"
            "def index():\n"
            "    return jsonify(message=\"Hello from __NAME__\")\n\n\n"
            "@app.get(\"/api/health\")\n"
            "def health():\n"
            "    return jsonify(status=\"ok\")\n\n\n"
            "if __name__ == \"__main__\":\n"
            "    app.run(host=\"127.0.0.1\", port=5000, debug=False)\n"
        ),
        "tests/test_app.py": (
            "from app import app\n\n\n"
            "def test_health():\n"
            "    client = app.test_client()\n"
            "    res = client.get(\"/api/health\")\n"
            "    assert res.status_code == 200\n"
            "    assert res.get_json() == {\"status\": \"ok\"}\n"
        ),
        "requirements.txt": "flask\npytest\n",
        "README.md": "# __NAME__\n\n__DESC__\n\n## Run\n\n    python app.py   # http://127.0.0.1:5000\n\n## Test\n\n    python -m pytest -q\n",
    },
    "node": {
        "package.json": (
            "{\n  \"name\": \"__SLUG__\",\n  \"version\": \"1.0.0\",\n  \"description\": \"__DESC__\",\n"
            "  \"main\": \"index.js\",\n  \"scripts\": {\n    \"start\": \"node index.js\",\n"
            "    \"test\": \"node --test\"\n  },\n  \"license\": \"MIT\"\n}\n"
        ),
        "index.js": (
            "function greet(name = 'world') {\n  return `Hello, ${name}!`;\n}\n\n"
            "module.exports = { greet };\n\n"
            "if (require.main === module) {\n  console.log(greet());\n}\n"
        ),
        "test/index.test.js": (
            "const test = require('node:test');\nconst assert = require('node:assert');\n"
            "const { greet } = require('../index');\n\n"
            "test('greet default', () => {\n  assert.strictEqual(greet(), 'Hello, world!');\n});\n"
        ),
        "README.md": "# __NAME__\n\n__DESC__\n\n## Run\n\n    npm start\n\n## Test\n\n    npm test\n",
    },
    "web": {
        "index.html": (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "  <title>__NAME__</title>\n  <link rel=\"stylesheet\" href=\"style.css\">\n</head>\n<body>\n"
            "  <main>\n    <h1>__NAME__</h1>\n    <button id=\"btn\">Clicked 0 times</button>\n  </main>\n"
            "  <script src=\"script.js\"></script>\n</body>\n</html>\n"
        ),
        "style.css": (
            "body { font-family: system-ui, sans-serif; margin: 0; display: grid; place-items: center; min-height: 100vh; }\n"
            "button { padding: 0.6rem 1.2rem; font-size: 1rem; cursor: pointer; }\n"
        ),
        "script.js": (
            "let count = 0;\nconst btn = document.getElementById('btn');\n"
            "btn.addEventListener('click', () => {\n  count += 1;\n  btn.textContent = `Clicked ${count} times`;\n});\n"
        ),
        "README.md": "# __NAME__\n\n__DESC__\n\n## Run\n\n    python -m http.server 8000\n",
    },
    "cpp": {
        "src/main.cpp": (
            "#include <iostream>\n#include <string>\n\n"
            "std::string greet(const std::string& name) { return \"Hello, \" + name + \"!\"; }\n\n"
            "int main(int argc, char** argv) {\n"
            "    if (argc > 1 && std::string(argv[1]) == \"--selftest\") {\n"
            "        return greet(\"world\") == \"Hello, world!\" ? 0 : 1;\n    }\n"
            "    std::cout << greet(\"world\") << std::endl;\n    return 0;\n}\n"
        ),
        "Makefile": (
            "CXX ?= g++\nCXXFLAGS ?= -std=c++17 -Wall -Wextra -O2\nTARGET := build/app\n\n"
            "all: $(TARGET)\n\n$(TARGET): src/main.cpp\n\t@mkdir -p build\n\t$(CXX) $(CXXFLAGS) -o $@ $<\n\n"
            "run: all\n\t./$(TARGET)\n\ntest: all\n\t./$(TARGET) --selftest\n\nclean:\n\trm -rf build\n"
        ),
        "README.md": "# __NAME__\n\n__DESC__\n\n    make        # build\n    make run\n    make test\n",
    },
    "empty": {"README.md": "# __NAME__\n\n__DESC__\n"},
}


# --------------------------------------------------------------------------
# Tools: projects
# --------------------------------------------------------------------------
@tool
def list_projects() -> str:
    """List all projects in the agent workspace and show which one is active."""
    active = project_name()
    rows = []
    for item in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        if not item.is_dir() or item.name.startswith(".") or item.name == "exports":
            continue
        mark = "*" if item.name == active else " "
        rows.append(f"{mark} {item.name}  [{', '.join(detect_kinds(item))}]")
    if not rows:
        return "No projects yet. Use create_project(name, template) to start one."
    return "Projects (* = active):\n" + "\n".join(rows)


@tool
def create_project(name: str, template: str = "python", description: str = "") -> str:
    """Create a NEW project folder from a template and make it the active project.
    Templates: python, flask, node, web, cpp, empty. Then edit the files to fit the user's goal."""
    name = name.strip()
    template = template.strip().lower()
    if not NAME_RE.fullmatch(name):
        return "Invalid project name. Use letters, digits, '.', '_' or '-' (max 64 chars, no spaces)."
    if template not in TEMPLATES:
        return f"Unknown template '{template}'. Choose one of: {', '.join(TEMPLATES)}."
    d = ROOT / name
    if d.exists() and any(d.iterdir()):
        return f"Project '{name}' already exists. Use switch_project('{name}') to work on it."
    d.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9._-]", "-", name.lower())
    files = dict(TEMPLATES[template])
    files[".gitignore"] = GITIGNORE
    for relpath, content in files.items():
        content = (content.replace("__NAME__", name).replace("__SLUG__", slug)
                   .replace("__DESC__", description.strip() or "TODO: describe this project."))
        target = d / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _write_json(STATE_FILE, {"project": name})
    if shutil.which("git"):
        _run_raw("git init -q", 30, cwd=d)
    listing = "\n".join(f"  {p}" for p in sorted(files))
    hints = {
        "python": "Next: install_dependencies, then run_tests.",
        "flask": "Next: install_dependencies, run_tests, then start_process('python app.py').",
        "node": "Next: run_tests (uses node --test, no install needed).",
        "web": "Next: start_process('python -m http.server 8000') and http_request to check it.",
        "cpp": "Next: run_build, then run_tests.",
        "empty": "Next: create the files you need.",
    }
    return (f"Created project '{name}' from template '{template}' and made it ACTIVE.\n"
            f"Files:\n{listing}\n{hints[template]}")


@tool
def switch_project(name: str) -> str:
    """Make an existing project the active one. All file tools then work inside it."""
    name = name.strip()
    if name in ("", ".", "/"):
        _write_json(STATE_FILE, {"project": ""})
        return "No active project (working at workspace root). Use list_projects."
    if not NAME_RE.fullmatch(name) or not (ROOT / name).is_dir():
        return f"Project '{name}' not found. Use list_projects to see what exists."
    _write_json(STATE_FILE, {"project": name})
    return f"Active project is now '{name}'."


@tool
def project_overview() -> str:
    """Understand the active project: type, test/build/run commands, files, git state, tasks.
    Call this first when starting work or after switching projects."""
    d = workdir()
    name = project_name()
    if not name:
        return ("NO ACTIVE PROJECT.\n" + list_projects()
                + "\nUse create_project to start a new one or switch_project to continue one.")
    out = [f"PROJECT: {name}", f"PATH: {d}", f"KIND: {', '.join(detect_kinds(d))}"]
    for label, fn in (("TEST", detect_test), ("BUILD", detect_build), ("RUN", detect_run)):
        cmd, why = fn(d)
        out.append(f"{label}: {cmd or '-'}  ({why})")
    if (d / ".git").exists():
        _, o, _, _, _ = _run_raw("git status --short --branch", 20)
        out.append("GIT: " + (o.strip().splitlines()[0] if o.strip() else "clean"))
    else:
        out.append("GIT: not initialised (git_init)")
    out.append("FILES:\n" + "\n".join("  " + ln for ln in _tree_lines(d, 3, 80)))
    readme = _text(d, "README.md").strip().splitlines()[:12]
    if readme:
        out.append("README (head):\n" + "\n".join("  " + ln for ln in readme))
    out.append("TASKS:\n" + manage_tasks("list"))
    out.append("RUNNING PROCESSES:\n" + list_processes())
    return "\n".join(out)


@tool
def workspace_info() -> str:
    """Show workspace location and installed tool versions (python, node, git, gcc, ...)."""
    rows = [f"WORKSPACE ROOT: {ROOT}", f"ACTIVE PROJECT: {project_name() or '(none)'}",
            f"SERVER PYTHON: {sys.executable}"]
    for label, cmd in [("python", "python --version"), ("pip", "python -m pip --version"),
                       ("node", "node --version"), ("npm", "npm --version"),
                       ("git", "git --version"), ("g++", "g++ --version"),
                       ("make", "make --version"), ("cmake", "cmake --version"),
                       ("cargo", "cargo --version"), ("go", "go version"), ("java", "java -version")]:
        rows.append(f"{label}: {_version(cmd)}")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Tools: files
# --------------------------------------------------------------------------
@tool
def list_files(path: str = ".") -> str:
    """List files and folders in one directory of the active project (not recursive)."""
    d = safe_path(path)
    if not d.exists():
        return f"Directory does not exist: {path}"
    if not d.is_dir():
        return f"Not a directory: {path}"
    rows = []
    for item in sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if item.name in IGNORED_DIRS:
            continue
        rows.append(f"[DIR]  {item.name}/" if item.is_dir()
                    else f"[FILE] {item.name} ({_human(item.stat().st_size)})")
    return "\n".join(rows) if rows else "Directory is empty."


@tool
def project_tree(path: str = ".", max_depth: int = 4) -> str:
    """Show the recursive folder structure (skips node_modules, .git, .venv, build output)."""
    base = safe_path(path)
    if not base.is_dir():
        return f"Not a directory: {path}"
    lines = _tree_lines(base, max(1, min(int(max_depth), 10)), MAX_TREE)
    return "\n".join(lines) if lines else "Empty."


@tool
def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Read a text file with line numbers. Always read before editing. end_line=0 means to the end."""
    p = safe_path(path)
    lines = read_text_file(p).splitlines()
    start = max(1, int(start_line))
    end = len(lines) if int(end_line) <= 0 else min(int(end_line), len(lines))
    if not lines:
        return f"FILE: {path} (empty)"
    if start > len(lines):
        return f"File has only {len(lines)} lines."
    body = "\n".join(f"{n:>6}: {lines[n - 1]}" for n in range(start, end + 1))
    return f"FILE: {path} ({len(lines)} lines, showing {start}-{end})\n" + truncate_output(body, MAX_READ_CHARS)


@tool
def write_file(path: str, content: str) -> str:
    """Create a file or completely replace it. Creates parent folders. Use replace_text for small edits."""
    p = safe_path(path)
    if p.is_dir():
        return f"'{path}' is a directory."
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content, encoding="utf-8")
    n = len(content.splitlines())
    return (f"{'Replaced' if existed else 'Created'} {path} ({n} lines, {len(content)} chars)"
            + _after_edit(p, content))


@tool
def append_file(path: str, content: str) -> str:
    """Append text to the end of a file (creates it if missing)."""
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
    return f"Appended {len(content)} chars to {path}" + _after_edit(p, p.read_text(encoding="utf-8"))


@tool
def replace_text(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
    """Replace exact text in a file (best for small precise edits). old_text must match exactly,
    including indentation. Use read_file first."""
    p = safe_path(path)
    text = read_text_file(p)
    if not old_text:
        return "old_text is empty. Use write_file, append_file or insert_lines to add content."
    count = text.count(old_text)
    if count == 0 and "\r\n" in text:  # file uses Windows line endings
        crlf_old, crlf_new = old_text.replace("\n", "\r\n"), new_text.replace("\n", "\r\n")
        if text.count(crlf_old):
            old_text, new_text, count = crlf_old, crlf_new, text.count(crlf_old)
    if count == 0:
        return "old_text not found. No changes made." + _closest_lines(text, old_text)
    if count > 1 and not replace_all:
        return (f"Found {count} matches. Add more surrounding lines to old_text "
                "or set replace_all=true. No changes made.")
    updated = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
    p.write_text(updated, encoding="utf-8")
    return f"Replaced {count if replace_all else 1} occurrence(s) in {path}" + _after_edit(p, updated)


@tool
def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines start_line..end_line (inclusive, 1-based) with new_content.
    Empty new_content deletes those lines."""
    p = safe_path(path)
    lines = read_text_file(p).splitlines()
    s, e = int(start_line), int(end_line)
    if s < 1 or e < s or e > len(lines):
        return f"Invalid range {s}-{e}. File has {len(lines)} lines."
    new_lines = new_content.splitlines()
    lines[s - 1:e] = new_lines
    updated = "\n".join(lines) + "\n"
    p.write_text(updated, encoding="utf-8")
    shown = _snippet(lines, s, max(s, s + len(new_lines) - 1)) if lines else ""
    return f"Replaced lines {s}-{e} in {path}.\n{shown}" + _after_edit(p, updated)


@tool
def insert_lines(path: str, after_line: int, content: str) -> str:
    """Insert content after the given line number (0 = at the top of the file)."""
    p = safe_path(path)
    lines = read_text_file(p).splitlines() if p.exists() else []
    n = int(after_line)
    if n < 0 or n > len(lines):
        return f"Invalid line {n}. File has {len(lines)} lines."
    new_lines = content.splitlines()
    lines[n:n] = new_lines
    updated = "\n".join(lines) + "\n"
    p.write_text(updated, encoding="utf-8")
    return (f"Inserted {len(new_lines)} line(s) into {path} after line {n}.\n"
            f"{_snippet(lines, n + 1, n + max(1, len(new_lines)))}" + _after_edit(p, updated))


@tool
def create_directory(path: str) -> str:
    """Create a directory (and missing parents)."""
    safe_path(path).mkdir(parents=True, exist_ok=True)
    return f"Created directory: {path}"


@tool
def delete_file(path: str) -> str:
    """Delete one file. Only when the user asked for it or it is clearly part of the task."""
    p = safe_path(path)
    if not p.exists():
        return f"File does not exist: {path}"
    if not p.is_file():
        return f"Not a file: {path} (use delete_directory)"
    p.unlink()
    return f"Deleted file: {path}"


@tool
def delete_directory(path: str, recursive: bool = False) -> str:
    """Delete a directory. recursive=true is required if it is not empty."""
    d = safe_path(path)
    if d == workdir():
        return "Refused: cannot delete the project root."
    if not d.exists():
        return f"Directory does not exist: {path}"
    if not d.is_dir():
        return f"Not a directory: {path}"
    try:
        shutil.rmtree(d) if recursive else d.rmdir()
    except OSError as e:
        return f"Could not delete: {e}. Use recursive=true for non-empty directories."
    return f"Deleted directory: {path}"


@tool
def copy_file(source: str, destination: str) -> str:
    """Copy a file or directory to another location inside the project."""
    src, dst = safe_path(source), safe_path(destination)
    if not src.exists():
        return f"Source does not exist: {source}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return f"Copied {source} -> {destination}"


@tool
def move_file(source: str, destination: str) -> str:
    """Move or rename a file or directory inside the project."""
    src, dst = safe_path(source), safe_path(destination)
    if not src.exists():
        return f"Source does not exist: {source}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"Moved {source} -> {destination}"


@tool
def file_info(path: str) -> str:
    """Get type, size, line count and modification time of a file or directory."""
    p = safe_path(path)
    if not p.exists():
        return f"Path does not exist: {path}"
    st = p.stat()
    info = [f"Path: {path}", f"Type: {'directory' if p.is_dir() else 'file'}",
            f"Size: {_human(st.st_size)}", f"Modified: {time.ctime(st.st_mtime)}"]
    if p.is_file() and p.suffix.lower() not in BINARY_SUFFIXES and st.st_size < MAX_FILE_BYTES:
        try:
            info.append(f"Lines: {len(p.read_text(encoding='utf-8').splitlines())}")
        except UnicodeDecodeError:
            pass
    return "\n".join(info)


@tool
def find_files(pattern: str = "*", path: str = ".") -> str:
    """Find files by name pattern, e.g. '*.py', 'test_*.js', 'src/*.ts'."""
    root = safe_path(path)
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [x for x in dirnames if x not in IGNORED_DIRS]
        for fn in filenames:
            fp = Path(dirpath) / fn
            r = str(fp.relative_to(workdir()))
            if fnmatch.fnmatch(fn, pattern) or fnmatch.fnmatch(r, pattern):
                hits.append(r)
                if len(hits) >= 200:
                    return "\n".join(sorted(hits)) + "\n... (limited to 200)"
    return "\n".join(sorted(hits)) if hits else "No files matched."


@tool
def search_files(query: str, path: str = ".", use_regex: bool = False,
                 max_results: int = 50, file_glob: str = "") -> str:
    """Search text inside project files (functions, imports, errors, TODOs).
    Optional file_glob such as '*.py' limits which files are searched."""
    root = safe_path(path)
    if not root.exists():
        return f"Path does not exist: {path}"
    limit = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    try:
        pattern = re.compile(query, re.IGNORECASE) if use_regex else None
    except re.error as e:
        return f"Invalid regex: {e}"
    needle = query.lower()
    results = []
    if root.is_file():
        walker = [(str(root.parent), [], [root.name])]
    else:
        walker = os.walk(root)
    for dirpath, dirnames, filenames in walker:
        dirnames[:] = [x for x in dirnames if x not in IGNORED_DIRS]
        for fn in sorted(filenames):
            fp = Path(dirpath) / fn
            if fp.suffix.lower() in BINARY_SUFFIXES or (file_glob and not fnmatch.fnmatch(fn, file_glob)):
                continue
            try:
                if fp.stat().st_size > 1_000_000:
                    continue
                text = fp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for no, line in enumerate(text.splitlines(), start=1):
                if (pattern.search(line) if pattern else needle in line.lower()):
                    results.append(f"{fp.relative_to(workdir())}:{no}: {line.strip()[:300]}")
                    if len(results) >= limit:
                        return "\n".join(results) + f"\n... (stopped at {limit} results)"
    return "\n".join(results) if results else "No matches found."


@tool
def check_syntax(path: str = ".") -> str:
    """Quick syntax check of .py, .json, .js and .sh files (one file or the whole project)."""
    root = safe_path(path)
    files = [root] if root.is_file() else []
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if x not in IGNORED_DIRS]
            files += [Path(dirpath) / f for f in filenames]
    problems, checked = [], 0
    has_node = bool(shutil.which("node"))
    for fp in files[:600]:
        suffix = fp.suffix.lower()
        if suffix in (".py", ".json"):
            try:
                text = fp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            checked += 1
            problem = syntax_problem(fp, text)
            if problem:
                problems.append(f"{rel(fp)}: {problem}")
        elif suffix in (".js", ".mjs", ".cjs") and has_node:
            checked += 1
            code, _, err, _, _ = _run_raw(f"node --check {shlex.quote(str(fp))}", 20)
            if code != 0:
                problems.append(f"{rel(fp)}: {err.strip().splitlines()[0] if err.strip() else 'syntax error'}")
        elif suffix == ".sh":
            checked += 1
            code, _, err, _, _ = _run_raw(f"bash -n {shlex.quote(str(fp))}", 20)
            if code != 0:
                problems.append(f"{rel(fp)}: {err.strip()}")
    if problems:
        return f"Checked {checked} file(s). PROBLEMS:\n" + "\n".join(problems)
    return f"Checked {checked} file(s). No syntax errors found."


# --------------------------------------------------------------------------
# Tools: run / test / build / lint / dependencies
# --------------------------------------------------------------------------
@tool
def run_command(command: str, timeout: int = 120) -> str:
    """Run a shell command in the active project (bash). For commands that finish.
    For servers or watchers use start_process. Uses the project's .venv automatically."""
    return run_process(command, timeout)


@tool
def run_python(code: str, timeout: int = 60) -> str:
    """Run a short Python snippet in the project for quick checks (uses project .venv if present)."""
    return run_process(f"python -c {shlex.quote(code)}", timeout)


@tool
def run_tests(command: str = "", timeout: int = 300) -> str:
    """Run the project's tests. Leave command empty to auto-detect (pytest, npm test, cargo, go, make...)."""
    d = workdir()
    chosen = command.strip()
    if chosen:
        header = f"TEST COMMAND: {chosen}\n\n"
    else:
        chosen, why = detect_test(d)
        if not chosen:
            return f"No tests found: {why}. Write tests first (for example tests/test_*.py)."
        header = f"AUTO-DETECTED: {chosen} ({why})\n\n"
    result = run_process(chosen, timeout)
    hint = ""
    if "No module named pytest" in result or "No module named 'pytest'" in result:
        hint = "\n\nHINT: pytest is missing. Call install_dependencies(packages='pytest')."
    elif "No module named" in result:
        hint = "\n\nHINT: a Python module is missing. Call install_dependencies."
    elif "Cannot find module" in result or "MODULE_NOT_FOUND" in result:
        hint = "\n\nHINT: Node dependencies are missing. Call install_dependencies."
    return header + result + hint


@tool
def run_build(command: str = "", timeout: int = 600) -> str:
    """Build the project. Leave command empty to auto-detect (npm build, cargo, cmake, make, tsc, go, ...)."""
    d = workdir()
    chosen = command.strip()
    if chosen:
        header = f"BUILD COMMAND: {chosen}\n\n"
    else:
        chosen, why = detect_build(d)
        if not chosen:
            return f"Nothing to build: {why}."
        header = f"AUTO-DETECTED: {chosen} ({why})\n\n"
    result = run_process(chosen, timeout)
    if "RESULT: FAILED" in result and ("Cannot find module" in result or "MODULE_NOT_FOUND" in result):
        result += "\n\nHINT: run install_dependencies first."
    return header + result


@tool
def run_linter(fix: bool = False) -> str:
    """Lint the project (ruff, npm run lint, tsc, cargo clippy, go vet). fix=true applies auto-fixes
    where supported. Falls back to a syntax check."""
    d = workdir()
    scripts = _pkg_json(d).get("scripts", {})
    cmd = None
    if scripts.get("lint"):
        cmd = "npm run lint" + (" -- --fix" if fix else "")
    elif (d / "Cargo.toml").exists():
        cmd = "cargo clippy"
    elif (d / "go.mod").exists():
        cmd = "go vet ./..."
    elif (d / "tsconfig.json").exists():
        cmd = "npx tsc --noEmit"
    elif _which("ruff") and any(d.glob("**/*.py")):
        cmd = "ruff check ." + (" --fix" if fix else "")
    if cmd:
        return run_process(cmd, 300)
    return "No linter configured, falling back to syntax check.\n" + check_syntax(".")


@tool
def format_code() -> str:
    """Auto-format code if a formatter is available (npm run format, ruff format, cargo fmt, gofmt)."""
    d = workdir()
    scripts = _pkg_json(d).get("scripts", {})
    if scripts.get("format"):
        return run_process("npm run format", 300)
    if (d / "Cargo.toml").exists():
        return run_process("cargo fmt", 120)
    if (d / "go.mod").exists():
        return run_process("gofmt -w .", 120)
    if _which("ruff") and any(d.glob("**/*.py")):
        return run_process("ruff format .", 120)
    if _which("black") and any(d.glob("**/*.py")):
        return run_process("black .", 120)
    return "No formatter available. Install one with install_dependencies(packages='ruff')."


PKG_TOKEN = re.compile(r"^[A-Za-z0-9@][A-Za-z0-9@/._=<>~!,\[\]-]*$")


@tool
def install_dependencies(packages: str = "", manager: str = "auto", timeout: int = 600) -> str:
    """Install dependencies. No packages: install what the project declares (requirements.txt,
    pyproject.toml, package.json, Cargo, go.mod). With packages (space separated): install them.
    manager: auto | pip | npm. Python packages go into the project's own .venv (created if missing)
    and are added to requirements.txt."""
    d = workdir()
    tokens = shlex.split(packages) if packages.strip() else []
    for t in tokens:
        if not PKG_TOKEN.match(t):
            return f"Invalid package name: {t}"
    manager = manager.strip().lower()
    if manager == "auto":
        if (d / "package.json").exists():
            manager = "npm"
        elif (d / "Cargo.toml").exists():
            manager = "cargo"
        elif (d / "go.mod").exists():
            manager = "go"
        else:
            manager = "pip"
    quoted = " ".join(shlex.quote(t) for t in tokens)
    if manager == "npm":
        return run_process(f"npm install {quoted}".strip(), timeout)
    if manager == "cargo":
        return run_process(f"cargo add {quoted}" if tokens else "cargo fetch", timeout)
    if manager == "go":
        return run_process(f"go get {quoted}" if tokens else "go mod download", timeout)
    if manager != "pip":
        return "manager must be auto, pip or npm."

    venv_existed = (d / ".venv" / "bin" / "python").exists()
    cmds = ["[ -x .venv/bin/python ] || python -m venv .venv"]
    pip = ".venv/bin/python -m pip install -q"
    if (not venv_existed or not tokens) and (d / "requirements.txt").exists():
        cmds.append(f"{pip} -r requirements.txt")
    elif not tokens and (d / "pyproject.toml").exists():
        cmds.append(f"{pip} -e .")
    elif not tokens and not venv_existed:
        return "No requirements.txt or pyproject.toml found, and no packages given. Nothing to install."
    if tokens:
        cmds.append(f"{pip} {quoted}")
    result = run_process(" && ".join(cmds), timeout)
    if tokens and "RESULT: SUCCESS" in result:
        req = d / "requirements.txt"
        existing = req.read_text(encoding="utf-8") if req.exists() else ""
        known = {re.split(r"[<>=!~\[ ]", ln.strip())[0].lower() for ln in existing.splitlines() if ln.strip()}
        add = [t for t in tokens if re.split(r"[<>=!~\[]", t)[0].lower() not in known]
        if add:
            req.write_text(existing.rstrip("\n") + ("\n" if existing else "") + "\n".join(add) + "\n",
                           encoding="utf-8")
            result += f"\n\nAdded to requirements.txt: {', '.join(add)}"
    return result


# --------------------------------------------------------------------------
# Tools: background processes + HTTP
# --------------------------------------------------------------------------
PROCS = {}


def _alive(entry) -> bool:
    return entry["proc"].poll() is None


def _tail(path: Path, lines: int) -> str:
    try:
        data = path.read_bytes()[-200_000:].decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-max(1, lines):])


@tool
def start_process(command: str, name: str = "server") -> str:
    """Start a long-running command in the background (dev server, watcher, API). Returns after 3 seconds
    with the first output. Then use process_output, http_request and stop_process."""
    reason = command_is_blocked(command)
    if reason:
        return f"BLOCKED: {reason}."
    if not NAME_RE.fullmatch(name):
        return "Invalid process name (letters, digits, '.', '_', '-')."
    if name in PROCS and _alive(PROCS[name]):
        return f"A process named '{name}' is already running. stop_process('{name}') first or use another name."
    log = Path(tempfile.gettempdir()) / f"agent_{os.getpid()}_{name}.log"
    handle = open(log, "wb")
    proc = subprocess.Popen(
        ["/bin/bash", "-c", command], cwd=str(workdir()), env=build_env(),
        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
    )
    PROCS[name] = {"proc": proc, "cmd": command, "log": log, "started": time.time(), "handle": handle}
    time.sleep(3)
    head = _tail(log, 30)
    if proc.poll() is not None:
        return (f"Process '{name}' EXITED immediately (code {proc.returncode}).\n"
                f"OUTPUT:\n{head or '(none)'}")
    return f"Started '{name}' (pid {proc.pid}): {command}\nFIRST OUTPUT:\n{head or '(none yet)'}"


@tool
def process_output(name: str = "server", lines: int = 60) -> str:
    """Show the latest output of a background process and whether it is still running."""
    entry = PROCS.get(name)
    if not entry:
        return f"No process named '{name}'. Use list_processes."
    state = "RUNNING" if _alive(entry) else f"EXITED (code {entry['proc'].returncode})"
    return f"{name}: {state}\n{truncate_output(_tail(entry['log'], int(lines)), 12_000) or '(no output)'}"


@tool
def stop_process(name: str = "server") -> str:
    """Stop a background process started with start_process."""
    entry = PROCS.get(name)
    if not entry:
        return f"No process named '{name}'."
    if _alive(entry):
        _kill_group(entry["proc"], signal.SIGTERM)
        try:
            entry["proc"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_group(entry["proc"], signal.SIGKILL)
    final = _tail(entry["log"], 15)
    entry["handle"].close()
    del PROCS[name]
    return f"Stopped '{name}'.\nLast output:\n{final or '(none)'}"


@tool
def list_processes() -> str:
    """List background processes started with start_process."""
    if not PROCS:
        return "No background processes."
    rows = []
    for name, e in PROCS.items():
        state = "running" if _alive(e) else f"exited({e['proc'].returncode})"
        rows.append(f"{name}: {state}, {int(time.time() - e['started'])}s, cmd: {e['cmd']}")
    return "\n".join(rows)


def _cleanup() -> None:
    for e in list(PROCS.values()):
        if _alive(e):
            _kill_group(e["proc"], signal.SIGTERM)


atexit.register(_cleanup)


@tool
def http_request(url: str, method: str = "GET", body: str = "", headers: str = "",
                 wait_seconds: int = 0) -> str:
    """Call a LOCAL http server (localhost / 127.0.0.1 only) to test an API or website.
    body: text or JSON string. headers: JSON object string. wait_seconds: retry while the server starts."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return "Only http://localhost or http://127.0.0.1 URLs are allowed."
    hdrs = {}
    if headers.strip():
        try:
            hdrs = json.loads(headers)
        except json.JSONDecodeError:
            return "headers must be a JSON object string, e.g. {\"Authorization\": \"Bearer x\"}"
    data = body.encode("utf-8") if body else None
    if data and "Content-Type" not in hdrs:
        hdrs["Content-Type"] = "application/json" if body.lstrip()[:1] in "{[" else "text/plain"
    deadline = time.time() + max(0, min(int(wait_seconds), 60))
    while True:
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read(300_000).decode("utf-8", errors="replace")
                return (f"STATUS: {resp.status}\nCONTENT-TYPE: {resp.headers.get('Content-Type', '?')}\n"
                        f"BODY:\n{truncate_output(text, 8000)}")
        except urllib.error.HTTPError as e:
            text = e.read(100_000).decode("utf-8", errors="replace")
            return f"STATUS: {e.code} {e.reason}\nBODY:\n{truncate_output(text, 8000)}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if time.time() < deadline:
                time.sleep(1)
                continue
            return (f"CONNECTION FAILED: {getattr(e, 'reason', e)}\n"
                    "Is the server running? Check process_output. You can pass wait_seconds=10.")


# --------------------------------------------------------------------------
# Tools: git
# --------------------------------------------------------------------------
def _need_git():
    if not (workdir() / ".git").exists():
        raise ToolError("This project is not a git repository. Call git_init first.")


@tool
def git_init() -> str:
    """Initialise a git repository in the active project."""
    if (workdir() / ".git").exists():
        return "Already a git repository."
    if not (workdir() / ".gitignore").exists():
        (workdir() / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    return run_process("git init", 30)


@tool
def git_status() -> str:
    """Show git status (changed and untracked files, current branch)."""
    _need_git()
    return run_process("git status --short --branch", 60)


@tool
def git_diff(path: str = "", staged: bool = False) -> str:
    """Show uncommitted changes. Optionally limit to a path. staged=true shows staged changes."""
    _need_git()
    cmd = "git --no-pager diff" + (" --staged" if staged else "")
    if path:
        cmd += f" -- {shlex.quote(rel(safe_path(path)))}"
    return run_process(cmd, 60)


@tool
def git_log(count: int = 10) -> str:
    """Show recent commits."""
    _need_git()
    return run_process(f"git --no-pager log --oneline -n {max(1, min(int(count), 50))}", 60)


@tool
def git_create_branch(name: str) -> str:
    """Create and switch to a new git branch."""
    _need_git()
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", name):
        return "Invalid branch name."
    return run_process(f"git switch -c {shlex.quote(name)}", 60)


@tool
def git_switch(name: str) -> str:
    """Switch to an existing git branch."""
    _need_git()
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", name):
        return "Invalid branch name."
    return run_process(f"git switch {shlex.quote(name)}", 60)


@tool
def git_commit(message: str) -> str:
    """Stage everything and commit. ONLY when the user asked for a commit."""
    _need_git()
    if not message.strip():
        return "Commit message cannot be empty."
    _, email, _, _, _ = _run_raw("git config user.email", 10)
    if not email.strip():
        _run_raw("git config user.email 'agent@localhost' && git config user.name 'Local Coding Agent'", 10)
    return run_process(f"git add -A && git commit -m {shlex.quote(message)}", 120)


# --------------------------------------------------------------------------
# Tools: tasks, checkpoints, packaging
# --------------------------------------------------------------------------
def _tasks_file() -> Path:
    return META / "tasks" / f"{project_name() or '_root'}.json"


@tool
def manage_tasks(action: str = "list", text: str = "") -> str:
    """Keep a to-do list for the active project. action: add (text, one task per line),
    done (text = task number), list, clear. Use it to plan big requests and track progress."""
    f = _tasks_file()
    tasks = _read_json(f, [])
    action = action.strip().lower()
    if action == "add":
        new = [ln.strip(" -*") for ln in text.splitlines() if ln.strip(" -*")]
        if not new:
            return "Give the task text to add."
        start = max([t["id"] for t in tasks], default=0)
        for i, t in enumerate(new, start=1):
            tasks.append({"id": start + i, "text": t, "done": False})
        _write_json(f, tasks)
    elif action == "done":
        key = text.strip().lstrip("#")
        hit = next((t for t in tasks if str(t["id"]) == key), None) or \
            next((t for t in tasks if key and key.lower() in t["text"].lower()), None)
        if not hit:
            return f"No task matches '{text}'."
        hit["done"] = True
        _write_json(f, tasks)
    elif action == "clear":
        _write_json(f, [])
        return "Task list cleared."
    elif action != "list":
        return "action must be add, done, list or clear."
    if not tasks:
        return "(no tasks)"
    left = sum(1 for t in tasks if not t["done"])
    body = "\n".join(f"[{'x' if t['done'] else ' '}] {t['id']}. {t['text']}" for t in tasks)
    return f"{body}\n({left} remaining)"


def _checkpoint_dir() -> Path:
    return META / "checkpoints" / (project_name() or "_root")


def _zip_dir(base: Path, dest: Path, exclude: set) -> int:
    count = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in exclude]
            for fn in filenames:
                fp = Path(dirpath) / fn
                if fp.is_symlink():
                    continue
                z.write(fp, fp.relative_to(base))
                count += 1
    return count


def _safe_label(label: str, default: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]", "-", label.strip())[:60]
    return label or default


@tool
def create_checkpoint(name: str = "") -> str:
    """Save a snapshot of the project files (before risky changes). Restore later with restore_checkpoint."""
    label = _safe_label(name, time.strftime("cp-%Y%m%d-%H%M%S"))
    dest = _checkpoint_dir() / f"{label}.zip"
    n = _zip_dir(workdir(), dest, IGNORED_DIRS | {"exports"})
    return f"Checkpoint '{label}' saved ({n} files)."


@tool
def list_checkpoints() -> str:
    """List saved checkpoints of the active project."""
    d = _checkpoint_dir()
    items = sorted(d.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True) if d.exists() else []
    if not items:
        return "No checkpoints."
    return "\n".join(f"{p.stem}  ({time.ctime(p.stat().st_mtime)}, {_human(p.stat().st_size)})" for p in items)


@tool
def restore_checkpoint(name: str) -> str:
    """Restore project files from a checkpoint (replaces current files; a backup is made first)."""
    zp = _checkpoint_dir() / f"{_safe_label(name, '')}.zip"
    if not zp.exists():
        return f"Checkpoint '{name}' not found. Use list_checkpoints."
    base = workdir()
    backup = time.strftime("before-restore-%Y%m%d-%H%M%S")
    _zip_dir(base, _checkpoint_dir() / f"{backup}.zip", IGNORED_DIRS | {"exports"})
    for item in list(base.iterdir()):
        if item.name in IGNORED_DIRS or item.name == "exports":
            continue
        shutil.rmtree(item) if item.is_dir() and not item.is_symlink() else item.unlink()
    with zipfile.ZipFile(zp) as z:
        for member in z.namelist():
            if member.startswith("/") or ".." in Path(member).parts:
                raise ToolError("Unsafe path inside checkpoint archive.")
        z.extractall(base)
    return f"Restored checkpoint '{name}'. Previous state saved as '{backup}'."


@tool
def package_project(name: str = "", include_build_output: bool = True) -> str:
    """Zip the project (source plus dist/build output) into the workspace 'exports' folder so the
    user can take it. Excludes node_modules, .venv and .git."""
    if not project_name():
        return "No active project. Use switch_project first."
    exclude = set(IGNORED_DIRS) | {"exports"}
    if include_build_output:
        exclude -= {"dist", "build"}
    label = _safe_label(name, f"{project_name()}-{time.strftime('%Y%m%d-%H%M%S')}")
    dest = EXPORTS / f"{label}.zip"
    n = _zip_dir(workdir(), dest, exclude)
    return f"Packaged {n} files -> {dest} ({_human(dest.stat().st_size)})"


if __name__ == "__main__":
    print(f"[local-coding-agent] workspace: {ROOT}", file=sys.stderr)
    mcp.run()
