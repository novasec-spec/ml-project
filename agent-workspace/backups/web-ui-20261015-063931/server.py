"""Workspace-scoped MCP tools for the remote Ollama coding agent."""
import difflib, os, shlex, subprocess, sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Workspace Coding Agent")
WORKSPACE = Path(os.environ.get("CODING_WORKSPACE", str(Path.home()/"HopeLinkCancer-AI"))).expanduser().resolve()
MAX_FILE_BYTES = 500_000
IGNORED = {".git", ".venv", "venv", "node_modules", "__pycache__", ".next", "dist", "build", ".expo", "coverage"}
ALLOWED = {"python", "python3", "cd", "pytest", "npm", "npx", "node", "git"}

def safe_path(p="."):
    raw=Path(p).expanduser(); target=(raw if raw.is_absolute() else WORKSPACE/raw).resolve()
    if target != WORKSPACE and WORKSPACE not in target.parents: raise ValueError("Blocked path outside workspace")
    return target

def require_file(p):
    if not p.is_file(): raise FileNotFoundError(str(p))
    if p.stat().st_size > MAX_FILE_BYTES: raise ValueError(f"File exceeds {MAX_FILE_BYTES} bytes")

def diff(path,a,b): return "".join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile="a/"+path,tofile="b/"+path)) or "(No differences.)"

@mcp.tool()
def workspace_info() -> str:
    """Show configured workspace and existence status."""
    return f"Workspace: {WORKSPACE}\nExists: {WORKSPACE.exists()}\nPython: {sys.executable}"

@mcp.tool()
def list_files(path: str=".", max_entries: int=250) -> str:
    """List files, excluding dependency and build directories."""
    root=safe_path(path)
    if not root.is_dir(): raise NotADirectoryError(str(root))
    out=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(x in IGNORED for x in p.parts):
            out.append(str(p.relative_to(WORKSPACE)))
            if len(out)>=max(1,min(int(max_entries),500)): break
    return "\n".join(out) or "(No files found.)"

@mcp.tool()
def read_file(path: str, start_line: int=1, max_lines: int=500) -> str:
    """Read a UTF-8 file with bounded line ranges."""
    p=safe_path(path); require_file(p)
    try: lines=p.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as e: raise ValueError("File is not UTF-8 text") from e
    start=max(1,int(start_line)); count=max(1,min(int(max_lines),1000))
    out="\n".join(f"{i+start}: {line}" for i,line in enumerate(lines[start-1:start-1+count]))
    return out[:30000] or "(No lines in requested range.)"

@mcp.tool()
def search_code(query: str, path: str=".", max_results: int=100) -> str:
    """Case-insensitive literal search across workspace text files."""
    if not query: raise ValueError("query must not be empty")
    root=safe_path(path)
    if not root.exists(): raise FileNotFoundError(path)
    out=[]; candidates=[root] if root.is_file() else root.rglob("*")
    for p in candidates:
        if not p.is_file() or any(x in IGNORED for x in p.parts): continue
        try:
            if p.stat().st_size>MAX_FILE_BYTES: continue
            lines=p.read_text(encoding="utf-8").splitlines()
        except (OSError,UnicodeDecodeError): continue
        for n,line in enumerate(lines,1):
            if query.casefold() in line.casefold():
                out.append(f"{p.relative_to(WORKSPACE)}:{n}: {line[:240]}")
                if len(out)>=max(1,min(int(max_results),300)): return "\n".join(out)
    return "\n".join(out) or "(No matches.)"

@mcp.tool()
def propose_edit(path: str, old_text: str, new_text: str) -> str:
    """Preview a unique exact replacement without modifying the file."""
    p=safe_path(path); require_file(p)
    if not old_text: raise ValueError("old_text must not be empty")
    old=p.read_text(encoding="utf-8"); n=old.count(old_text)
    if n!=1: raise ValueError(f"Expected old_text exactly once; found {n}")
    return diff(path,old,old.replace(old_text,new_text,1))

@mcp.tool()
def apply_edit(path: str, old_text: str, new_text: str, confirm: bool=False) -> str:
    """Apply a unique exact replacement only after client approval."""
    if confirm is not True: raise PermissionError("confirm=true required")
    p=safe_path(path); require_file(p); old=p.read_text(encoding="utf-8")
    if not old_text or old.count(old_text)!=1: raise ValueError("old_text must occur exactly once")
    new=old.replace(old_text,new_text,1); tmp=p.with_name(p.name+".agent-tmp"); tmp.write_text(new,encoding="utf-8"); tmp.replace(p)
    return f"Updated {path}\n"+diff(path,old,new)

@mcp.tool()
def write_file(path: str, content: str, overwrite: bool=False, confirm: bool=False) -> str:
    """Create a file; overwriting requires explicit approval and overwrite=true."""
    if confirm is not True: raise PermissionError("confirm=true required")
    if len(content.encode())>MAX_FILE_BYTES: raise ValueError("Content too large")
    p=safe_path(path)
    if p.exists() and not overwrite: raise FileExistsError(f"{path} exists; overwrite=true required")
    if p.exists() and p.is_dir(): raise IsADirectoryError(path)
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+".agent-tmp"); tmp.write_text(content,encoding="utf-8"); tmp.replace(p)
    return f"{'Updated' if overwrite else 'Created'} {path} ({len(content.encode())} bytes)"

@mcp.tool()
def create_directory(path: str, confirm: bool=False) -> str:
    """Create a workspace directory after client approval."""
    if confirm is not True: raise PermissionError("confirm=true required")
    p=safe_path(path); p.mkdir(parents=True,exist_ok=True); return f"Directory ready: {p.relative_to(WORKSPACE)}"

@mcp.tool()
def run_command(command: str, timeout_seconds: int=60, confirm: bool=False) -> str:
    """Run one allow-listed command; no shell pipelines; approval required."""
    if confirm is not True: raise PermissionError("confirm=true required")
    args=shlex.split(command)
    if not args: raise ValueError("Empty command")
    exe=Path(args[0]).name
    if exe not in ALLOWED: raise PermissionError(f"Executable {exe!r} blocked; allowed: {', '.join(sorted(ALLOWED))}")
    if exe=="git" and len(args)>1 and args[1] in {"push","reset","clean","checkout","restore","commit","rm"}: raise PermissionError("Git operation blocked for safety")
    result=subprocess.run(args,cwd=WORKSPACE,capture_output=True,text=True,timeout=max(1,min(int(timeout_seconds),120)),check=False)
    output=(result.stdout+"\n"+result.stderr).strip()
    return f"Exit code: {result.returncode}\n{output[-18000:]}"

@mcp.tool()
def run_python_tests(path: str=".", timeout_seconds: int=90, confirm: bool=False) -> str:
    """Run pytest on a workspace path after client approval."""
    if confirm is not True: raise PermissionError("confirm=true required")
    p=safe_path(path)
    if not p.exists(): raise FileNotFoundError(path)
    result=subprocess.run([sys.executable,"-m","pytest",str(p)],cwd=WORKSPACE,capture_output=True,text=True,timeout=max(1,min(int(timeout_seconds),120)),check=False)
    output=(result.stdout+"\n"+result.stderr).strip()
    return f"Exit code: {result.returncode}\n{output[-18000:]}"

if __name__=="__main__":
    # Keep stdout clean: MCP stdio uses stdout for JSON-RPC only.
    print(f"[coding-agent] workspace={WORKSPACE}",file=sys.stderr,flush=True)
    mcp.run()
