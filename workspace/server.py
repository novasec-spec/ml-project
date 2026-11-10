from mcp.server.fastmcp import FastMCP
import pathlib, datetime

mcp = FastMCP("local-tools")

@mcp.tool()
def now() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().isoformat()

@mcp.tool()
def read_note(name: str) -> str:
    """Read a text note from ~/notes by filename."""
    p = pathlib.Path.home() / "notes" / name
    return p.read_text() if p.exists() else "Note not found."

if __name__ == "__main__":
    mcp.run()  # stdio transport
