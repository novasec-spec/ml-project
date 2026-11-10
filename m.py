from pathlib import Path
from mcp.server import MCPServer

mcp = MCPServer("Local AI Tools")

BASE_DIR = Path.home() / "1st"


@mcp.tool()
def read_file(filename: str) -> str:
    """Read a text file from the local AI project directory."""
    file_path = (BASE_DIR / filename).resolve()

    # Security: don't allow access outside BASE_DIR
    if BASE_DIR not in file_path.parents and file_path != BASE_DIR:
        return "Error: access outside the project directory is not allowed."

    if not file_path.exists():
        return f"Error: file '{filename}' does not exist."

    if not file_path.is_file():
        return f"Error: '{filename}' is not a file."

    try:
        return file_path.read_text()
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def list_files() -> str:
    """List files in the local AI project directory."""
    files = []

    for path in BASE_DIR.iterdir():
        if path.is_file():
            files.append(path.name)

    if not files:
        return "No files found."

    return "\n".join(sorted(files))


if __name__ == "__main__":
    mcp.run()
