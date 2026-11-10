from pathlib import Path
from mcp.server import MCPServer

mcp = MCPServer("Local AI Tools")

# --------------------------------------------------
# SECURITY
# --------------------------------------------------

BASE_DIR = (Path.home() / "ml-project" / "mcp-agent").resolve()


def safe_path(filename: str) -> Path:
    """Return a safe path inside BASE_DIR."""

    path = (BASE_DIR / filename).resolve()

    if path != BASE_DIR and BASE_DIR not in path.parents:
        raise ValueError(
            "Access outside the MCP project folder is not allowed."
        )

    return path


# --------------------------------------------------
# LIST FILES
# --------------------------------------------------

@mcp.tool()
def list_files() -> str:
    """List files and folders inside the MCP project."""

    items = []

    for path in BASE_DIR.iterdir():

        if path.is_dir():
            items.append(f"[DIR]  {path.name}")

        else:
            items.append(f"[FILE] {path.name}")

    if not items:
        return "The project folder is empty."

    return "\n".join(sorted(items))


# --------------------------------------------------
# READ FILE
# --------------------------------------------------

@mcp.tool()
def read_file(filename: str) -> str:
    """Read the complete contents of a text file."""

    try:
        path = safe_path(filename)

        if not path.exists():
            return f"Error: '{filename}' does not exist."

        if not path.is_file():
            return f"Error: '{filename}' is not a file."

        return path.read_text()

    except Exception as e:
        return f"Error reading file: {e}"


# --------------------------------------------------
# WRITE FILE
# --------------------------------------------------

@mcp.tool()
def write_file(filename: str, content: str) -> str:
    """Create or completely replace a text file."""

    try:
        path = safe_path(filename)

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content)

        return f"Successfully wrote {len(content)} characters to '{filename}'."

    except Exception as e:
        return f"Error writing file: {e}"


# --------------------------------------------------
# APPEND FILE
# --------------------------------------------------

@mcp.tool()
def append_file(filename: str, content: str) -> str:
    """Add text to the end of an existing or new text file."""

    try:
        path = safe_path(filename)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a") as file:
            file.write(content)

        return f"Successfully appended text to '{filename}'."

    except Exception as e:
        return f"Error appending to file: {e}"


# --------------------------------------------------
# CREATE FOLDER
# --------------------------------------------------

@mcp.tool()
def create_folder(folder_name: str) -> str:
    """Create a folder inside the MCP project."""

    try:
        path = safe_path(folder_name)

        path.mkdir(parents=True, exist_ok=True)

        return f"Successfully created folder '{folder_name}'."

    except Exception as e:
        return f"Error creating folder: {e}"


# --------------------------------------------------
# FILE INFORMATION
# --------------------------------------------------

@mcp.tool()
def file_info(filename: str) -> str:
    """Show basic information about a file."""

    try:
        path = safe_path(filename)

        if not path.exists():
            return f"Error: '{filename}' does not exist."

        if path.is_dir():
            return f"'{filename}' is a directory."

        size = path.stat().st_size

        return (
            f"File: {filename}\n"
            f"Size: {size} bytes\n"
            f"Type: {path.suffix or 'no extension'}"
        )

    except Exception as e:
        return f"Error getting file information: {e}"


# --------------------------------------------------
# START MCP SERVER
# --------------------------------------------------

if __name__ == "__main__":
    mcp.run()
