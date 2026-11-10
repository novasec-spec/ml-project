from mcp.server import MCPServer

mcp = MCPServer("Local AI Tools")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """Greet a person."""
    return f"Hello {name}! Your local AI has an MCP tool."


if __name__ == "__main__":
    mcp.run()
