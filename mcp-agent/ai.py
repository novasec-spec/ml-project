import asyncio
from ollama import chat
from mcp import Client, StdioServerParameters

MODEL = "qwen3:0.6b"


async def main():

    server = StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    print("Connecting to MCP server...")

    async with Client(server) as client:

        # Discover MCP tools
        tools_result = await client.list_tools()

        print("\nMCP connected!")
        print("Available tools:")

        for tool in tools_result.tools:
            print(f"  • {tool.name}")

        # Convert MCP tools into Ollama tools
        ollama_tools = []

        for tool in tools_result.tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            })

        messages = []

        print("\n================================")
        print(" Local MCP AI")
        print(" Model:", MODEL)
        print("================================")
        print("Type 'exit' to quit.")

        while True:

            user = input("\nYou: ")

            if user.lower() == "exit":
                break

            messages.append({
                "role": "user",
                "content": user,
            })

            # Ask Ollama
            response = chat(
                model=MODEL,
                messages=messages,
                tools=ollama_tools,
            )

            # Did the AI request an MCP tool?
            if response.message.tool_calls:

                messages.append(response.message)

                for tool_call in response.message.tool_calls:

                    tool_name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    print(
                        f"\n[MCP → {tool_name}({arguments})]"
                    )

                    # Execute MCP tool
                    result = await client.call_tool(
                        tool_name,
                        arguments,
                    )

                    # Extract returned text
                    tool_output = ""

                    for content in result.content:

                        if hasattr(content, "text"):
                            tool_output += content.text

                    print(
                        f"[MCP ← {tool_output}]"
                    )

                    messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": tool_output,
                    })

                # Let Ollama interpret the tool result
                final_response = chat(
                    model=MODEL,
                    messages=messages,
                )

                messages.append(final_response.message)

                print(
                    "\nAI:",
                    final_response.message.content,
                )

            else:

                messages.append(response.message)

                print(
                    "\nAI:",
                    response.message.content,
                )


if __name__ == "__main__":
    asyncio.run(main())
