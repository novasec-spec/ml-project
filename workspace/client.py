import asyncio, ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "qwen2.5:0.5b"

async def main():
    params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            ollama_tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            } for t in tools]

            messages = []
            while True:
                messages.append({"role": "user", "content": input("> ")})
                while True:
                    msg = ollama.chat(model=MODEL, messages=messages,
                                      tools=ollama_tools).message
                    messages.append(msg)
                    if not msg.tool_calls:
                        print(msg.content)
                        break
                    for call in msg.tool_calls:
                        res = await session.call_tool(
                            call.function.name, call.function.arguments)
                        text = "\n".join(c.text for c in res.content
                                         if c.type == "text")
                        messages.append({"role": "tool",
                                         "tool_name": call.function.name,
                                         "content": text})

asyncio.run(main())
