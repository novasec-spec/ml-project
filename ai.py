import asyncio
import subprocess

from ollama import chat
from mcp import client, StdioServerParameters


def get_models():
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    lines = result.stdout.strip().splitlines()
    if len(lines) <= 1:
        return []

    models = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])

    return models


def choose_model():
    models = get_models()

    if not models:
        print("No Ollama models found.")
        return None

    print("\n================================")
    print("       SELECT AI MODEL")
    print("================================")

    for index, model in enumerate(models, start=1):
        print(f"{index}. {model}")

    print("================================")

    while True:
        choice = input("\nChoose model number: ").strip()

        try:
            number = int(choice)
            if 1 <= number <= len(models):
                selected = models[number - 1]
                print(f"\nSelected model: {selected}")
                return selected
        except ValueError:
            pass

        print("Invalid choice. Enter one of the numbers above.")


SYSTEM_PROMPT = """
You are an autonomous local software engineering agent.

You have MCP tools for inspecting, editing, executing, testing, and
version-controlling a project inside the provided workspace.

CORE RULES:
1. Understand the user's goal before acting.
2. Inspect the project before making changes.
3. Read relevant files before editing them.
4. Search the project when you need to locate code or symbols.
5. Prefer small precise edits with replace_text when possible.
6. Create folders/files when needed.
7. Use run_command for builds, scripts, package commands, linters,
   formatters, diagnostics, and other development commands.
8. Use run_tests after meaningful code changes.
9. If a command or test fails, read the error, find the cause,
   fix the code, and run the check again.
10. Repeat the fix-and-test cycle until the task works or a real
    blocker remains.
11. Never claim success without tool output proving it.
12. Never invent file contents, command output, or test results.
13. Keep project work inside the provided workspace.
14. Do not delete files/directories unless the user clearly
    requested it or it is necessary and clearly authorized.
15. Do not create Git commits unless the user asks for a commit.
16. When asked to build something, actually modify/create the
    project rather than only explaining what the user should do.
17. Before finishing, verify the result and summarize exactly
    what changed and what checks passed or failed.

TOOL SELECTION:
- project_tree/list_files = understand structure
- read_file = inspect code
- search_files = find code/symbols/text
- write_file = create or replace a complete file
- replace_text = precise existing-code edit
- create_directory = create folders
- copy_file = make a backup/copy
- move_file = rename/move a file
- delete_file/delete_directory = remove project items
- run_command = execute development commands
- run_python = quick Python checks
- run_tests = run or detect tests
- git_status/git_diff/git_log = inspect Git
- git_create_branch = isolate larger work
- git_commit = commit only when authorized
- workspace_info = inspect environment

WORK STYLE:
Inspect -> understand -> edit -> run -> inspect errors -> fix -> test -> verify.
Use tools instead of pretending to perform actions.
"""


async def main():
    model = choose_model()
    if not model:
        return

    server = StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    print("\nConnecting to MCP server...")

    async with client(server) as Client:
        tools_result = await Client.list_tools()

        print("\nMCP connected!")
        print("\nAvailable tools:")
        for tool in tools_result.tools:
            print(f"  • {tool.name}")

        ollama_tools = []

        for tool in tools_result.tools:
            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        print("\n================================")
        print("       LOCAL CODING AGENT")
        print("================================")
        print(f"Model: {model}")
        print(f"Tools: {len(ollama_tools)}")
        print("Workspace: agent-workspace")
        print("================================")
        print("\nType 'exit' to quit.")

        while True:
            user = input("\nYou: ").strip()

            if user.lower() == "exit":
                break

            if not user:
                continue

            messages.append(
                {
                    "role": "user",
                    "content": user,
                }
            )

            tool_round = 0
            max_tool_rounds = 40

            while True:
                tool_round += 1

                if tool_round > max_tool_rounds:
                    print("\nAI: Tool-call limit reached for this request.")
                    break

                try:
                    response = chat(
                        model=model,
                        messages=messages,
                        tools=ollama_tools,
                    )
                except Exception as e:
                    print(f"\nAI ERROR: {type(e).__name__}: {e}")
                    break

                if not response.message.tool_calls:
                    messages.append(response.message)
                    print("\nAI:", response.message.content)
                    break

                messages.append(response.message)

                for tool_call in response.message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    print(f"\n[MCP → {tool_name}]")
                    print(f"Arguments: {arguments}")

                    try:
                        result = await Client.call_tool(
                            tool_name,
                            arguments,
                        )

                        output_parts = []

                        for content in result.content:
                            if hasattr(content, "text"):
                                output_parts.append(content.text)

                        output = "\n".join(output_parts).strip()

                        if not output:
                            output = "Tool completed successfully."

                    except Exception as e:
                        output = (
                            f"Tool execution error: "
                            f"{type(e).__name__}: {e}"
                        )

                    print(f"[MCP ← {output}]")

                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": output,
                        }
                    )


if __name__ == "__main__":
    asyncio.run(main())
