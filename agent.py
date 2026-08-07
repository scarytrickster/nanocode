#!/usr/bin/env python3
"""
nanocode - A tiny terminal coding agent.

A minimalist coding agent that can read, write, and edit files,
run shell commands, search the web, and manage tasks.
"""

import json
import os
import platform
import sys
from typing import Any, Optional

from dotenv import load_dotenv

from config.settings import MODEL, client
from models.config import AgentConfig, ToolCall
from tools import get_all_tools
from tools.base import Tool

load_dotenv()


def get_system_prompt() -> str:
    """Generate the system prompt with environment info."""
    prompt = (
        "You are nanocode, a terminal coding agent. Be concise. Prefer tools over guessing.\n"
        "Use the todo_write tool to plan any task with more than a couple of steps.\n\n"
        f"Environment:\n"
        f"cwd: {os.getcwd()}\n"
        f"os: {platform.system()} {platform.release()}\n"
        f"python: {sys.version}\n"
        f"files in cwd: {', '.join(sorted(os.listdir()))}"
    )

    if os.path.exists("NANOCODE.md"):
        try:
            with open("NANOCODE.md", encoding="utf-8") as f:
                prompt += f"\n\nProject instructions:\n{f.read()}"
        except Exception:
            pass

    return prompt


def parse_tool_calls(stream) -> tuple[str, list[ToolCall], Optional[str]]:
    """Parse streaming response for content and tool calls."""
    reply = ""
    tool_calls: list[ToolCall] = []
    finish_reason = None

    for chunk in stream:
        choice = chunk.choices[0]

        if choice.delta.content:
            print(choice.delta.content, end="", flush=True)
            reply += choice.delta.content

        for tc in choice.delta.tool_calls or []:
            if tc.index >= len(tool_calls):
                tool_calls.append(ToolCall(id="", name=""))

            call = tool_calls[tc.index]
            call.id += tc.id or ""
            call.name += tc.function.name or ""
            call.arguments += tc.function.arguments or ""

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    print()
    return reply, tool_calls, finish_reason


def get_user_approval(tool: Tool, args: dict[str, Any]) -> bool:
    """Prompt user for approval before executing a write tool."""
    try:
        answer = input(f"{tool.name}({json.dumps(args)}) [y/n] ")
        return answer.strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def run_agent(
    messages: list[dict[str, Any]],
    tools: list[Tool],
    config: AgentConfig = None,
) -> str:
    """Run the agent loop with the given messages and tools."""
    if config is None:
        config = AgentConfig()

    tools_by_name = {t.name: t for t in tools}
    tool_schemas = [t.to_schema() for t in tools]

    for _ in range(config.max_iterations):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tool_schemas,
                stream=True,
            )
        except Exception as e:
            print(f"\nError calling API: {e}", file=sys.stderr)
            return f"Error: {e}"

        reply, tool_calls, finish_reason = parse_tool_calls(stream)

        if finish_reason == "tool_calls" and tool_calls:
            assistant_msg = {
                "role": "assistant",
                "content": reply,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in tool_calls:
                try:
                    args = json.loads(tc.arguments)
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error parsing arguments: {e}",
                    })
                    continue

                tool = tools_by_name.get(tc.name)
                if not tool:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: Unknown tool {tc.name}",
                    })
                    continue

                if config.plan_mode and not tool.is_read_only:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            "Plan mode is on: write tools are disabled. "
                            "Present a plan and ask the user to approve it."
                        ),
                    })
                    continue

                if not tool.is_read_only and not config.auto_approve:
                    if not get_user_approval(tool, args):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "User denied the tool call.",
                        })
                        continue

                try:
                    result = tool.execute(args)
                except Exception as e:
                    result = f"Error executing {tc.name}: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            messages.append({"role": "assistant", "content": reply})
            return reply

    return reply


def print_banner():
    """Print the application banner."""
    print("┌──────────────────────────────────────────────┐")
    print("│ nanocode — a tiny coding agent               │")
    print("│ /plan toggles plan mode · ctrl-c/ctrl-d quits │")
    print("└──────────────────────────────────────────────┘")


def main():
    """Main entry point for the agent."""
    print_banner()

    plan_mode = False
    messages = [{"role": "system", "content": get_system_prompt()}]
    tools = get_all_tools()

    while True:
        try:
            prompt = "plan > " if plan_mode else "> "
            user_input = input(prompt)

            if user_input.strip() == "/plan":
                plan_mode = not plan_mode
                print(f"Plan mode {'on' if plan_mode else 'off'}")
                continue

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})
            config = AgentConfig(plan_mode=plan_mode)
            run_agent(messages, tools, config=config)

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
