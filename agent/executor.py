import json
from typing import Any, Optional

from models.config import AgentConfig, ToolCall
from tools.base import Tool


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


def execute_tool_calls(
    tool_calls: list[ToolCall],
    messages: list[dict[str, Any]],
    tools_by_name: dict[str, Tool],
    config: AgentConfig,
) -> None:
    """Execute tool calls and append results to the conversation."""
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
