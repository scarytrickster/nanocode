
"""
agent/executor.py

Responsible for running the reasoning/tool execution loop.

The Executor is the only component that talks directly to the LLM
and executes tool calls.

It DOES NOT:
- know about the CLI
- know about planning
- know about memory
- know about reflection
"""

from __future__ import annotations

import json
from typing import Any

from config.settings import MODEL, client
from models.config import ToolCall
from agent.state import AgentState,AgentStatus


class Executor:
    """
    Executes one complete agent task.
    """

    def __init__(self):
        self.client = client

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def run(self, state: AgentState) -> None:
        """
        Execute the agent until it produces a final answer.
        """

        state.status = AgentStatus.RUNNING

        tools_by_name = {
            tool.name: tool
            for tool in state.tools
        }

        tool_schemas = [
            tool.to_schema()
            for tool in state.tools
        ]

        reply = ""

        while state.iteration < state.config.max_iterations:

            state.iteration += 1

            stream = self.client.chat.completions.create(
                model=MODEL,
                messages=state.messages,
                tools=tool_schemas,
                stream=True,
            )

            reply, tool_calls, finish_reason = self._parse_stream(stream)

            # -------------------------------------------------
            # Tool Calls
            # -------------------------------------------------

            if finish_reason == "tool_calls" and tool_calls:

                assistant_message = {
                    "role": "assistant",
                    "content": reply,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }

                state.messages.append(assistant_message)

                for tool_call in tool_calls:

                    result = self._execute_tool(
                        tool_call,
                        tools_by_name,
                    )

                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

                continue

            # -------------------------------------------------
            # Final Response
            # -------------------------------------------------

            state.messages.append(
                {
                    "role": "assistant",
                    "content": reply,
                }
            )

            state.final_response = reply
            state.status = AgentStatus.COMPLETED

            return 

        state.status = AgentStatus.FAILED
        return 

    # ---------------------------------------------------------
    # Stream Parsing
    # ---------------------------------------------------------

    def _parse_stream(
        self,
        stream,
    ) -> tuple[str, list[ToolCall], str | None]:

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
                    tool_calls.append(
                        ToolCall(
                            id="",
                            name="",
                        )
                    )

                call = tool_calls[tc.index]

                call.id += tc.id or ""

                call.name += tc.function.name or ""

                call.arguments += tc.function.arguments or ""

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        print()

        return reply, tool_calls, finish_reason

    # ---------------------------------------------------------
    # Tool Execution
    # ---------------------------------------------------------

    def _execute_tool(
        self,
        tool_call: ToolCall,
        tools_by_name: dict[str, Any],
    ) -> str:

        try:

            args = json.loads(tool_call.arguments)

        except Exception as e:

            return f"Error parsing arguments: {e}"

        tool = tools_by_name.get(tool_call.name)

        if tool is None:

            return f"Unknown tool: {tool_call.name}"

        if (
            self._plan_mode(tool, args)
        ):

            return (
                "Plan mode enabled. "
                "Write tools are disabled."
            )

        try:

            return tool.execute(args)

        except Exception as e:

            return (
                f"Error executing "
                f"{tool_call.name}: {e}"
            )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _plan_mode(
        self,
        tool,
        args,
    ) -> bool:

        del args

        return False
