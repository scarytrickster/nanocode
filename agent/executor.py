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

from agent import state
from agent.approval import ApprovalManager, ApprovalRejected
from config.settings import MODEL, client
from models.config import ToolCall
from agent.state import AgentState, AgentStatus
from agent.tracer import Tracer


class Executor:
    """
    Executes one complete agent task.
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        approval: ApprovalManager | None = None,
    ):
        self.client = client
        self.tracer = tracer or Tracer()
        self.approval = approval or ApprovalManager()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def run(self, state: AgentState) -> str:
        """
        Execute the agent until it produces a final answer
        or human approval stops execution.
        """

        state.status = AgentStatus.RUNNING

        with self.tracer.span(
            "executor",
            component="agent",
            task=state.task,
        ):
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

                execution_messages = self._build_messages(state)

                with self.tracer.span(
                    "llm",
                    component="executor",
                    iteration=state.iteration,
                ):
                    stream = self.client.chat.completions.create(
                        model=MODEL,
                        messages=execution_messages,
                        tools=tool_schemas,
                        stream=True,
                    )

                    reply, tool_calls, finish_reason = (
                        self._parse_stream(stream)
                    )

                self.tracer.record(
                    "llm.response",
                    component="executor",
                    iteration=state.iteration,
                    finish_reason=finish_reason,
                    tool_calls=len(tool_calls),
                )

                # -------------------------------------------------
                # Tool calls
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
                        try:
                            with self.tracer.span(
                                "tool",
                                component="executor",
                                tool=tool_call.name,
                            ):
                                result = self._execute_tool(
                                    tool_call,
                                    tools_by_name,
                                )

                        except ApprovalRejected as e:
                            state.status = AgentStatus.HUMAN_REJECTED

                            self.tracer.record(
                                "execution.stopped",
                                component="executor",
                                reason="human_rejected",
                                tool=tool_call.name,
                            )
        
                            raise

                        state.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )

                    continue

                # -------------------------------------------------
                # Final response
                # -------------------------------------------------

                state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                    }
                )

                state.final_response = reply
                state.status = AgentStatus.COMPLETED

                return reply

            # -----------------------------------------------------
            # Max iterations reached
            # -----------------------------------------------------

            state.status = AgentStatus.MAX_ITERATIONS

            self.tracer.record(
                "execution.max_iterations",
                component="executor",
                reason="max_iterations",
            )

            return reply

    # ---------------------------------------------------------
    # Message Building
    # ---------------------------------------------------------

    def _build_messages(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        messages = list(state.messages)

        if state.plan:
            plan_text = "\n".join(
                f"{i}. {step}"
                for i, step in enumerate(state.plan, start=1)
            )

            plan_message = {
                "role": "system",
                "content": (
                    "Execution plan for the current task:\n\n"
                    f"{plan_text}\n\n"
                    "Use this plan to guide your execution. "
                    "Adapt the plan when necessary based on tool results."
                ),
            }

            messages.insert(1, plan_message)

        return messages

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

            # Do NOT print here.
            # The CLI renderer is responsible for presentation.
            if choice.delta.content:
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

        return reply, tool_calls, finish_reason

    # ---------------------------------------------------------
    # Tool Execution
    # ---------------------------------------------------------

    def _execute_tool(
        self,
        tool_call: ToolCall,
        tools_by_name: dict[str, Any],
    ) -> str:

        # -----------------------------------------------------
        # Parse arguments
        # -----------------------------------------------------

        try:
            args = json.loads(tool_call.arguments)

        except Exception as e:
            self.tracer.record(
                "tool.failed",
                component="executor",
                tool=tool_call.name,
                error=f"argument_parse_error: {e}",
            )

            return f"Error parsing arguments: {e}"

        # -----------------------------------------------------
        # Find tool
        # -----------------------------------------------------

        tool = tools_by_name.get(tool_call.name)

        if tool is None:
            self.tracer.record(
                "tool.failed",
                component="executor",
                tool=tool_call.name,
                error="unknown_tool",
            )

            return f"Unknown tool: {tool_call.name}"

        # -----------------------------------------------------
        # Plan mode
        # -----------------------------------------------------

        if self._plan_mode(tool, args):
            self.tracer.record(
                "tool.rejected",
                component="executor",
                tool=tool_call.name,
                reason="plan_mode",
            )

            return (
                "Plan mode enabled. "
                "Write tools are disabled."
            )

        # -----------------------------------------------------
        # Human approval
        # -----------------------------------------------------

        approved = self.approval.approve_tool(
            tool_call.name,
            args,
        )

        if not approved:
            self.tracer.record(
                "tool.rejected",
                component="executor",
                tool=tool_call.name,
                reason="human_rejected",
            )

            raise ApprovalRejected(
                f"Tool execution rejected by human: "
                f"{tool_call.name}"
            )

        # -----------------------------------------------------
        # Execute tool
        # -----------------------------------------------------

        self.tracer.record(
            "tool.approved",
            component="executor",
            tool=tool_call.name,
        )

        try:
            result = tool.execute(args)

            self.tracer.record(
                "tool.executed",
                component="executor",
                tool=tool_call.name,
            )

            return result

        except Exception as e:
            self.tracer.record(
                "tool.failed",
                component="executor",
                tool=tool_call.name,
                error=str(e),
            )

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