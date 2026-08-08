from pyexpat.errors import messages
import sys
from dataclasses import dataclass, field
from typing import Any

from agent import state
from agent.tracer import Tracer
from config.settings import MODEL, client
from models.config import AgentConfig
from tools import get_all_tools
from tools.base import Tool

from agent.executor import Executor
from agent.state import get_system_prompt
from agent.planner import Planner
from agent.evaluator import Evaluator


@dataclass
class AgentState:
    """Mutable state for one agent run."""

    task: str
    messages: list[dict[str, Any]]
    tools: list[Tool]
    config: AgentConfig
    iteration: int = 0
    status: str = "running"
    final_response: str = ""
    tools_by_name: dict[str, Tool] = field(init=False)
    tool_schemas: list[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.tool_schemas = [tool.to_schema() for tool in self.tools]


class NanoCodeAgent:
    """High-level agent facade following create_state -> executor -> evaluator -> response."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        config: AgentConfig | None = None,
        executor: Executor | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:

        self.tools = tools if tools is not None else get_all_tools()
        self.config = config if config is not None else AgentConfig()

        # One shared tracer for the entire agent
        self.tracer = Tracer()

        # Executor shares the tracer
        self.executor = (
            executor
            if executor is not None
            else Executor(tracer=self.tracer)
        )

        # Planner shares the tracer
        self.planner = Planner(
            tracer=self.tracer
        )

        # Evaluator shares the tracer
        self.evaluator = Evaluator(
            tracer=self.tracer
        )

        self.messages = (
            messages
            if messages is not None
            else [
                {
                    "role": "system",
                    "content": get_system_prompt(),
                }
            ]
        )

    def create_state(self, task: str) -> AgentState:
        """Create an AgentState for a user task."""

        self.messages.append(
            {
                "role": "user",
                "content": task,
            }
        )

        return AgentState(
            task=task,
            messages=self.messages,
            tools=self.tools,
            config=self.config,
        )

    def run(self, task: str) -> str:
        """Run a task, evaluate the result, and return the final response."""

        state = self.create_state(task)

        self.planner.run(state)

        self.executor.run(state)

        evaluation = self.evaluator.evaluate(state)

        return state.final_response

def run_agent(
    messages: list[dict[str, Any]],
    tools: list[Tool],
    config: AgentConfig = None,
) -> str:
    """Compatibility wrapper for the older functional API."""
    if config is None:
        config = AgentConfig()

    state = AgentState(
        task="",
        messages=messages,
        tools=tools,
        config=config,
    )
    executor = Executor()
    executor.run(state)
    return state.final_response

def print_banner() -> None:
    """Print the application banner."""
    print("+----------------------------------------------+")
    print("| nanocode - a tiny coding agent               |")
    print("| /plan toggles plan mode - ctrl-c/ctrl-d quits |")
    print("+----------------------------------------------+")


def main() -> None:
    """Main entry point for the agent."""
    print_banner()

    plan_mode = False
    agent = NanoCodeAgent()

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

            agent.config = AgentConfig(plan_mode=plan_mode)
            agent.run(user_input)

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
