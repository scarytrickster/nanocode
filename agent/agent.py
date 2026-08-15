from pyexpat.errors import messages
import sys
from typing import Any

from agent import state
from agent.reflector import Reflector
from agent.tracer import Tracer
from config.settings import MODEL, client
from models.config import AgentConfig
from tools import get_all_tools
from tools.base import Tool

from agent.executor import Executor
from agent.state import AgentState, AgentStatus,get_system_prompt
from agent.planner import Planner
from agent.evaluator import Evaluator
from agent.memory import Experience, Memory
from collections.abc import Callable
from agent.approval import ApprovalManager, ApprovalRejected


# @dataclass
# class AgentState:
#     """Mutable state for one agent run."""

#     task: str
#     messages: list[dict[str, Any]]
#     tools: list[Tool]
#     config: AgentConfig
#     iteration: int = 0
#     status: str = "running"
#     final_response: str = ""
#     tools_by_name: dict[str, Tool] = field(init=False)
#     tool_schemas: list[dict[str, Any]] = field(init=False)

#     def __post_init__(self) -> None:
#         self.tools_by_name = {tool.name: tool for tool in self.tools}
#         self.tool_schemas = [tool.to_schema() for tool in self.tools]


class NanoCodeAgent:
    """High-level agent facade following create_state -> executor -> evaluator -> response."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        config: AgentConfig | None = None,
        executor: Executor | None = None,
        messages: list[dict[str, Any]] | None = None,
        console_trace: bool = True,
        trace_callback: Callable[[Any], None] | None = None,
    ) -> None:

        self.tools = tools if tools is not None else get_all_tools()
        self.config = config if config is not None else AgentConfig()

        self.approval = ApprovalManager(
            mode=self.config.approval_mode
        )

        # One shared tracer for the entire agent
        self.tracer = Tracer(
            enabled=True,
            console=console_trace,
            on_event=trace_callback,
        )
        # Executor shares the tracer
        self.executor = (
            executor
            if executor is not None
            else Executor(
                tracer=self.tracer,
                approval=self.approval,
            )
        )

        # Planner shares the tracer
        self.planner = Planner(
            tracer=self.tracer
        )

        # Evaluator shares the tracer
        self.evaluator = Evaluator(
            tracer=self.tracer
        )

        # Reflector shares the tracer
        self.reflector = Reflector(
            tracer=self.tracer
        )

        self.memory = Memory()
    
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
        """Run a task with evaluation-driven retries."""

        state = self.create_state(task)

        # Retrieve relevant previous experiences once.
        experiences = self.memory.retrieve(task)

        self.tracer.record(
            "memory.retrieved",
            component="memory",
            task=task,
            matches=len(experiences),
        )

        retry_context: dict[str, str] | None = None
        last_reflection = None

        while True:

            # Plan using memory + optional retry context.
            self.planner.run(
                state,
                experiences=experiences,
                retry_context=retry_context,
            )

            # Execute the current attempt.
            try:
                self.executor.run(state)

            except ApprovalRejected:
                state.status = AgentStatus.HUMAN_REJECTED

                self.tracer.record(
                    "agent.stopped",
                    component="agent",
                    reason="human_rejected",
                )

                state.final_response = (
                    "Execution stopped: "
                    "the requested tool action was rejected by the human."
                )
    
                return state.final_response

            # Evaluate the attempt.
            evaluation = self.evaluator.evaluate(state)

            # Always reflect after evaluation.
            reflection = self.reflector.reflect(
                state,
                evaluation,
            )

            last_reflection = reflection

            # Successful execution → stop immediately.
            if evaluation.success:
                if state.retry_count > 0:
                    self.tracer.record(
                        "retry.completed",
                        component="agent",
                        attempt=state.retry_count,
                        status="success",
                    )

                break

            # Check retry limit.
            if state.retry_count >= state.config.max_retries:
                self.tracer.record(
                    "retry.exhausted",
                    component="agent",
                    attempts=state.retry_count,
                )
                break

            # Reflection must provide an improvement before retrying.
            if not reflection.should_improve:
                break

            # Build context for the next planning attempt.
            retry_context = {
                "diagnosis": reflection.diagnosis,
                "improvement": reflection.improvement,
            }

            state.retry_count += 1

            self.tracer.record(
                "retry.started",
                component="agent",
                attempt=state.retry_count,
                reason="evaluation_failed",
            )


        # Store one experience only after the final attempt.
        if (
            not evaluation.success
            and last_reflection is not None
            and last_reflection.should_improve
        ):
            experience = Experience(
                task=state.task,
                diagnosis=last_reflection.diagnosis,
                improvement=last_reflection.improvement,
                success=False,
            )
    
            self.memory.add(experience)
    
            self.tracer.record(
                "memory.stored",
                component="memory",
                task=state.task,
                success=False,
            )

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

