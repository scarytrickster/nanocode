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
from agent.state import (
    AgentState,
    AgentStatus,
    ensure_system_message,
    get_system_prompt,
    system_message,
    user_message,
)
from agent.planner import Planner
from agent.evaluator import Evaluator
from agent.memory import Experience, Memory
from collections.abc import Callable
from langfuse import get_client

from agent.rsi import RSIContext
from rlm.router import STRATEGY_RLM, RLMRouter, RouteDecision


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
        router: RLMRouter | None = None,
        rlm_orchestrator: Any | None = None,
        rlm_enabled: bool = True,
    ) -> None:

        self.tools = tools if tools is not None else get_all_tools()
        self.config = config if config is not None else AgentConfig()

        # One shared tracer for the entire agent
        self.tracer = Tracer(
            enabled=True,
            console=console_trace,
            on_event=trace_callback,
        )

        self.langfuse = get_client()
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

        # Reflector shares the tracer
        self.reflector = Reflector(
            tracer=self.tracer
        )

        self.memory = Memory()

        # Routing layer. Child agents spawned by the RLM path are created with
        # rlm_enabled=False so a child can never route into RLM again.
        self.rlm_enabled = rlm_enabled

        self.router = (
            router
            if router is not None
            else (RLMRouter() if rlm_enabled else None)
        )

        self.rlm_orchestrator = rlm_orchestrator

        self.last_route_decision: RouteDecision | None = None

        # The RSI context of the most recent attempt, so a caller can inspect
        # what the final attempt actually learned from the ones before it.
        self.last_rsi_context: RSIContext | None = None
    
        # A supplied history is normalized, never trusted, so NanoCode's own
        # system instruction always leads the conversation.
        self.messages = (
            ensure_system_message(messages)
            if messages is not None
            else [system_message()]
        )

    def create_state(self, task: str) -> AgentState:
        """Create an AgentState for a user task."""

        # The task is appended as user content. It is never merged into, or
        # allowed to replace, the system instruction.
        self.messages.append(user_message(task))

        return AgentState(
            task=task,
            messages=self.messages,
            tools=self.tools,
            config=self.config,
        )

    def _get_rlm_orchestrator(self) -> Any:
        """Return the RLM orchestrator, creating the default one on demand."""

        if self.rlm_orchestrator is None:
            # Imported lazily: the orchestrator imports NanoCodeAgent.
            from rlm.orchestrator import RLMOrchestrator

            # The shared tracer goes down with it, so child agent events
            # surface through the same callback the CLI already listens to.
            self.rlm_orchestrator = RLMOrchestrator(tracer=self.tracer)

        return self.rlm_orchestrator

    def _route(self, task: str) -> RouteDecision | None:
        """Decide which execution path handles this task."""

        if self.router is None:
            return None

        decision = self.router.decide(task)

        self.last_route_decision = decision

        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="routing-decision",
            input={
                "task": task,
            },
        ) as span:

            span.update(
                output={
                    "task": task,
                    "strategy": decision.strategy,
                    "reason": decision.reason,
                    "score": decision.score,
                    "signals": decision.signals,
                }
            )

        self.tracer.record(
            "routing.decided",
            component="router",
            task=task,
            strategy=decision.strategy,
            score=decision.score,
            reason=decision.reason,
        )

        return decision

    def run(self, task: str) -> str:
        """Route a task, then run either the NanoCode pipeline or the RLM path."""

        langfuse = self.langfuse

        try:
            with langfuse.start_as_current_observation(
                as_type="span",
                name="nanocode-run",
                input={
                    "task": task,
                },
            ) as trace:

                decision = self._route(task)

                if decision is not None and decision.strategy == STRATEGY_RLM:

                    answer = self._get_rlm_orchestrator().run(task)

                    trace.update(
                        output={
                            "answer": answer,
                            "status": "completed",
                            "strategy": STRATEGY_RLM,
                        }
                    )

                    return answer

                state = self.create_state(task)

                experiences = self.memory.retrieve(task)

                self.tracer.record(
                    "memory.retrieved",
                    component="memory",
                    task=task,
                    matches=len(experiences),
                )

                rsi = RSIContext()
                last_reflection = None
                rsi_started = False

                self.last_rsi_context = rsi

                while True:

                    # The planner context for a retry is derived from the
                    # previous attempt's evaluation and reflection: attempt
                    # N+1 never replays attempt N's input unchanged.
                    retry_context = rsi.to_planner_context()

                    if rsi.is_retry:
                        self.tracer.record(
                            "rsi.retry.started",
                            component="rsi",
                            **rsi.summary(),
                        )

                    self.planner.run(
                        state,
                        experiences=experiences,
                        retry_context=retry_context,
                    )

                    self.executor.run(state)

                    if state.status == AgentStatus.HUMAN_REJECTED:
                        self.tracer.record(
                            "agent.stopped",
                            component="agent",
                            reason="human_rejected",
                        )

                        trace.update(
                            output={
                                "answer": state.final_response,
                                "status": "human_rejected",
                            }
                        )

                        return state.final_response

                    evaluation = self.evaluator.evaluate(state)

                    reflection = self.reflector.reflect(
                        state,
                        evaluation,
                    )

                    last_reflection = reflection

                    if evaluation.success:
                        if state.retry_count > 0:
                            self.tracer.record(
                                "retry.completed",
                                component="agent",
                                attempt=state.retry_count,
                                status="success",
                            )

                            self.tracer.record(
                                "rsi.completed",
                                component="rsi",
                                success=True,
                                **rsi.summary(),
                            )

                        break

                    # The next attempt's context, derived from this attempt's
                    # failure. Built before the limit checks so the reason a
                    # run stopped improving is always inspectable.
                    improved = rsi.next_attempt(
                        evaluation=evaluation,
                        reflection=reflection,
                        response=state.final_response,
                    )

                    self.tracer.record(
                        "rsi.reflection.completed",
                        component="rsi",
                        attempt=rsi.attempt,
                        has_improvement=improved.has_improvement,
                        should_improve=bool(
                            getattr(reflection, "should_improve", False)
                        ),
                    )

                    if state.retry_count >= state.config.max_retries:
                        self.tracer.record(
                            "retry.exhausted",
                            component="agent",
                            attempts=state.retry_count,
                        )

                        self.tracer.record(
                            "rsi.exhausted",
                            component="rsi",
                            attempts=rsi.attempt,
                            max_retries=state.config.max_retries,
                        )
                        break

                    if not reflection.should_improve:
                        break

                    if not rsi_started:
                        rsi_started = True

                        self.tracer.record(
                            "rsi.started",
                            component="rsi",
                            attempt=rsi.attempt,
                            reason="evaluation_failed",
                            has_improvement=improved.has_improvement,
                        )

                    # RSI shares the existing retry budget; it never keeps a
                    # counter of its own.
                    state.retry_count += 1

                    rsi = improved

                    self.last_rsi_context = rsi

                    self.tracer.record(
                        "retry.started",
                        component="agent",
                        attempt=state.retry_count,
                        reason="evaluation_failed",
                    )

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

                trace.update(
                    output={
                        "answer": state.final_response,
                        "status": state.status,
                        "success": evaluation.success,
                    }
                )

                return state.final_response

        finally:
            langfuse.flush()


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

