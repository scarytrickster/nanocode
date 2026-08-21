from __future__ import annotations

from typing import Any, Callable

from langfuse import get_client

from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.decomposer import (
    DeterministicRLMDecomposer,
    RLMChildTask,
    RLMDecomposer,
)
from rlm.nanocode_handler import NanoCodeCallHandler, create_nanocode_agent
from rlm.result import RLMResult
from rlm.runtime import RLMRuntime
from rlm.synthesizer import (
    FAILED_KEY,
    PARTIAL_KEY,
    RATE_LIMITED_KEY,
    RLMSynthesizer,
    SUCCESSFUL_KEY,
)

# Upper bound on the children the default runtime is willing to run, mirroring
# RLMBudget's own default. The runtime stays the authority on execution
# limits; a decomposer may propose more candidates than this.
DEFAULT_MAX_CHILDREN = RLMBudget().max_children


class RLMOrchestrator:
    """
    Smallest orchestration layer joining the router's `rlm` decision to the
    existing RLM machinery.

    It owns nothing that already exists: decomposition goes through
    RLMDecomposer, child execution through NanoCodeCallHandler and RLMRuntime,
    and combination through RLMSynthesizer.
    """

    def __init__(
        self,
        runtime: RLMRuntime | None = None,
        decomposer: RLMDecomposer | None = None,
        synthesizer: RLMSynthesizer | None = None,
        agent_factory: Callable[[], Any] | None = None,
    ) -> None:

        self.runtime = runtime
        self.decomposer = decomposer or DeterministicRLMDecomposer()
        self.synthesizer = synthesizer or RLMSynthesizer()

        self.agent_factory = (
            agent_factory
            if agent_factory is not None
            else create_nanocode_agent
        )

        self.langfuse = get_client()

        self.last_result: RLMResult | None = None
        self.last_child_tasks: list[RLMChildTask] = []

    def _create_runtime(self, child_count: int) -> RLMRuntime:
        """Create a runtime with a budget sized for one user request."""

        max_children = max(1, min(child_count, DEFAULT_MAX_CHILDREN))

        return RLMRuntime(
            call_handler=NanoCodeCallHandler(
                agent_factory=self.agent_factory,
            ),
            budget=RLMBudget(
                max_depth=2,
                max_children=max_children,
                max_iterations=max_children + 1,
            ),
        )

    def _decompose(self, context: RLMContext) -> list[RLMChildTask]:
        """Decompose the parent task, recorded as a span under the RLM trace."""

        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="rlm-decomposition",
            input={"task": context.task, "depth": context.depth},
        ) as span:

            children = list(
                self.decomposer.decompose(
                    task=context.task,
                    context=context,
                )
            )

            span.update(
                output={
                    "child_count": len(children),
                    "child_tasks": [child.task for child in children],
                }
            )

            return children

    def run(self, task: str) -> str:
        """
        Execute the RLM path for a task and return a NanoCode-style answer.

        The return type matches NanoCodeAgent.run(): a plain string.
        """

        context = RLMContext(task=task)

        child_tasks = self._decompose(context)

        self.last_child_tasks = list(child_tasks)

        if not child_tasks:
            # Nothing to run. Fail through the existing synthesis contract
            # rather than inventing a second failure model.
            result = self.synthesizer.synthesize([])

            self.last_result = result

            return self._to_answer(result, task)

        # A fresh runtime per request keeps the recursion budget per-request;
        # an injected runtime is always honoured.
        runtime = (
            self.runtime
            if self.runtime is not None
            else self._create_runtime(len(child_tasks))
        )

        result = runtime.call_and_synthesize(
            parent=context,
            tasks=child_tasks,
            synthesizer=self.synthesizer,
        )

        self.last_result = result

        return self._to_answer(result, task)

    def _to_answer(self, result: RLMResult, task: str) -> str:
        """Convert an RLMResult into the string NanoCode callers expect.

        The RLMResult.answer stays exactly what the synthesizer produced; the
        completeness notice is added here, on the user-facing string, so a
        partial investigation can never read as a complete one.
        """

        if result is None:
            return ""

        if result.answer:
            notice = self._completeness_notice(result)

            return f"{notice}{result.answer}" if notice else str(result.answer)

        if result.success:
            return ""

        return self._failure_answer(result, task)

    def _completeness_notice(self, result: RLMResult) -> str:
        """A short header stating how much of the investigation completed."""

        metadata = result.metadata or {}

        if not metadata.get(PARTIAL_KEY):
            return ""

        successful = metadata.get(SUCCESSFUL_KEY, 0)
        total = successful + metadata.get(FAILED_KEY, 0)
        rate_limited = metadata.get(RATE_LIMITED_KEY, 0)

        lines = [
            "Partial investigation: "
            f"{successful} of {total} child analyses completed successfully."
        ]

        if rate_limited:
            lines.append(
                f"{rate_limited} child "
                f"{'analysis was' if rate_limited == 1 else 'analyses were'} "
                "rate-limited by the model provider."
            )

        lines.append("")

        return "\n".join(lines) + "\n"

    def _failure_answer(self, result: RLMResult, task: str) -> str:
        """The message returned when no child produced an answer."""

        metadata = result.metadata or {}

        rate_limited = metadata.get(RATE_LIMITED_KEY, 0)

        if rate_limited:
            failed = metadata.get(FAILED_KEY, rate_limited)

            return (
                f"RLM execution produced no answer for task: {task} "
                f"({rate_limited} of {failed} child analyses were "
                "rate-limited by the model provider; try again shortly)."
            )

        return f"RLM execution produced no answer for task: {task}"
