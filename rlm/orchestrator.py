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
from rlm.synthesizer import RLMSynthesizer

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
        """Convert an RLMResult into the string NanoCode callers expect."""

        if result is None:
            return ""

        if result.answer:
            return str(result.answer)

        if result.success:
            return ""

        return f"RLM execution produced no answer for task: {task}"
