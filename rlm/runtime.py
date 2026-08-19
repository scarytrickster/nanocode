from __future__ import annotations

from rlm.budget import RLMBudget
from rlm.call import RLMCallHandler
from rlm.context import RLMContext
from rlm.result import RLMResult
from rlm.synthesizer import RLMSynthesizer


class RLMRuntime:
    """Runtime responsible for controlled recursive RLM execution."""

    def __init__(
        self,
        call_handler: RLMCallHandler,
        budget: RLMBudget | None = None,
    ) -> None:
        self.call_handler = call_handler
        self.budget = budget or RLMBudget()

    def call(
        self,
        parent: RLMContext,
        task: str,
        content: str = "",
    ) -> RLMResult:
        """Execute one controlled child RLM call."""

        child = parent.child(
            task=task,
            content=content,
        )

        if not self.budget.can_spawn_child(child.depth):
            raise RuntimeError(
                "RLM recursion budget exceeded"
            )

        self.budget.consume_child()
        self.budget.consume_iteration()

        return self.call_handler.call(child)

    def call_many(
        self,
        parent: RLMContext,
        tasks: list[tuple[str, str]],
    ) -> list[RLMResult]:
        """
        Execute multiple child RLM calls.

        Each tuple contains:
            (task, content)
        """

        results: list[RLMResult] = []

        for task, content in tasks:
            result = self.call(
                parent=parent,
                task=task,
                content=content,
            )

            results.append(result)

        return results

    def call_and_synthesize(
        self,
        parent: RLMContext,
        tasks: list[tuple[str, str]],
        synthesizer: RLMSynthesizer | None = None,
    ) -> RLMResult: 
        """
        Execute multiple child RLM calls and synthesize their results
        into a single parent result.
        """

        results = self.call_many(
            parent=parent,
            tasks=tasks,
        )

        if synthesizer is None:
            synthesizer = RLMSynthesizer()

        return synthesizer.synthesize(results)