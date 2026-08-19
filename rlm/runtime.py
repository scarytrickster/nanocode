from __future__ import annotations

from rlm.budget import RLMBudget
from rlm.call import RLMCallHandler
from rlm.context import RLMContext
from rlm.result import RLMResult


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