from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.context import RLMContext
from rlm.result import RLMResult
from rlm.runtime import RLMRuntime


@dataclass
class RLMREPL:
    """
    Controlled programmatic environment available to an RLM.

    The REPL provides access to the current context and allows
    controlled recursive calls through RLMRuntime.
    """

    context: RLMContext
    runtime: RLMRuntime | None = None

    _variables: dict[str, Any] = field(default_factory=dict)

    def inspect(
        self,
        start: int = 0,
        end: int | None = None,
    ) -> str:
        """Return a section of the current context."""

        content = self.context.content

        if end is None:
            end = len(content)

        return content[start:end]

    def search(self, query: str) -> list[int]:
        """
        Find occurrences of a string in the current context.

        Returns character offsets where matches begin.
        """

        if not query:
            return []

        content = self.context.content
        positions: list[int] = []

        start = 0

        while True:
            position = content.find(query, start)

            if position == -1:
                break

            positions.append(position)
            start = position + len(query)

        return positions

    def slice(self, start: int, end: int) -> str:
        """Return a specific section of the context."""

        return self.context.content[start:end]

    def set(self, name: str, value: Any) -> None:
        """Store a temporary REPL variable."""

        self._variables[name] = value

    def get(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """Retrieve a temporary REPL variable."""

        return self._variables.get(name, default)

    def variables(self) -> dict[str, Any]:
        """Return a copy of the current REPL variables."""

        return dict(self._variables)

    def call(
        self,
        task: str,
        content: str = "",
    ) -> RLMResult:
        """
        Make a controlled recursive RLM call.

        Recursive execution must go through RLMRuntime so that
        depth, child-count, and iteration budgets are enforced.
        """

        if self.runtime is None:
            raise RuntimeError(
                "Recursive calls are not available in this REPL"
            )

        return self.runtime.call(
            parent=self.context,
            task=task,
            content=content,
        )