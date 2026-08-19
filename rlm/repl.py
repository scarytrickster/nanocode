from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm.call import RLMCallHandler
from rlm.context import RLMContext
from rlm.result import RLMResult


@dataclass
class RLMREPL:
    """Controlled programmatic environment for an RLM."""

    context: RLMContext
    call_handler: RLMCallHandler | None = None

    _variables: dict[str, Any] = field(default_factory=dict)

    def inspect(self, start: int = 0, end: int | None = None) -> str:
        content = self.context.content

        if end is None:
            end = len(content)

        return content[start:end]

    def search(self, query: str) -> list[int]:
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
        return self.context.content[start:end]

    def set(self, name: str, value: Any) -> None:
        self._variables[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self._variables.get(name, default)

    def variables(self) -> dict[str, Any]:
        return dict(self._variables)

    def call(
        self,
        task: str,
        content: str = "",
    ) -> RLMResult:
        """Make a controlled recursive call."""

        if self.call_handler is None:
            raise RuntimeError(
                "Recursive calls are not available in this REPL"
            )

        child_context = self.context.child(
            task=task,
            content=content,
        )

        return self.call_handler.call(child_context)