from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from rlm.context import RLMContext


@dataclass
class RLMREPL:
    """
    Programmatic environment available to an RLM.

    The REPL provides controlled access to the current context and
    exposes operations that the RLM can use to inspect and manipulate
    that context.
    """

    context: RLMContext

    _variables: dict[str, Any] = field(default_factory=dict)

    def inspect(self, start: int = 0, end: int | None = None) -> str:
        """Return a section of the current context."""

        content = self.context.content

        if end is None:
            end = len(content)

        return content[start:end]

    def search(self, query: str) -> list[int]:
        """
        Find occurrences of a string in the current context.

        Returns the character offsets where matches begin.
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
        """Return a specific context slice."""

        return self.context.content[start:end]

    def set(self, name: str, value: Any) -> None:
        """Store a temporary REPL variable."""

        self._variables[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Retrieve a temporary REPL variable."""

        return self._variables.get(name, default)

    def variables(self) -> dict[str, Any]:
        """Return a copy of the current REPL variables."""

        return dict(self._variables)