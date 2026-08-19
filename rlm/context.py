from dataclasses import dataclass, field
from typing import Any


@dataclass
class RLMContext:
    """Context available to an RLM execution."""

    task: str
    content: str = ""

    depth: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def child(
        self,
        task: str,
        content: str = "",
    ) -> "RLMContext":
        """Create an isolated child context."""

        return RLMContext(
            task=task,
            content=content,
            depth=self.depth + 1,
            metadata=dict(self.metadata),
        )