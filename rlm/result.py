from dataclasses import dataclass, field
from typing import Any


@dataclass
class RLMResult:
    """Result returned by an RLM worker."""

    answer: str = ""
    success: bool = False

    depth: int = 0
    children_created: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)