from abc import ABC, abstractmethod

from rlm.context import RLMContext
from rlm.result import RLMResult


class RLMCallHandler(ABC):
    """Interface used by the REPL to make recursive calls."""

    @abstractmethod
    def call(self, context: RLMContext) -> RLMResult:
        """Execute a recursive child call."""
        raise NotImplementedError