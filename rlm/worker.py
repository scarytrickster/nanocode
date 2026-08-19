from abc import ABC, abstractmethod

from rlm.context import RLMContext
from rlm.result import RLMResult


class RLMWorker(ABC):
    """Base interface for a recursive worker."""

    @abstractmethod
    def run(self, context: RLMContext) -> RLMResult:
        """Process one focused recursive task."""
        raise NotImplementedError