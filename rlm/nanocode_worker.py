from agent.agent import NanoCodeAgent
from rlm.context import RLMContext
from rlm.result import RLMResult
from rlm.worker import RLMWorker


class NanoCodeRLMWorker(RLMWorker):
    """Runs a focused RLM task using NanoCode."""

    def __init__(self, agent: NanoCodeAgent) -> None:
        self.agent = agent

    def run(self, context: RLMContext) -> RLMResult:
        response = self.agent.run(context.task)

        return RLMResult(
            answer=response,
            success=True,
            depth=context.depth,
            children_created=0,
        )