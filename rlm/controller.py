from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.result import RLMResult
from rlm.synthesizer import RLMSynthesizer
from rlm.worker import RLMWorker


class RLMController:
    """Controls recursive language-model execution."""

    def __init__(
        self,
        worker: RLMWorker,
        synthesizer: RLMSynthesizer | None = None,
        budget: RLMBudget | None = None,
    ) -> None:

        self.worker = worker

        self.synthesizer = (
            synthesizer
            if synthesizer is not None
            else RLMSynthesizer()
        )

        self.budget = (
            budget
            if budget is not None
            else RLMBudget()
        )

    def run(
        self,
        context: RLMContext,
    ) -> RLMResult:
        """Run one RLM step."""

        self.budget.consume_iteration()

        return self.worker.run(context)