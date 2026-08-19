from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.controller import RLMController
from rlm.result import RLMResult
from rlm.worker import RLMWorker


class FakeWorker(RLMWorker):

    def run(self, context: RLMContext) -> RLMResult:
        return RLMResult(
            answer=f"Processed: {context.task}",
            success=True,
            depth=context.depth,
        )


def test_controller_runs_worker():

    controller = RLMController(
        worker=FakeWorker(),
    )

    context = RLMContext(
        task="Analyze this project",
    )

    result = controller.run(context)

    assert result.success is True
    assert result.answer == "Processed: Analyze this project"
    assert result.depth == 0


def test_child_context_increments_depth():

    parent = RLMContext(
        task="Parent task",
    )

    child = parent.child(
        task="Child task",
    )

    assert parent.depth == 0
    assert child.depth == 1


def test_budget_limits_iterations():

    budget = RLMBudget(
        max_iterations=1,
    )

    controller = RLMController(
        worker=FakeWorker(),
        budget=budget,
    )

    context = RLMContext(
        task="Test",
    )

    controller.run(context)

    try:
        controller.run(context)
        assert False, "Expected budget failure"
    except RuntimeError as exc:
        assert "iteration budget exceeded" in str(exc)