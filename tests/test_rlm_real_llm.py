from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.nanocode_handler import (
    NanoCodeCallHandler,
    create_nanocode_agent,
)
from rlm.runtime import RLMRuntime


def test_rlm_runtime_executes_real_nanocode_agent():

    handler = NanoCodeCallHandler(
        agent_factory=create_nanocode_agent,
    )

    budget = RLMBudget(
        max_depth=1,
        max_children=1,
        max_iterations=1,
    )

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget,
    )

    parent = RLMContext(
        task="Parent RLM task",
        content="",
    )

    result = runtime.call(
        parent=parent,
        task="What is Python?",
        content="",
    )

    assert result.success is True
    assert isinstance(result.answer, str)
    assert result.answer.strip() != ""

    assert result.depth == 1
    assert budget.children_created == 1
    assert budget.iterations == 1