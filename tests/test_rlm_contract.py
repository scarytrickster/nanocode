from xml.sax import handler

from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.controller import RLMController
from rlm.result import RLMResult
from rlm.repl import RLMREPL
from rlm.worker import RLMWorker
from rlm.call import RLMCallHandler
from rlm.runtime import RLMRuntime



# ---------------------------------------------------------------------------
# Fake worker used to test the RLM controller without an LLM.
# ---------------------------------------------------------------------------

class FakeWorker(RLMWorker):

    def run(self, context: RLMContext) -> RLMResult:
        return RLMResult(
            answer=f"Processed: {context.task}",
            success=True,
            depth=context.depth,
        )


# ---------------------------------------------------------------------------
# RLM Controller tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RLM REPL tests
# ---------------------------------------------------------------------------

def test_repl_inspects_context():

    context = RLMContext(
        task="Analyze document",
        content="Python is a programming language.",
    )

    repl = RLMREPL(context)

    assert repl.inspect() == "Python is a programming language."


def test_repl_can_slice_context():

    context = RLMContext(
        task="Analyze document",
        content="0123456789",
    )

    repl = RLMREPL(context)

    assert repl.slice(2, 6) == "2345"


def test_repl_can_search_context():

    context = RLMContext(
        task="Find Python references",
        content="Python is great. Python is popular.",
    )

    repl = RLMREPL(context)

    matches = repl.search("Python")

    assert matches == [0, 17]


def test_repl_variables_are_isolated():

    context = RLMContext(
        task="Test variables",
    )

    repl = RLMREPL(context)

    repl.set("answer", "Python")

    assert repl.get("answer") == "Python"
    assert repl.get("missing") is None


def test_repl_variables_returns_copy():

    context = RLMContext(
        task="Test variables",
    )

    repl = RLMREPL(context)

    repl.set("value", 42)

    variables = repl.variables()
    variables["value"] = 100

    assert repl.get("value") == 42


# ---------------------------------------------------------------------------
# Recursive call tests
# ---------------------------------------------------------------------------

class FakeCallHandler(RLMCallHandler):

    def __init__(self):
        self.calls = []

    def call(self, context: RLMContext) -> RLMResult:
        self.calls.append(context)

        return RLMResult(
            answer=f"Child answered: {context.task}",
            success=True,
            depth=context.depth,
        )


def test_repl_can_make_recursive_call():

    context = RLMContext(
        task="Analyze project",
        content="Project source code",
    )

    handler = FakeCallHandler()

    runtime = RLMRuntime(
        call_handler=handler,
    )

    repl = RLMREPL(
        context=context,
        runtime=runtime,
    )

    result = repl.call(
        task="Analyze the agent architecture",
        content="agent source",
    )

    assert result.success is True

    assert result.answer == (
        "Child answered: Analyze the agent architecture"
    )

    assert result.depth == 1

    assert len(handler.calls) == 1

    assert handler.calls[0].task == (
        "Analyze the agent architecture"
    )

    assert handler.calls[0].content == "agent source"


def test_repl_without_runtime_rejects_recursive_call():

    context = RLMContext(
        task="Test",
    )

    repl = RLMREPL(context)

    try:
        repl.call("Child task")
        assert False, "Expected recursive call failure"

    except RuntimeError as exc:
        assert "Recursive calls are not available" in str(exc)


def test_runtime_creates_child_and_tracks_budget():

    handler = FakeCallHandler()

    budget = RLMBudget(
        max_depth=3,
        max_children=2,
        max_iterations=5,
    )

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget,
    )

    parent = RLMContext(
        task="Parent task",
        content="Parent context",
    )

    result = runtime.call(
        parent=parent,
        task="Child task",
        content="Child context",
    )

    assert result.success is True
    assert result.depth == 1

    assert len(handler.calls) == 1
    assert handler.calls[0].task == "Child task"
    assert handler.calls[0].content == "Child context"

    assert budget.children_created == 1
    assert budget.iterations == 1


def test_runtime_respects_max_children():

    handler = FakeCallHandler()

    budget = RLMBudget(
        max_children=1,
        max_iterations=5,
    )

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget,
    )

    parent = RLMContext(
        task="Parent",
    )

    runtime.call(
        parent=parent,
        task="Child 1",
    )

    try:
        runtime.call(
            parent=parent,
            task="Child 2",
        )
        assert False, "Expected child budget failure"

    except RuntimeError as exc:
        assert "recursion budget exceeded" in str(exc)

    assert len(handler.calls) == 1


def test_runtime_respects_max_depth():

    handler = FakeCallHandler()

    budget = RLMBudget(
        max_depth=1,
        max_children=10,
        max_iterations=10,
    )

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget,
    )

    parent = RLMContext(
        task="Parent",
    )

    # Parent depth = 0 → child depth = 1 is all owed.
    result = runtime.call(
        parent=parent,
        task="Child",
    )

    assert result.success is True
    assert result.depth == 1

    # A depth-1 context cannot create another child
    # when max_depth == 1.
    child = parent.child(
        task="Child",
    )

    try:
        runtime.call(
            parent=child,
            task="Grandchild",
        )
        assert False, "Expected depth budget failure"

    except RuntimeError as exc:
        assert "recursion budget exceeded" in str(exc)

def test_repl_call_uses_runtime_budget():

    handler = FakeCallHandler()

    budget = RLMBudget(
        max_depth=1,
        max_children=1,
        max_iterations=10,
    )

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget,
    )

    context = RLMContext(
        task="Parent",
    )

    repl = RLMREPL(
        context=context,
        runtime=runtime,
    )

    first = repl.call(
        task="Child 1",
    )

    assert first.success is True
    assert first.depth == 1

    try:
        repl.call(
            task="Child 2",
        )
        assert False, "Expected child budget failure"

    except RuntimeError as exc:
        assert "recursion budget exceeded" in str(exc)

    assert len(handler.calls) == 1


def test_runtime_can_execute_multiple_children():

    handler = FakeCallHandler()

    budget = RLMBudget(
        max_depth=2,
        max_children=3,
        max_iterations=10,
    )

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget,
    )

    parent = RLMContext(
        task="Analyze project",
    )

    results = runtime.call_many(
        parent=parent,
        tasks=[
            ("Analyze agent", "agent source"),
            ("Analyze tools", "tool source"),
            ("Analyze memory", "memory source"),
        ],
    )

    assert len(results) == 3

    assert all(result.success for result in results)

    assert [result.depth for result in results] == [
        1,
        1,
        1,
    ]

    assert [result.answer for result in results] == [
        "Child answered: Analyze agent",
        "Child answered: Analyze tools",
        "Child answered: Analyze memory",
    ]

    assert budget.children_created == 3
    assert budget.iterations == 3

    assert len(handler.calls) == 3


def test_runtime_call_many_respects_child_budget():

    handler = FakeCallHandler()

    budget = RLMBudget(
        max_depth=2,
        max_children=2,
        max_iterations=10,
    )

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget,
    )

    parent = RLMContext(
        task="Analyze project",
    )

    try:
        runtime.call_many(
            parent=parent,
            tasks=[
                ("Child 1", ""),
                ("Child 2", ""),
                ("Child 3", ""),
            ],
        )

        assert False, "Expected child budget failure"

    except RuntimeError as exc:
        assert "recursion budget exceeded" in str(exc)

    assert len(handler.calls) == 2
    assert budget.children_created == 2