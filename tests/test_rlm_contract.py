from rlm.context import RLMContext
from rlm.repl import RLMREPL


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