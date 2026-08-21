"""Contract tests for the deterministic RLM router.

None of these tests call an LLM or touch the network.
"""

import pytest

from rlm.router import (
    DEFAULT_THRESHOLD,
    STRATEGY_NORMAL,
    STRATEGY_RLM,
    RLMRouter,
    RouteDecision,
)


@pytest.fixture
def router() -> RLMRouter:
    return RLMRouter()


# ---------------------------------------------------------------------------
# NORMAL routing
# ---------------------------------------------------------------------------

NORMAL_TASKS = [
    "What is Python?",
    "Explain this function.",
    "Fix the typo in auth.py.",
    "Rename this variable.",
    "Add a missing import.",
    "Fix this one-line bug.",
]


@pytest.mark.parametrize("task", NORMAL_TASKS)
def test_simple_tasks_route_to_normal(router, task):

    decision = router.decide(task)

    assert decision.strategy == STRATEGY_NORMAL, (task, decision.reason)
    assert decision.is_rlm is False


# ---------------------------------------------------------------------------
# RLM routing
# ---------------------------------------------------------------------------

RLM_TASKS = [
    "Find the root cause of the authentication bug across the project.",
    "Analyze the entire project and determine why login fails.",
    "Investigate why the application keeps failing.",
    "Compare these implementations.",
    "Analyze multiple files and find the underlying issue.",
]


@pytest.mark.parametrize("task", RLM_TASKS)
def test_complex_tasks_route_to_rlm(router, task):

    decision = router.decide(task)

    assert decision.strategy == STRATEGY_RLM, (task, decision.reason)
    assert decision.is_rlm is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["", "   ", "\n\t  "])
def test_blank_tasks_route_to_normal(router, task):

    decision = router.decide(task)

    assert decision.strategy == STRATEGY_NORMAL
    assert decision.score == 0.0
    assert decision.signals == []


def test_simple_task_containing_find_routes_to_normal(router):

    decision = router.decide("Find the typo in auth.py.")

    assert decision.strategy == STRATEGY_NORMAL, decision.reason


def test_simple_task_containing_analyze_routes_to_normal(router):

    decision = router.decide("Analyze this function.")

    assert decision.strategy == STRATEGY_NORMAL, decision.reason


def test_complex_task_containing_find_routes_to_rlm(router):

    decision = router.decide("Find the root cause across the project.")

    assert decision.strategy == STRATEGY_RLM, decision.reason


def test_complex_task_containing_analyze_routes_to_rlm(router):

    decision = router.decide("Analyze the entire codebase.")

    assert decision.strategy == STRATEGY_RLM, decision.reason


# ---------------------------------------------------------------------------
# Decision metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", NORMAL_TASKS + RLM_TASKS + [""])
def test_decision_exposes_full_contract(router, task):

    decision = router.decide(task)

    assert isinstance(decision, RouteDecision)

    assert decision.strategy in (STRATEGY_NORMAL, STRATEGY_RLM)
    assert isinstance(decision.reason, str)
    assert isinstance(decision.score, float)
    assert isinstance(decision.signals, list)

    assert decision.reason.strip() != ""
    assert 0.0 <= decision.score <= 1.0

    assert decision.is_rlm is (decision.strategy == STRATEGY_RLM)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", NORMAL_TASKS + RLM_TASKS)
def test_repeated_calls_produce_identical_decisions(router, task):

    decisions = [router.decide(task) for _ in range(3)]

    first = decisions[0]

    for other in decisions[1:]:
        assert other.strategy == first.strategy
        assert other.score == first.score
        assert other.reason == first.reason
        assert other.signals == first.signals


def test_separate_router_instances_agree(task="Analyze the entire codebase."):

    assert RLMRouter().decide(task) == RLMRouter().decide(task)


# ---------------------------------------------------------------------------
# Threshold behavior
# ---------------------------------------------------------------------------

def test_default_threshold(router):

    assert router.threshold == DEFAULT_THRESHOLD

    below = router.decide("Analyze this function.")
    above = router.decide("Analyze the entire codebase.")

    assert below.score < DEFAULT_THRESHOLD
    assert below.strategy == STRATEGY_NORMAL

    assert above.score >= DEFAULT_THRESHOLD
    assert above.strategy == STRATEGY_RLM


def test_lower_threshold_promotes_task_to_rlm():

    task = "Analyze this function."

    assert RLMRouter().decide(task).strategy == STRATEGY_NORMAL

    permissive = RLMRouter(threshold=0.2)

    assert permissive.decide(task).strategy == STRATEGY_RLM


def test_higher_threshold_demotes_task_to_normal():

    task = "Analyze the entire codebase."

    assert RLMRouter().decide(task).strategy == STRATEGY_RLM

    strict = RLMRouter(threshold=1.01)

    assert strict.decide(task).strategy == STRATEGY_NORMAL


def test_threshold_does_not_change_score():

    task = "Analyze the entire codebase."

    assert RLMRouter().decide(task).score == RLMRouter(threshold=0.9).decide(task).score


# ---------------------------------------------------------------------------
# Signal behavior
# ---------------------------------------------------------------------------

def test_matching_signals_are_reported(router):

    decision = router.decide("Analyze the entire project.")

    assert "analysis:analyze" in decision.signals
    assert "breadth:entire project" in decision.signals


def test_signals_are_deterministic(router):

    task = "Investigate why the application keeps failing."

    assert router.decide(task).signals == router.decide(task).signals


def test_simple_signals_reduce_the_score(router):

    neutral = router.decide("Find the bug in auth.py.")
    dampened = router.decide("Find the typo in auth.py.")

    assert "simple:typo" in dampened.signals
    assert dampened.score < neutral.score
    assert dampened.strategy == STRATEGY_NORMAL


@pytest.mark.parametrize(
    "task",
    [
        "Find it.",
        "Analyze it.",
        "Find and analyze it.",
        "Review this line.",
    ],
)
def test_generic_verbs_never_force_rlm(router, task):

    decision = router.decide(task)

    assert decision.strategy == STRATEGY_NORMAL, (task, decision.reason)
    assert decision.score < DEFAULT_THRESHOLD
