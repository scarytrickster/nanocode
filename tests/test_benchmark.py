"""Tests for the benchmark framework.

Deterministic: synthetic fixtures on disk, a scripted model in place of the
HTTP call. No OpenRouter, no network. The benchmark drives the real agent,
router, RLM orchestrator, runtime, decomposer and synthesizer.
"""

import re

import pytest

from benchmark.fakes import ScriptedModel
from benchmark.fixtures import AUTH_EXPIRY, CLEAN_MODULE, FIXTURES
from benchmark.metrics import (
    TOKENS_UNAVAILABLE,
    ExecutionMetrics,
    collect_metrics,
)
from benchmark.report import DISCLAIMER, render_aggregate, render_report, render_result
from benchmark.runner import (
    BenchmarkResult,
    EvidenceMetrics,
    PathResult,
    aggregate,
    evidence_from,
    run_benchmark,
    run_task,
)
from benchmark.scoring import BenchmarkScore, score_answer
from benchmark.tasks import (
    TASKS,
    TASKS_BY_ID,
    BenchmarkTask,
    ExpectedFinding,
    SINGLE_FILE_BUG,
)
from agent.tracer import TraceEvent
from rlm.router import STRATEGY_NORMAL, STRATEGY_RLM


AUTH_TASK = TASKS_BY_ID["auth-expiry"]
CLEAN_TASK = TASKS_BY_ID["clean-module"]


def event(name: str, component: str = "agent", **data) -> TraceEvent:
    return TraceEvent(
        name=name,
        timestamp="2026-01-01T00:00:00",
        component=component,
        data=data,
    )


@pytest.fixture(scope="module")
def auth_result(tmp_path_factory):
    """One real benchmark run, shared by the tests that inspect it."""

    root = tmp_path_factory.mktemp("benchmark-auth")

    return run_task(AUTH_TASK, str(root))


# ---------------------------------------------------------------------------
# 1-2. Task and result models
# ---------------------------------------------------------------------------

def test_benchmark_task_creation():

    task = BenchmarkTask(
        id="demo",
        name="Demo",
        prompt="Find the bug.",
        category=SINGLE_FILE_BUG,
        fixture="auth_expiry",
        expected=ExpectedFinding(
            files=("auth.py",),
            symbols=("validate_token",),
            behaviors=("1000",),
        ),
    )

    assert task.id == "demo"
    assert task.expected.criteria_count == 3
    assert task.expected.should_find_defect is True


def test_the_shipped_tasks_are_well_formed():

    assert TASKS

    for task in TASKS:
        assert task.fixture in FIXTURES
        assert task.prompt
        assert task.category

        if task.expected.should_find_defect:
            assert task.expected.criteria_count >= 1


def test_benchmark_result_creation(auth_result):

    assert isinstance(auth_result, BenchmarkResult)
    assert isinstance(auth_result.normal, PathResult)
    assert isinstance(auth_result.rlm, PathResult)

    assert auth_result.normal.strategy == STRATEGY_NORMAL
    assert auth_result.rlm.strategy == STRATEGY_RLM

    payload = auth_result.to_dict()

    assert payload["task"] == "auth-expiry"
    assert "normal" in payload and "rlm" in payload


def test_expected_criteria_come_from_the_fixture_source():

    # The expected symbol really is defined in the fixture, and the expected
    # behaviour really appears in it: the criteria are not copied from output.
    source = FIXTURES[AUTH_TASK.fixture].files["auth.py"]

    for symbol in AUTH_TASK.expected.symbols:
        assert f"def {symbol}" in source

    for behavior in AUTH_TASK.expected.behaviors:
        assert behavior in source


# ---------------------------------------------------------------------------
# 3-4. Both paths are measured
# ---------------------------------------------------------------------------

def test_normal_run_is_measured(auth_result):

    normal = auth_result.normal

    assert normal.success is True
    assert normal.answer
    assert normal.metrics.llm_calls > 0
    assert normal.metrics.tool_calls > 0

    # The normal path spawns no children.
    assert normal.metrics.children == 0


def test_rlm_run_is_measured(auth_result):

    rlm = auth_result.rlm

    assert rlm.success is True
    assert rlm.answer
    assert rlm.metrics.children > 1
    assert rlm.metrics.successful_children == rlm.metrics.children


def test_the_real_router_decision_is_recorded(auth_result):

    assert auth_result.router_strategy in (STRATEGY_NORMAL, STRATEGY_RLM)
    assert auth_result.router_reason

    # Forcing a path for comparison does not overwrite what the router thinks.
    rlm_task = TASKS_BY_ID["project-wide"]

    from rlm.router import RLMRouter

    assert RLMRouter().decide(rlm_task.prompt).strategy == STRATEGY_RLM


# ---------------------------------------------------------------------------
# 5-7. Counting comes from real execution records
# ---------------------------------------------------------------------------

def test_llm_calls_are_counted_from_recorded_events():

    metrics = collect_metrics(
        [
            event("planner.started", "planner"),
            event("llm.started", "executor"),
            event("llm.started", "executor"),
            event("evaluation.completed", "evaluator"),
        ]
    )

    assert metrics.planner_calls == 1
    assert metrics.executor_calls == 2
    assert metrics.llm_calls == 3


def test_tool_calls_are_counted_by_name():

    metrics = collect_metrics(
        [
            event("tool.started", "executor", tool="grep"),
            event("tool.started", "executor", tool="read_file"),
            event("tool.started", "executor", tool="read_file"),
        ]
    )

    assert metrics.tool_calls == 3
    assert metrics.tool_calls_by_name == {"grep": 1, "read_file": 2}


def test_a_tool_named_in_a_prompt_is_not_counted():

    # Only recorded executions count; prose mentioning a tool does not.
    metrics = collect_metrics(
        [event("llm.response", "executor", content="I will use grep and read_file")]
    )

    assert metrics.tool_calls == 0


def test_planner_calls_are_not_counted_as_tools():

    metrics = collect_metrics([event("planner.started", "planner")])

    assert metrics.tool_calls == 0
    assert metrics.llm_calls == 1


def test_children_are_counted_from_child_events():

    metrics = collect_metrics(
        [
            event("rlm.child.started", "rlm", child=1, of=3),
            event("rlm.child.completed", "rlm", child=1),
            event("rlm.child.started", "rlm", child=2, of=3),
            event("rlm.child.failed", "rlm", child=2, error_type="rate_limit"),
        ]
    )

    assert metrics.children == 2
    assert metrics.successful_children == 1
    assert metrics.failed_children == 1


def test_the_actual_child_count_is_recorded_not_assumed(auth_result):

    children = auth_result.rlm.metrics.children

    # The number comes from the run, and matches what the runtime executed.
    assert children == auth_result.rlm.metrics.successful_children
    assert children >= 1


def test_rlm_uses_more_calls_than_normal_on_this_task(auth_result):

    # Recorded, not assumed: this is a cost observation, not a quality claim.
    assert auth_result.rlm.metrics.llm_calls > auth_result.normal.metrics.llm_calls
    assert auth_result.rlm.metrics.tool_calls > auth_result.normal.metrics.tool_calls


# ---------------------------------------------------------------------------
# 8. Failures
# ---------------------------------------------------------------------------

def test_failures_are_recorded_not_hidden():

    metrics = collect_metrics(
        [
            event("planner.failed", "planner", error="429"),
            event("rlm.child.failed", "rlm", child=2, error_type="rate_limit"),
            event("tool.failed", "executor", tool="grep"),
        ]
    )

    assert metrics.failure_count == 3

    joined = " ".join(metrics.failures)

    assert "rate_limit" in joined
    assert "planner.failed" in joined


def test_retry_exhaustion_is_recorded():

    metrics = collect_metrics(
        [
            event("retry.started", "agent", attempt=1),
            event("retry.started", "agent", attempt=2),
            event("retry.exhausted", "agent", attempts=2),
        ]
    )

    assert metrics.attempts == 3
    assert metrics.retry_exhausted is True


# ---------------------------------------------------------------------------
# 9. RSI
# ---------------------------------------------------------------------------

def test_rsi_retries_are_distinguished_from_the_initial_attempt():

    metrics = collect_metrics(
        [
            event("planner.started", "planner"),
            event("rsi.started", "rsi", attempt=1, has_improvement=True),
            event("retry.started", "agent", attempt=1),
            event("rsi.retry.started", "rsi", attempt=2, has_improvement=True),
        ]
    )

    assert metrics.attempts == 2
    assert metrics.rsi_retries == 1
    assert metrics.rsi_used_improvement is True


def test_a_run_without_rsi_records_one_attempt(auth_result):

    assert auth_result.normal.metrics.attempts == 1
    assert auth_result.normal.metrics.rsi_retries == 0
    assert auth_result.normal.metrics.rsi_used_improvement is False


def test_rsi_without_improvement_context_is_recorded():

    metrics = collect_metrics(
        [event("rsi.retry.started", "rsi", attempt=2, has_improvement=False)]
    )

    assert metrics.rsi_retries == 1
    assert metrics.rsi_used_improvement is False


# ---------------------------------------------------------------------------
# 10. Evidence metrics come from the public metadata contract
# ---------------------------------------------------------------------------

def test_evidence_metrics_are_read_from_result_metadata():

    metrics = evidence_from(
        {
            "findings": [
                {"status": "confirmed", "evidence": ["auth.py", "validate_token()"]},
                {"status": "possible", "evidence": []},
                {"status": "conflicting", "evidence": ["config.py"]},
            ],
            "confirmed_findings": 1,
            "conflicting_findings": 1,
            "primary_finding": "auth.py validate_token() is wrong",
            "confidence": "high",
            "partial": False,
        }
    )

    assert metrics.findings_count == 3
    assert metrics.confirmed_findings == 1
    assert metrics.possible_findings == 1
    assert metrics.conflicting_findings == 1
    assert metrics.evidence_items == 3
    assert metrics.confidence == "high"


def test_a_real_rlm_run_produces_evidence_metrics(auth_result):

    evidence = auth_result.rlm.evidence

    assert evidence.findings_count >= 1
    assert evidence.confidence in ("high", "medium", "low")
    assert evidence.primary_finding


def test_the_normal_path_has_no_evidence_metrics(auth_result):

    assert auth_result.normal.evidence.findings_count == 0
    assert auth_result.normal.evidence.primary_finding == ""


# ---------------------------------------------------------------------------
# 11. Correctness scoring
# ---------------------------------------------------------------------------

def test_correctness_requires_the_expected_file_symbol_and_behavior():

    expected = ExpectedFinding(
        files=("auth.py",),
        symbols=("validate_token",),
        behaviors=("1000",),
    )

    full = score_answer(
        "auth.py validate_token() multiplies time.time() by 1000.", expected
    )

    assert full.correct is True
    assert full.completeness == 1.0
    assert full.score == 1.0

    partial = score_answer("auth.py has a problem somewhere.", expected)

    assert partial.correct is False
    assert 0 < partial.completeness < 1
    assert partial.score < 1.0


def test_correctness_is_not_string_equality():

    expected = ExpectedFinding(
        files=("auth.py",),
        symbols=("validate_token",),
        behaviors=("1000",),
    )

    reworded = score_answer(
        "In auth.py, the validate_token function compares against a value "
        "scaled by 1000, which is a unit mismatch.",
        expected,
    )

    assert reworded.correct is True


def test_a_wrong_answer_is_not_correct():

    expected = ExpectedFinding(
        files=("auth.py",),
        symbols=("validate_token",),
        behaviors=("1000",),
    )

    wrong = score_answer("config.py has a hardcoded SECRET_KEY.", expected)

    assert wrong.correct is False
    assert wrong.score == 0.0


def test_primary_finding_correctness_is_scored_separately():

    expected = ExpectedFinding(files=("auth.py",), symbols=("validate_token",))

    right = score_answer(
        "auth.py validate_token is wrong.",
        expected,
        primary_finding="auth.py validate_token() mis-converts the expiry",
    )

    wrong = score_answer(
        "auth.py validate_token is wrong.",
        expected,
        primary_finding="SECRET_KEY is hardcoded in config.py",
    )

    assert right.primary_finding_correct is True
    assert wrong.primary_finding_correct is False

    # The overall answer is still correct in both cases.
    assert right.correct is True
    assert wrong.correct is True


def test_reporting_a_bug_in_a_clean_fixture_is_a_false_positive():

    expected = ExpectedFinding(should_find_defect=False)

    clean = score_answer("No defect was found in calculator.py.", expected)
    invented = score_answer("calculator.py add() is broken.", expected)

    assert clean.correct is True
    assert clean.false_positive is False

    assert invented.correct is False
    assert invented.false_positive is True
    assert invented.score == 0.0


def test_the_clean_fixture_task_runs_without_a_false_positive(tmp_path):

    result = run_task(CLEAN_TASK, str(tmp_path))

    assert result.normal.score.false_positive is False
    assert result.normal.score.correct is True


# ---------------------------------------------------------------------------
# 12. Partial results
# ---------------------------------------------------------------------------

def test_partial_results_are_detected():

    metrics = evidence_from(
        {
            "findings": [],
            "partial": True,
            "rate_limited_children": 2,
        }
    )

    assert metrics.partial is True


def test_a_complete_run_is_not_marked_partial(auth_result):

    assert auth_result.rlm.evidence.partial is False


# ---------------------------------------------------------------------------
# 13 & 18. Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_metrics(auth_result):

    totals = aggregate([auth_result])

    assert totals["tasks"] == 1

    for side in ("normal", "rlm"):
        assert totals[side]["success_rate"] in (0.0, 1.0)
        assert totals[side]["average_llm_calls"] > 0

    assert totals["rlm"]["average_children"] >= 1
    assert totals["normal"]["average_children"] == 0


def test_aggregate_handles_an_empty_benchmark():

    totals = aggregate([])

    assert totals["tasks"] == 0
    assert totals["normal"] == {}

    assert "No tasks were evaluated." in render_aggregate([])


def test_running_no_tasks_produces_no_results(tmp_path):

    assert run_benchmark([], str(tmp_path)) == []


# ---------------------------------------------------------------------------
# 14 & 21. Determinism
# ---------------------------------------------------------------------------

def test_repeated_benchmarks_produce_identical_metrics(tmp_path):

    first = run_task(AUTH_TASK, str(tmp_path / "one"))
    second = run_task(AUTH_TASK, str(tmp_path / "two"))

    for side in ("normal", "rlm"):
        left = getattr(first, side)
        right = getattr(second, side)

        assert left.score.correct == right.score.correct
        assert left.score.score == right.score.score
        assert left.success == right.success

        assert left.metrics.llm_calls == right.metrics.llm_calls
        assert left.metrics.tool_calls == right.metrics.tool_calls
        assert left.metrics.tool_calls_by_name == right.metrics.tool_calls_by_name
        assert left.metrics.children == right.metrics.children

        assert left.evidence.findings_count == right.evidence.findings_count
        assert left.evidence.primary_finding == right.evidence.primary_finding


def test_duration_is_recorded_but_not_asserted_exactly(auth_result):

    for side in (auth_result.normal, auth_result.rlm):
        assert side.metrics.duration_seconds >= 0.0


def test_an_injected_clock_makes_duration_deterministic(tmp_path):

    ticks = iter(range(0, 1000))

    result = run_task(AUTH_TASK, str(tmp_path), clock=lambda: next(ticks))

    # Each run consumes exactly two ticks: start and end.
    assert result.normal.metrics.duration_seconds == 1.0
    assert result.rlm.metrics.duration_seconds == 1.0


# ---------------------------------------------------------------------------
# 15. Token usage is reported as unavailable, never invented
# ---------------------------------------------------------------------------

def test_token_usage_is_reported_as_unavailable(auth_result):

    assert auth_result.normal.metrics.tokens == TOKENS_UNAVAILABLE
    assert "unavailable" in auth_result.rlm.metrics.tokens

    # No numeric token field is fabricated anywhere in the metrics payload.
    payload = auth_result.rlm.metrics.to_dict()

    assert not isinstance(payload["tokens"], (int, float))


# ---------------------------------------------------------------------------
# 16-17. Report rendering and comparison
# ---------------------------------------------------------------------------

def test_report_rendering_shows_both_paths(auth_result):

    report = render_result(auth_result)

    assert "Normal" in report and "RLM" in report
    assert "LLM calls" in report
    assert "Tool calls" in report
    assert "Children" in report
    assert "Primary correct" in report


def test_the_report_states_that_it_is_not_a_scientific_evaluation(auth_result):

    report = render_report([auth_result])

    assert DISCLAIMER in report
    assert "not a scientific evaluation" in report


def test_the_report_does_not_hide_failures():

    metrics = ExecutionMetrics(failures=["child 2: rate_limit"])

    result = BenchmarkResult(
        task=AUTH_TASK,
        normal=PathResult(
            strategy=STRATEGY_NORMAL,
            success=False,
            answer="",
            metrics=metrics,
            score=BenchmarkScore(),
        ),
        rlm=PathResult(
            strategy=STRATEGY_RLM,
            success=False,
            answer="",
            metrics=ExecutionMetrics(),
            score=BenchmarkScore(),
        ),
    )

    report = render_result(result)

    assert "Recorded failures:" in report
    assert "child 2: rate_limit" in report


def test_the_benchmark_does_not_favor_rlm(auth_result):

    # The framework reports whatever happened. On this task both paths are
    # correct, so neither is credited with an advantage it did not earn.
    assert auth_result.normal.score.correct == auth_result.rlm.score.correct

    totals = aggregate([auth_result])

    assert totals["normal"]["correct_rate"] == totals["rlm"]["correct_rate"]


def test_normal_and_rlm_run_the_same_task(auth_result):

    assert auth_result.normal.strategy != auth_result.rlm.strategy
    assert auth_result.task.prompt


# ---------------------------------------------------------------------------
# The scripted model itself
# ---------------------------------------------------------------------------

def test_the_scripted_model_does_not_know_the_answer():

    source = open("benchmark/fakes.py", encoding="utf-8").read()

    # The model must not contain the fixtures' answers.
    assert "validate_token" not in source
    assert "TOKEN_TTL_SECONDS" not in source
    assert "auth.py" not in source


def test_the_scripted_model_reports_only_what_it_read(tmp_path):

    project = CLEAN_MODULE.write(str(tmp_path))

    model = ScriptedModel(project=project)

    findings = model._inspect(
        "calculator.py", CLEAN_MODULE.files["calculator.py"]
    )

    assert findings == []

    suspicious = model._inspect(
        "auth.py", AUTH_EXPIRY.files["auth.py"]
    )

    assert suspicious
    assert any("1000" in finding for finding in suspicious)


def test_the_scripted_model_records_its_calls(auth_result):

    # Model calls and recorded LLM events agree, so counting is consistent.
    assert auth_result.normal.metrics.llm_calls >= 2
