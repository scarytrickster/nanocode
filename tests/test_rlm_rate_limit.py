"""Tests for rate-limit-aware RLM child execution.

Deterministic: fake agents and a fake OpenAI client. No OpenRouter, no HTTP,
no sleeping (the handler's sleep is injected).
"""

import httpx
import openai
import pytest

import config.settings as settings
from rlm.budget import RLMBudget
from rlm.context import RLMContext
from rlm.decomposer import DeterministicRLMDecomposer
from rlm.nanocode_handler import (
    ERROR_TYPE_ERROR,
    ERROR_TYPE_RATE_LIMIT,
    NanoCodeCallHandler,
    classify_error,
    is_rate_limit_error,
)
from rlm.orchestrator import RLMOrchestrator
from rlm.result import RLMResult
from rlm.runtime import RLMRuntime
from rlm.synthesizer import (
    FAILED_KEY,
    PARTIAL_KEY,
    RATE_LIMITED_KEY,
    RLMSynthesizer,
    SUCCESSFUL_KEY,
)


COMPLEX_TASK = "Find the authentication bug across the project."


def rate_limit_error(message: str = "temporarily rate-limited upstream"):
    """A real openai.RateLimitError, built without touching the network."""

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    response = httpx.Response(
        429,
        request=request,
        json={"error": {"message": "Provider returned error", "code": 429}},
    )

    return openai.RateLimitError(
        f"Error code: 429 - {message}",
        response=response,
        body=None,
    )


class RateLimitedAgent:
    """A child agent that raises 429 for its first `failures` attempts."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts = 0

    def run(self, task: str) -> str:
        self.attempts += 1

        if self.attempts <= self.failures:
            raise rate_limit_error()

        return f"answer for {task}"


class AlwaysRateLimitedAgent:
    def __init__(self) -> None:
        self.attempts = 0

    def run(self, task: str) -> str:
        self.attempts += 1
        raise rate_limit_error()


class SleepSpy:
    """Records the delays the handler asked for, without waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def build_handler(agent, retries=1, delay=5.0):
    sleep = SleepSpy()

    handler = NanoCodeCallHandler(
        agent_factory=lambda: agent,
        rate_limit_retries=retries,
        rate_limit_delay=delay,
        sleep=sleep,
    )

    return handler, sleep


def build_pipeline(agent_factory, retries=0, budget=None):
    """A full RLM pipeline with fake children and no real sleeping."""

    sleep = SleepSpy()

    handler = NanoCodeCallHandler(
        agent_factory=agent_factory,
        rate_limit_retries=retries,
        rate_limit_delay=0.0,
        sleep=sleep,
    )

    runtime = RLMRuntime(
        call_handler=handler,
        budget=budget or RLMBudget(max_depth=2, max_children=5, max_iterations=10),
    )

    return RLMOrchestrator(runtime=runtime), handler, sleep


# ---------------------------------------------------------------------------
# 1-2. Classification
# ---------------------------------------------------------------------------

def test_rate_limit_error_is_classified_as_a_rate_limit():

    error = rate_limit_error()

    assert is_rate_limit_error(error) is True
    assert classify_error(error) == ERROR_TYPE_RATE_LIMIT


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("child exploded"),
        ValueError("bad argument"),
        KeyError("missing"),
        TimeoutError("slow"),
        openai.APIConnectionError(request=httpx.Request("POST", "http://x")),
    ],
)
def test_other_errors_are_not_classified_as_rate_limits(error):

    assert is_rate_limit_error(error) is False
    assert classify_error(error) == ERROR_TYPE_ERROR


def test_a_non_rate_limit_failure_is_not_marked_rate_limited():

    class ExplodingAgent:
        def run(self, task: str) -> str:
            raise RuntimeError("child exploded")

    handler, sleep = build_handler(ExplodingAgent())

    result = handler.call(RLMContext(task=COMPLEX_TASK, depth=1))

    assert result.success is False
    assert result.metadata["error_type"] == ERROR_TYPE_ERROR
    assert result.metadata["rate_limited"] is False

    # Deterministic failures are never retried: they would fail identically.
    assert result.metadata["attempts"] == 1
    assert sleep.delays == []


# ---------------------------------------------------------------------------
# 3. SDK retry configuration
# ---------------------------------------------------------------------------

def test_sdk_does_not_silently_multiply_requests():

    assert settings.SDK_MAX_RETRIES == 0
    assert settings.client.max_retries == 0


def test_one_application_request_is_one_http_attempt():

    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)

        return httpx.Response(
            429,
            json={"error": {"message": "Provider returned error", "code": 429}},
        )

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        max_retries=settings.SDK_MAX_RETRIES,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

    # One request in, one request out -- no hidden 3x amplification.
    assert len(attempts) == 1


# ---------------------------------------------------------------------------
# 4-6. Bounded application retry
# ---------------------------------------------------------------------------

def test_rate_limit_retry_is_bounded():

    agent = AlwaysRateLimitedAgent()

    handler, sleep = build_handler(agent, retries=1)

    result = handler.call(RLMContext(task=COMPLEX_TASK, depth=1))

    # One retry means two attempts total, and exactly one wait.
    assert agent.attempts == 2
    assert result.metadata["attempts"] == 2
    assert len(sleep.delays) == 1

    assert result.success is False


def test_retry_count_is_configurable_and_never_unbounded():

    for retries in (0, 1, 3):
        agent = AlwaysRateLimitedAgent()

        handler, sleep = build_handler(agent, retries=retries)

        handler.call(RLMContext(task=COMPLEX_TASK, depth=1))

        assert agent.attempts == retries + 1
        assert len(sleep.delays) == retries


def test_a_negative_retry_count_is_clamped():

    agent = AlwaysRateLimitedAgent()

    handler, _ = build_handler(agent, retries=-5)

    handler.call(RLMContext(task=COMPLEX_TASK, depth=1))

    assert agent.attempts == 1


def test_retry_succeeds_when_the_rate_limit_clears():

    agent = RateLimitedAgent(failures=1)

    handler, sleep = build_handler(agent, retries=1, delay=5.0)

    result = handler.call(RLMContext(task=COMPLEX_TASK, depth=1))

    assert result.success is True
    assert result.answer == f"answer for {COMPLEX_TASK}"
    assert result.metadata["attempts"] == 2

    # It waited before retrying, with the configured delay.
    assert sleep.delays == [5.0]


def test_retry_fails_when_the_rate_limit_persists():

    agent = AlwaysRateLimitedAgent()

    handler, _ = build_handler(agent, retries=2)

    result = handler.call(RLMContext(task=COMPLEX_TASK, depth=1))

    assert agent.attempts == 3

    assert result.success is False
    assert result.answer == ""
    assert result.metadata["error_type"] == ERROR_TYPE_RATE_LIMIT
    assert result.metadata["rate_limited"] is True
    assert "429" in result.metadata["error"]
    assert result.metadata["task"] == COMPLEX_TASK

    # The child result still uses the existing RLMResult shape.
    assert isinstance(result, RLMResult)
    assert result.depth == 1


# ---------------------------------------------------------------------------
# 7. Child isolation is preserved
# ---------------------------------------------------------------------------

def test_a_rate_limited_child_does_not_prevent_later_children():

    executed: list[str] = []

    class MiddleChildRateLimited:
        """Child 2 is rate limited; children 1 and 3 succeed."""

        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: str) -> str:
            self.calls += 1
            executed.append(task)

            if self.calls == 2:
                raise rate_limit_error()

            return f"answer for {task}"

    agent = MiddleChildRateLimited()

    orchestrator, handler, _ = build_pipeline(lambda: agent)

    answer = orchestrator.run(COMPLEX_TASK)

    expected = [child.task for child in orchestrator.last_child_tasks]

    assert len(expected) == 3

    # Children 1, 2 and 3 all ran: the rate-limited middle child did not
    # abort its siblings.
    assert executed == expected

    metadata = orchestrator.last_result.metadata

    assert metadata[SUCCESSFUL_KEY] == 2
    assert metadata[RATE_LIMITED_KEY] == 1

    assert orchestrator.last_result.success is True
    assert answer


# ---------------------------------------------------------------------------
# 8. One success + two rate limits is reported as partial
# ---------------------------------------------------------------------------

def test_partial_results_are_reported_as_partial():

    class FirstChildOnly:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: str) -> str:
            self.calls += 1

            if self.calls == 1:
                return "auth.py uses milliseconds."

            raise rate_limit_error()

    # One shared agent, so "first call" means the first child, not the first
    # call of a freshly built agent.
    agent = FirstChildOnly()

    orchestrator, _, _ = build_pipeline(lambda: agent)

    answer = orchestrator.run(COMPLEX_TASK)

    metadata = orchestrator.last_result.metadata

    assert metadata[PARTIAL_KEY] is True
    assert metadata[SUCCESSFUL_KEY] == 1
    assert metadata[FAILED_KEY] == 2
    assert metadata[RATE_LIMITED_KEY] == 2

    # The user-facing answer cannot read as a complete investigation.
    assert "Partial investigation" in answer
    assert "1 of 3" in answer
    assert "rate-limited" in answer

    # The successful analysis is still delivered.
    assert "auth.py uses milliseconds." in answer


def test_the_stored_result_answer_is_not_rewritten():

    class FirstChildOnly:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: str) -> str:
            self.calls += 1

            if self.calls == 1:
                return "child one answer"

            raise rate_limit_error()

    agent = FirstChildOnly()

    orchestrator, _, _ = build_pipeline(lambda: agent)

    orchestrator.run(COMPLEX_TASK)

    # The synthesizer's answer stays exactly what the children produced; the
    # notice lives on the user-facing string only.
    assert orchestrator.last_result.answer == "child one answer"


def test_a_complete_run_carries_no_partial_notice():

    class HealthyAgent:
        def run(self, task: str) -> str:
            return f"answer for {task}"

    orchestrator, _, _ = build_pipeline(lambda: HealthyAgent())

    answer = orchestrator.run(COMPLEX_TASK)

    metadata = orchestrator.last_result.metadata

    assert metadata[PARTIAL_KEY] is False
    assert metadata[FAILED_KEY] == 0
    assert metadata[RATE_LIMITED_KEY] == 0

    assert "Partial investigation" not in answer
    assert answer == orchestrator.last_result.answer


# ---------------------------------------------------------------------------
# 9. All children rate limited is a failure, not an empty success
# ---------------------------------------------------------------------------

def test_all_children_rate_limited_produces_a_failure():

    orchestrator, _, _ = build_pipeline(lambda: AlwaysRateLimitedAgent())

    answer = orchestrator.run(COMPLEX_TASK)

    result = orchestrator.last_result

    assert result.success is False
    assert result.answer == ""
    assert result.children_created == 3

    metadata = result.metadata

    assert metadata[SUCCESSFUL_KEY] == 0
    assert metadata[FAILED_KEY] == 3
    assert metadata[RATE_LIMITED_KEY] == 3

    # The answer explains why, instead of looking like an empty success.
    assert COMPLEX_TASK in answer
    assert "rate-limited" in answer


# ---------------------------------------------------------------------------
# 10-11. Failure information survives into synthesis and the final result
# ---------------------------------------------------------------------------

def test_successful_child_results_remain_available_to_synthesis():

    synthesizer = RLMSynthesizer()

    results = [
        RLMResult(answer="auth.py uses milliseconds.", success=True, depth=1),
        RLMResult(
            answer="",
            success=False,
            depth=1,
            metadata={
                "error": "Error code: 429",
                "error_type": ERROR_TYPE_RATE_LIMIT,
                "rate_limited": True,
                "task": "child two",
            },
        ),
        RLMResult(
            answer="validation compares against seconds.",
            success=True,
            depth=1,
        ),
    ]

    result = synthesizer.synthesize(results)

    assert result.success is True

    # Existing behaviour: successful answers are joined, failures are not
    # concatenated into the answer.
    assert result.answer == (
        "auth.py uses milliseconds.\n\n"
        "validation compares against seconds."
    )


def test_rate_limit_metadata_reaches_the_final_result():

    synthesizer = RLMSynthesizer()

    results = [
        RLMResult(answer="ok", success=True, depth=1),
        RLMResult(
            answer="",
            success=False,
            depth=1,
            metadata={
                "error": "Error code: 429 - rate-limited upstream",
                "error_type": ERROR_TYPE_RATE_LIMIT,
                "rate_limited": True,
                "task": "child two task",
            },
        ),
    ]

    metadata = synthesizer.synthesize(results).metadata

    assert metadata[PARTIAL_KEY] is True
    assert metadata[RATE_LIMITED_KEY] == 1

    failures = metadata["failures"]

    assert len(failures) == 1
    assert failures[0]["error_type"] == ERROR_TYPE_RATE_LIMIT
    assert failures[0]["rate_limited"] is True
    assert failures[0]["task"] == "child two task"
    assert "429" in failures[0]["error"]


def test_failed_children_are_distinguished_by_reason():

    synthesizer = RLMSynthesizer()

    results = [
        RLMResult(answer="ok", success=True, depth=1),
        RLMResult(
            answer="",
            success=False,
            depth=1,
            metadata={
                "error": "429",
                "error_type": ERROR_TYPE_RATE_LIMIT,
                "rate_limited": True,
            },
        ),
        RLMResult(
            answer="",
            success=False,
            depth=1,
            metadata={
                "error": "child exploded",
                "error_type": ERROR_TYPE_ERROR,
                "rate_limited": False,
            },
        ),
    ]

    metadata = synthesizer.synthesize(results).metadata

    assert metadata[FAILED_KEY] == 2

    # Only the real rate limit is counted as one.
    assert metadata[RATE_LIMITED_KEY] == 1


# ---------------------------------------------------------------------------
# 12-14. Existing budgets, depth and iteration limits are untouched
# ---------------------------------------------------------------------------

def test_child_budget_is_still_enforced():

    orchestrator, handler, _ = build_pipeline(
        lambda: AlwaysRateLimitedAgent(),
        budget=RLMBudget(max_depth=2, max_children=1, max_iterations=10),
    )

    with pytest.raises(RuntimeError, match="budget"):
        orchestrator.run(COMPLEX_TASK)

    assert orchestrator.runtime.budget.children_created == 1


def test_retries_do_not_consume_extra_child_budget():

    agent = AlwaysRateLimitedAgent()

    orchestrator, _, _ = build_pipeline(lambda: agent, retries=2)

    orchestrator.run(COMPLEX_TASK)

    children = len(orchestrator.last_child_tasks)

    # 3 children x 3 attempts each = 9 agent runs, but still only 3 children.
    assert agent.attempts == children * 3
    assert orchestrator.runtime.budget.children_created == children
    assert orchestrator.runtime.budget.iterations == children


def test_max_depth_is_still_enforced():

    orchestrator, handler, _ = build_pipeline(
        lambda: AlwaysRateLimitedAgent(),
        budget=RLMBudget(max_depth=2, max_children=5, max_iterations=10),
    )

    parent = RLMContext(task=COMPLEX_TASK, depth=2)

    children = DeterministicRLMDecomposer().decompose(parent.task, parent)

    with pytest.raises(RuntimeError, match="budget"):
        orchestrator.runtime.call_and_synthesize(
            parent=parent,
            tasks=children,
            synthesizer=orchestrator.synthesizer,
        )

    assert orchestrator.runtime.budget.children_created == 0


def test_iteration_budget_is_still_enforced():

    orchestrator, _, _ = build_pipeline(
        lambda: AlwaysRateLimitedAgent(),
        budget=RLMBudget(max_depth=2, max_children=5, max_iterations=2),
    )

    with pytest.raises(RuntimeError, match="budget"):
        orchestrator.run(COMPLEX_TASK)

    assert orchestrator.runtime.budget.iterations == 2


# ---------------------------------------------------------------------------
# 15. Normal (non-RLM) execution is unchanged
# ---------------------------------------------------------------------------

def test_normal_non_rlm_execution_is_unchanged():

    from agent.agent import NanoCodeAgent
    from agent.state import AgentStatus
    from rlm.router import STRATEGY_NORMAL, RLMRouter

    calls: list[str] = []

    class RecordingStage:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, state, **kwargs):
            calls.append(self.name)

            if self.name == "executor":
                state.status = AgentStatus.COMPLETED
                state.final_response = "normal answer"

        def evaluate(self, state):
            calls.append("evaluator")
            return type("Evaluation", (), {"success": True, "reason": "ok"})()

        def reflect(self, state, evaluation):
            calls.append("reflector")
            return type(
                "Reflection",
                (),
                {"should_improve": False, "diagnosis": "", "improvement": ""},
            )()

    orchestrator, handler, _ = build_pipeline(lambda: AlwaysRateLimitedAgent())

    agent = NanoCodeAgent(
        console_trace=False,
        router=RLMRouter(),
        rlm_orchestrator=orchestrator,
    )

    agent.planner = RecordingStage("planner")
    agent.executor = RecordingStage("executor")
    agent.evaluator = RecordingStage("evaluator")
    agent.reflector = RecordingStage("reflector")

    answer = agent.run("What is Python?")

    assert agent.last_route_decision.strategy == STRATEGY_NORMAL
    assert answer == "normal answer"
    assert calls == ["planner", "executor", "evaluator", "reflector"]

    # The RLM stack, rate-limit handling included, stayed out of the way.
    assert orchestrator.last_result is None
    assert orchestrator.last_child_tasks == []
