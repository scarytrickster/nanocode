from agent import agent
from agent.agent import NanoCodeAgent
from models.config import AgentConfig


class FakePlanner:
    def __init__(self):
        self.calls = []

    def run(self, state, experiences=None, retry_context=None):
        self.calls.append(
            {
                "experiences": experiences,
                "retry_context": retry_context,
            }
        )


class FakeExecutor:
    def __init__(self):
        self.calls = 0

    def run(self, state):
        self.calls += 1
        state.final_response = f"response-{self.calls}"


class FakeEvaluation:
    def __init__(self, success):
        self.success = success
        self.score = 1.0 if success else 0.0
        self.reason = (
            "Task completed successfully."
            if success
            else "Task failed."
        )


class FakeEvaluator:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def evaluate(self, state):
        result = self.results[self.calls]
        self.calls += 1
        return FakeEvaluation(result)


class FakeReflection:
    def __init__(self, should_improve=True):
        self.should_improve = should_improve
        self.diagnosis = "The first attempt used the wrong approach."
        self.improvement = "Use a better approach on the next attempt."


class FakeReflector:
    def __init__(self):
        self.calls = 0

    def reflect(self, state, evaluation):
        self.calls += 1
        return FakeReflection()


def create_test_agent(evaluation_results, max_retries=2):
    config = AgentConfig(
        max_retries=max_retries,
    )

    agent = NanoCodeAgent(
        config=config,
    )

    agent.planner = FakePlanner()
    agent.executor = FakeExecutor()
    agent.evaluator = FakeEvaluator(evaluation_results)
    agent.reflector = FakeReflector()

    return agent


def test_retry_after_failure():
    """Agent should retry when the first attempt fails."""

    agent = create_test_agent(
        evaluation_results=[False, True],
        max_retries=2,
    )

    response = agent.run("Fix the failing test")

    assert response == "response-2"

    assert agent.executor.calls == 2
    assert agent.evaluator.calls == 2
    assert agent.reflector.calls == 2

    assert len(agent.planner.calls) == 2

    print("✅ Retry after failure test passed")


def test_no_retry_on_success():
    """Agent should not retry when the first attempt succeeds."""

    agent = create_test_agent(
        evaluation_results=[True],
        max_retries=2,
    )

    response = agent.run("What is Python?")

    assert response == "response-1"

    assert agent.executor.calls == 1
    assert agent.evaluator.calls == 1
    assert agent.reflector.calls == 1
    assert len(agent.planner.calls) == 1

    print("✅ No retry on success test passed")


def test_retry_context_reaches_planner():
    """Reflection information should reach the next planner attempt."""

    agent = create_test_agent(
        evaluation_results=[False, True],
        max_retries=2,
    )

    agent.run("Fix the failing test")

    assert len(agent.planner.calls) == 2

    first_call = agent.planner.calls[0]
    second_call = agent.planner.calls[1]

    assert first_call["retry_context"] is None

    retry_context = second_call["retry_context"]

    assert retry_context is not None
    assert (
        retry_context["diagnosis"]
        == "The first attempt used the wrong approach."
    )
    assert (
        retry_context["improvement"]
        == "Use a better approach on the next attempt."
    )

    print("✅ Retry context reaches planner test passed")


def test_retry_limit():
    """Agent should stop after max_retries is reached."""

    agent = create_test_agent(
        evaluation_results=[False, False, False],
        max_retries=2,
    )

    response = agent.run("Fix the failing test")

    assert response == "response-3"

    # Initial attempt + 2 retries
    assert agent.executor.calls == 3
    assert agent.evaluator.calls == 3
    assert agent.reflector.calls == 3
    assert len(agent.planner.calls) == 3

    print("✅ Retry limit test passed")


def test_retry_count():
    """Agent should track the number of retries."""

    agent = create_test_agent(
        evaluation_results=[False, True],
        max_retries=2,
    )

    agent.run("Fix the failing test")

    # One retry occurred.
    #
    # The state is created inside run(), so this test mainly
    # verifies retry behavior through the call counts above.
    assert agent.executor.calls == 2

    print("✅ Retry count behavior test passed")


if __name__ == "__main__":
    test_retry_after_failure()
    test_no_retry_on_success()
    test_retry_context_reaches_planner()
    test_retry_limit()
    test_retry_count()

    print("\n✅ All agent retry tests passed")