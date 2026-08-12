from agent.agent import NanoCodeAgent
from agent.memory import Experience
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


class FakeEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, state):
        self.calls += 1

        # First attempt fails, retry succeeds.
        return FakeEvaluation(
            success=self.calls > 1
        )


class FakeReflection:
    should_improve = True
    diagnosis = "The first attempt used the wrong approach."
    improvement = "Use the previous experience before trying again."


class FakeReflector:
    def __init__(self):
        self.calls = 0

    def reflect(self, state, evaluation):
        self.calls += 1
        return FakeReflection()


def test_memory_and_retry_context_reach_planner():
    """Verify memory and retry context are both available to the retry planner."""

    agent = NanoCodeAgent(
        config=AgentConfig(max_retries=1),
    )

    agent.planner = FakePlanner()
    agent.executor = FakeExecutor()
    agent.evaluator = FakeEvaluator()
    agent.reflector = FakeReflector()

    # Seed Phase 6 memory.
    seed_experience = Experience(
        task="Fix a Python test",
        diagnosis="The test failed because the wrong file was used.",
        improvement="Inspect the project structure before running the test.",
        success=False,
    )

    agent.memory.add(seed_experience)

    response = agent.run("Fix another failing Python test")

    assert response == "response-2"

    # Two planner calls:
    # 1. Initial attempt
    # 2. Retry
    assert len(agent.planner.calls) == 2

    first_call = agent.planner.calls[0]
    second_call = agent.planner.calls[1]

    # Memory should be available on the initial planning attempt.
    assert first_call["experiences"]
    assert len(first_call["experiences"]) >= 1

    memory = first_call["experiences"][0]

    assert memory.task == "Fix a Python test"
    assert memory.improvement == (
        "Inspect the project structure before running the test."
    )

    # No retry context on the first attempt.
    assert first_call["retry_context"] is None

    # Retry context should appear on the second attempt.
    assert second_call["retry_context"] is not None

    retry_context = second_call["retry_context"]

    assert retry_context["diagnosis"] == (
        "The first attempt used the wrong approach."
    )

    assert retry_context["improvement"] == (
        "Use the previous experience before trying again."
    )

    # Memory should still be available during the retry.
    assert second_call["experiences"]
    assert len(second_call["experiences"]) >= 1

    print("✅ Memory + retry context integration test passed")


if __name__ == "__main__":
    test_memory_and_retry_context_reach_planner()

    print("\n✅ All memory + retry tests passed")