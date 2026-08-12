from agent.memory import Experience
from agent.planner import Planner
from agent.tracer import Tracer


def test_planner_retry_context():
    planner = Planner(tracer=Tracer())

    retry_context = {
        "diagnosis": "The command used the wrong file path.",
        "improvement": "Inspect the project structure before retrying.",
    }

    prompt = planner._build_user_prompt(
        task="Fix the failing Python test.",
        experiences=[],
        retry_context=retry_context,
    )

    assert "Previous attempt failed:" in prompt
    assert "The command used the wrong file path." in prompt
    assert "Inspect the project structure before retrying." in prompt
    assert "Current task:" in prompt
    assert "Fix the failing Python test." in prompt

    print("✅ Planner retry context test passed")


def test_planner_retry_context_with_memory():
    planner = Planner(tracer=Tracer())

    experiences = [
        Experience(
            task="Fix a Python test",
            diagnosis="The test failed because of an incorrect file path.",
            improvement="Inspect the project structure first.",
            success=False,
        )
    ]

    retry_context = {
        "diagnosis": "The second attempt still used the wrong path.",
        "improvement": "Verify the exact file location before running the test.",
    }

    prompt = planner._build_user_prompt(
        task="Fix the failing Python test.",
        experiences=experiences,
        retry_context=retry_context,
    )

    # Previous memory should be present
    assert "Fix a Python test" in prompt
    assert "Inspect the project structure first." in prompt

    # Current retry context should also be present
    assert "The second attempt still used the wrong path." in prompt
    assert "Verify the exact file location before running the test." in prompt

    # Current task should remain present
    assert "Fix the failing Python test." in prompt

    print("✅ Planner memory + retry context test passed")


if __name__ == "__main__":
    test_planner_retry_context()
    test_planner_retry_context_with_memory()
    print("\n✅ All planner retry tests passed")