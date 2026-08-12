from agent.agent import NanoCodeAgent
from models.config import AgentConfig


def main() -> None:
    print("\n--- RETRY E2E TEST ---")

    config = AgentConfig(
        max_retries=1,
    )

    agent = NanoCodeAgent(
        config=config,
    )

    # Seed an experience so the real planner has memory available.
    from agent.memory import Experience

    agent.memory.add(
        Experience(
            task="Run a Python command incorrectly",
            diagnosis="The command used an invalid Python invocation.",
            improvement="Verify the active Python interpreter before running the command.",
            success=False,
        )
    )

    response = agent.run(
        "Find the Python version used in this environment. "
        "Use the available tools to verify the result."
    )

    print("\n--- FINAL RESPONSE ---")
    print(response)

    print("\n--- RETRY E2E TEST COMPLETED ---")


if __name__ == "__main__":
    main()