import sys

from langfuse import get_client

from agent.agent import NanoCodeAgent
from config.settings import require_api_key
from models.config import AgentConfig


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python cli.py "your task here"')
        return

    require_api_key()

    task = " ".join(sys.argv[1:])

    agent = NanoCodeAgent(
        config=AgentConfig()
    )

    try:
        response = agent.run(task)

        print("\n--- FINAL RESPONSE ---")
        print(response)
    finally:
        get_client().flush()


if __name__ == "__main__":
    main()