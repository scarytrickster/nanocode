import sys

from agent.agent import NanoCodeAgent
from models.config import AgentConfig


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python cli.py "your task here"')
        return

    task = " ".join(sys.argv[1:])

    agent = NanoCodeAgent(
        config=AgentConfig()
    )

    response = agent.run(task)

    print("\n--- FINAL RESPONSE ---")
    print(response)


if __name__ == "__main__":
    main()