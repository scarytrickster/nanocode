from agent.agent import NanoCodeAgent

from cli.prompt import InteractivePrompt


def main() -> None:
    agent = NanoCodeAgent()
    prompt = InteractivePrompt()

    print("NanoCode CLI")
    print("Type your task or press Ctrl+C to exit.")

    while True:
        try:
            task = prompt.get_input()

            if not task:
                continue

            response = agent.run(task)

            print("\n" + response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()