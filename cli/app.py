from agent.agent import NanoCodeAgent


def main() -> None:
    agent = NanoCodeAgent()

    print("NanoCode CLI")
    print("Type your task or press Ctrl+C to exit.")

    while True:
        try:
            task = input("\nnanocode > ").strip()

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