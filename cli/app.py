import os

from agent.agent import NanoCodeAgent
from config.settings import MODEL

from cli.banner import BannerInfo, render_banner
from cli.prompt import InteractivePrompt
from cli.workspace import get_workspace


class NanoCodeCLI:
    def __init__(self) -> None:
        self.agent = NanoCodeAgent()
        self.prompt = InteractivePrompt()
        self.running = True

    def run(self) -> None:
        self._print_startup()

        try:
            while self.running:
                task = self.prompt.get_input()

                if not task:
                    continue

                response = self.agent.run(task)

                if response:
                    print(f"\n{response}")

        except KeyboardInterrupt:
            self._shutdown()

        except EOFError:
            self._shutdown()

    def _print_startup(self) -> None:
        info = BannerInfo(
            model=MODEL,
            workspace=str(get_workspace()),
            memory_enabled=True,
            tools=["bash", "files", "web"],
        )

        print(render_banner(info))
        print("Type your task or press Ctrl+C to exit.")

    def _shutdown(self) -> None:
        if not self.running:
            return

        self.running = False
        print("\n\nGoodbye!")


def main() -> None:
    cli = NanoCodeCLI()
    cli.run()


if __name__ == "__main__":
    main()