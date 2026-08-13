from agent.agent import NanoCodeAgent
from config.settings import MODEL

from cli.banner import BannerInfo, render_banner
from cli.commands import CommandContext, CommandHandler
from cli.prompt import InteractivePrompt
from cli.renderer import TerminalRenderer
from cli.session import SessionState
from cli.workspace import get_workspace


class NanoCodeCLI:
    def __init__(self) -> None:
        self.agent = NanoCodeAgent(
            console_trace=False,
            trace_callback=self._handle_trace,

        )        
        self.prompt = InteractivePrompt()
        self.commands = CommandHandler()
        self.renderer = TerminalRenderer()

        self.session = SessionState(
            workspace=str(get_workspace()),
        )

        self.running = True

    def run(self) -> None:
        self._print_startup()

        try:
            while self.running:
                task = self.prompt.get_input()

                if not task:
                    continue

                # Handle CLI commands locally.
                if task.startswith("/"):
                    self._handle_command(task)
                    continue

                # Store user message in the current session.
                self.session.add_message("user", task)

                # Send normal tasks to the agent.
                response = self.agent.run(task)

                if response:
                    self.session.add_message("assistant", response)
                    self.renderer.print_response(response)

        except KeyboardInterrupt:
            self._shutdown()

        except EOFError:
            self._shutdown()

    def _handle_command(self, command: str) -> None:

        if command.strip().lower() == "/history":
            self._show_history()
            return
        context = CommandContext(
            model=MODEL,
            workspace=self.session.workspace,
            memory_enabled=True,
            mode=self.session.mode,
            message_count=self.session.message_count,
        )

        result = self.commands.handle(command, context)

        if result == "__CLEAR_SESSION__":
            self._clear_session()
            return

        if result:
            self.renderer.print_command_result(result)

        if self.commands.should_exit:
            self.running = False

    def _handle_trace(self, event) -> None:
        self.renderer.handle_trace(event)

    def _show_history(self) -> None:
        if not self.session.messages:
            self.renderer.print_command_result(
                "No messages in the current session."
            )
            return

        lines = ["Session History"]

        for message in self.session.messages:
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "")

            # Don't display the system prompt.
            if role.lower() == "system":
                continue

            lines.append(f"\n{role}:")
            lines.append(content)

        self.renderer.print_command_result(
            "\n".join(lines)
        )

    def _clear_session(self) -> None:
        self.session.clear()
        self.renderer.print_success("Session cleared.")

    def _print_startup(self) -> None:
        workspace = get_workspace()

        info = BannerInfo(
            model=MODEL,
            workspace=str(workspace),
            memory_enabled=True,
            tools=["bash", "files", "web"],
        )

        print(render_banner(info))
        print("Type your task or press Ctrl+C to exit.")

    def _shutdown(self) -> None:
        if not self.running:
            return

        self.running = False
        self.renderer.print_success("Goodbye!")


def main() -> None:
    cli = NanoCodeCLI()
    cli.run()


if __name__ == "__main__":
    main()