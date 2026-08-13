from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from cli.commands import CommandHandler


class InteractivePrompt:
    def __init__(self) -> None:
        self.session = PromptSession(
            history=InMemoryHistory(),
            completer=WordCompleter(
                CommandHandler.COMMANDS,
                ignore_case=True,
                sentence=True,
            ),
        )

    def get_input(self) -> str:
        return self.session.prompt("nanocode > ").strip()