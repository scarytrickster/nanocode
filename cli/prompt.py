from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory


class InteractivePrompt:
    def __init__(self) -> None:
        self.session = PromptSession(
            history=InMemoryHistory(),
        )

    def get_input(self) -> str:
        return self.session.prompt("nanocode > ").strip()