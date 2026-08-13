from dataclasses import dataclass


@dataclass
class CommandContext:
    model: str
    workspace: str
    memory_enabled: bool
    mode: str
    message_count: int = 0


class CommandHandler:
    COMMANDS = [
        "/help",
        "/model",
        "/mode",
        "/status",
        "/memory",
        "/clear",
        "/history",
        "/exit",
        "/quit",
    ]

    def __init__(self) -> None:
        self.should_exit = False

    def handle(
        self,
        command: str,
        context: CommandContext,
    ) -> str | None:

        

        command = command.strip().lower()

        if command == "/help":
            return self.help()

        if command == "/model":
            return self.model(context)

        if command == "/mode":
            return self.mode(context)

        if command == "/status":
            return self.status(context)

        if command == "/memory":
            return self.memory(context)

        if command == "/clear":
            return "__CLEAR_SESSION__"

        if command in {"/exit", "/quit"}:
            self.should_exit = True
            return "Goodbye!"

        if command == "/history":
            return (
                "Session history is currently managed by the CLI session."
            )

        return (
            f"Unknown command: {command}\n"
            "Type /help to see available commands."
        )

    def help(self) -> str:
        return (
            "NanoCode Commands\n"
            "\n"
            "  /help      Show available commands\n"
            "  /model     Show the current model\n"
            "  /status    Show NanoCode status\n"
            "  /mode      Show the current mode\n"
            "  /memory    Show memory status\n"
            "  /history   Show session history\n"
            "  /clear     Clear the current session\n"
            "  /exit      Exit NanoCode\n"
            "  /quit      Exit NanoCode"
        )

    def mode(self, context: CommandContext) -> str:
        return (
            f"Current mode: {context.mode}\n"
            "\n"
            "Available modes:\n"
            "  chat      Normal interactive agent mode\n"
            "  plan      Planning-focused mode\n"
            "  execute   Execution-focused mode"
        )

    def model(self, context: CommandContext) -> str:
        return f"Model: {context.model}"

    def status(self, context: CommandContext) -> str:
        memory = (
            "enabled"
            if context.memory_enabled
            else "disabled"
        )

        return (
            "NanoCode Status\n"
            f"  Model:     {context.model}\n"
            f"  Workspace: {context.workspace}\n"
            f"  Mode:      {context.mode}\n"
            f"  Memory:    {memory}\n"
            f"  Messages:  {context.message_count}"
        )

    def memory(self, context: CommandContext) -> str:
        status = (
            "enabled"
            if context.memory_enabled
            else "disabled"
        )

        return f"Memory: {status}"