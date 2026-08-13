from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SessionState:
    workspace: str
    mode: str = "chat"
    started_at: datetime = field(default_factory=datetime.now)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    def clear(self) -> None:
        self.messages.clear()

    @property
    def message_count(self) -> int:
        return len(self.messages)