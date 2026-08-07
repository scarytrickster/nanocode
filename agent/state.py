import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from models.config import AgentConfig
from tools.base import Tool


class AgentStatus(str, Enum):
    """Possible statuses for an agent run."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState:
    """Single source of truth for one NanoCode agent execution."""

    task: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    config: AgentConfig = field(default_factory=AgentConfig)
    iteration: int = 0
    status: str = "running"
    final_response: str = ""


def get_system_prompt() -> str:
    """Generate the system prompt with environment info."""
    prompt = (
        "You are nanocode, a terminal coding agent. Be concise. Prefer tools over guessing.\n"
        "Use the todo_write tool to plan any task with more than a couple of steps.\n\n"
        f"Environment:\n"
        f"cwd: {os.getcwd()}\n"
        f"os: {platform.system()} {platform.release()}\n"
        f"python: {sys.version}\n"
        f"files in cwd: {', '.join(sorted(os.listdir()))}"
    )

    if os.path.exists("NANOCODE.md"):
        try:
            with open("NANOCODE.md", encoding="utf-8") as f:
                prompt += f"\n\nProject instructions:\n{f.read()}"
        except Exception:
            pass

    return prompt
