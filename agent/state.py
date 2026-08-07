"""
agent/state.py

Defines the runtime state for a single NanoCode agent execution.
Every module (planner, executor, reflection, memory, etc.) will
read from and write to this object.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from models.config import AgentConfig
from tools.base import Tool


class AgentStatus(str, Enum):
    """Current execution status of the agent."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState:
    """
    Represents the complete runtime state of one execution.

    This object is shared across every module in the agent.
    It intentionally contains only runtime data—not business logic.
    """

    # ------------------------------------------------------------------
    # Task Information
    # ------------------------------------------------------------------

    task: str = ""

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    messages: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Available Tools
    # ------------------------------------------------------------------

    tools: list[Tool] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    config: AgentConfig = field(default_factory=AgentConfig)

    # ------------------------------------------------------------------
    # Runtime Information
    # ------------------------------------------------------------------

    iteration: int = 0

    status: AgentStatus = AgentStatus.IDLE

    final_response: str = ""

    error: str | None = None

    @property
    def is_finished(self) -> bool:
        """
        Returns True if execution has completed.
        """
        return self.status in (
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
        )

    def reset(self) -> None:
        """
        Reset the runtime state so it can be reused.
        """
        self.messages.clear()
        self.iteration = 0
        self.status = AgentStatus.IDLE
        self.final_response = ""
        self.error = None