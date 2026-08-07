from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: str = ""


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    auto_approve: bool = False
    plan_mode: bool = False
    max_iterations: int = 50
