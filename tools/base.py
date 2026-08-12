from dataclasses import field
from typing import Any


class Tool:
    """Base class for all tools."""
    
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    is_read_only: bool = False

    def execute(self, args: dict[str, Any]) -> str:
        """Execute the tool with given arguments."""
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }