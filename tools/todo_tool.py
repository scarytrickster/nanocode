from tools.base import Tool
from typing import Any

class TodoWriteTool(Tool):
    """Manage task lists."""
    
    name = "todo_write"
    description = "Write the current task list. Replaces the whole list each call."
    parameters = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["items"],
    }
    is_read_only = True

    def __init__(self):
        self.items: list[dict[str, str]] = []

    def execute(self, args: dict[str, Any]) -> str:
        self.items = args["items"]
        marks = {"pending": " ", "in_progress": "~", "done": "x"}
        return "\n".join(
            f"[{marks[item['status']]}] {item['content']}" 
            for item in self.items
        )
