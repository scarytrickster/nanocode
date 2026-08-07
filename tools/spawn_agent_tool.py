from typing import Any

from tools.base import Tool


class SpawnAgentTool(Tool):
    """Spawn a sub-agent with a fresh context."""

    name = "task"
    description = "Spawn a sub-agent with a fresh context to do a task; returns its final answer."
    parameters = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short description of the task."},
            "prompt": {"type": "string", "description": "Full instructions for the sub-agent."},
        },
        "required": ["description", "prompt"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        from agent import get_system_prompt, run_agent
        from models.config import AgentConfig
        from tools import get_all_tools

        sub_tools = [t for t in get_all_tools() if t.name != self.name]
        sub_messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": args["prompt"]},
        ]
        return run_agent(sub_messages, sub_tools, config=AgentConfig(auto_approve=True))
