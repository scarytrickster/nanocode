from typing import Optional

from tools.base import Tool
from tools.bash_tools import BashTool
from tools.file_tools import EditFileTool, ReadFileTool, WriteFileTool
from tools.grep_tool import GrepTool
from tools.spawn_agent_tool import SpawnAgentTool
from tools.todo_tool import TodoWriteTool
from tools.web_tools import WebFetchTool, WebSearchTool

_all_tools: Optional[list[Tool]] = None


def get_all_tools() -> list[Tool]:
    """Get all available tools (cached)."""
    global _all_tools
    if _all_tools is None:
        _all_tools = [
            ReadFileTool(),
            WriteFileTool(),
            EditFileTool(),
            GrepTool(),
            BashTool(),
            TodoWriteTool(),
            WebFetchTool(),
            WebSearchTool(),
            SpawnAgentTool(),
        ]
    return _all_tools
