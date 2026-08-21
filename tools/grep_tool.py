

from tools.base import Tool
from tools.ignore import iter_files
from tools.output_limit import limit_tool_output
from typing import Any
import re

class GrepTool(Tool):
    """Search files under a directory for lines matching a regex."""
    
    name = "grep"
    description = "Search files under a directory for lines matching a regex."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for."},
            "path": {"type": "string", "description": "Directory to search in.", "default": "."},
        },
        "required": ["pattern"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            regex = re.compile(args["pattern"])
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []
        search_path = args.get("path", ".")
        
        # iter_files prunes dependency/generated directories so recursive
        # exploration cannot pull an entire .venv into the context.
        for filepath in iter_files(search_path):
            try:
                with open(filepath, encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{filepath}:{lineno}: {line.rstrip()}")
            except (UnicodeDecodeError, OSError):
                continue
        
        if not matches:
            return "No matches found."

        # Traversal filtering keeps dependency trees out; output limiting
        # keeps a legitimate but huge match set from flooding the context.
        return limit_tool_output("\n".join(matches))