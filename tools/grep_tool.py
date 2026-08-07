

from tools.base import Tool
from typing import Any
import os
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
        
        for dirpath, _, filenames in os.walk(search_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        for lineno, line in enumerate(f, 1):
                            if regex.search(line):
                                matches.append(f"{filepath}:{lineno}: {line.rstrip()}")
                except (UnicodeDecodeError, OSError):
                    continue
        
        return "\n".join(matches) if matches else "No matches found."