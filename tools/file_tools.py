from tools.base import Tool
from tools.output_limit import limit_tool_output
from typing import Any


class ReadFileTool(Tool):
    """Read a file from disk."""
    
    name = "read_file"
    description = "Read a file from disk and return its contents."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
        },
        "required": ["path"],
    }
    is_read_only = True

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], encoding="utf-8") as f:
                # Reading any file stays allowed; only the amount handed
                # back to the model is bounded.
                return limit_tool_output(f.read())
        except FileNotFoundError:
            return f"Error: File not found: {args['path']}"
        except PermissionError:
            return f"Error: Permission denied: {args['path']}"
        except Exception as e:
            return f"Error reading file: {e}"


class WriteFileTool(Tool):
    """Write content to a file."""
    
    name = "write_file"
    description = "Write content to a file, creating or overwriting it."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(args["content"])
            return f"Wrote {args['path']}"
        except Exception as e:
            return f"Error writing file: {e}"


class EditFileTool(Tool):
    """Replace an exact string in a file."""
    
    name = "edit_file"
    description = "Replace an exact string in a file with a new string."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old_string": {"type": "string", "description": "Exact string to replace."},
            "new_string": {"type": "string", "description": "Replacement string."},
        },
        "required": ["path", "old_string", "new_string"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            with open(args["path"], encoding="utf-8") as f:
                content = f.read()
            
            if args["old_string"] not in content:
                return f"Error: old_string not found in {args['path']}"
            
            updated_content = content.replace(args["old_string"], args["new_string"])
            with open(args["path"], "w", encoding="utf-8") as f:
                f.write(updated_content)
            return f"Edited {args['path']}"
        except FileNotFoundError:
            return f"Error: File not found: {args['path']}"
        except Exception as e:
            return f"Error editing file: {e}"