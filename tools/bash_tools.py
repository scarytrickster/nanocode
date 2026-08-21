


from tools.base import Tool
from tools.output_limit import limit_tool_output
import subprocess   
from typing import Any

class BashTool(Tool):
    """Run a shell command."""
    
    name = "bash"
    description = "Run a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
        },
        "required": ["command"],
    }
    is_read_only = False

    def execute(self, args: dict[str, Any]) -> str:
        try:
            result = subprocess.run(
                args["command"], 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30
            )
            # A command like `find /` can emit megabytes; bound it before
            # it reaches the model's context.
            return limit_tool_output(result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error running command: {e}"