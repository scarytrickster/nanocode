import os
import platform
import sys


def get_system_prompt() -> str:
    """Generate the system prompt with environment info."""
    prompt = (
        "You are nanocode, a terminal coding agent. Be concise. Prefer tools over guessing.\n"
        "Use the todo_write tool to plan any task with more than a couple of steps.\n\n"
        f"Environment:\n"
        f"cwd: {os.getcwd()}\n"
        f"os: {platform.system()} {platform.release()}\n"
        f"python: {sys.version}\n"
        f"files in cwd: {', '.join(sorted(os.listdir()))}"
    )

    if os.path.exists("NANOCODE.md"):
        try:
            with open("NANOCODE.md", encoding="utf-8") as f:
                prompt += f"\n\nProject instructions:\n{f.read()}"
        except Exception:
            pass

    return prompt
