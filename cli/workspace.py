from pathlib import Path


def get_workspace() -> Path:
    return Path.cwd().resolve()