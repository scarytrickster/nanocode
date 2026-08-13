from dataclasses import dataclass


@dataclass
class BannerInfo:
    model: str
    workspace: str
    memory_enabled: bool
    tools: list[str]
    version: str = "0.1.0"


def render_banner(info: BannerInfo) -> str:
    memory_status = "enabled" if info.memory_enabled else "disabled"
    tools = " • ".join(info.tools) if info.tools else "none"

    return f"""
╭──────────────────────────────────────────────╮
│                  NANOCODE                    │
│                                              │
│  Version:   {info.version:<31}│
│  Model:     {info.model:<31}│
│  Workspace: {info.workspace:<31}│
│  Memory:    {memory_status:<31}│
│  Tools:     {tools:<31}│
╰──────────────────────────────────────────────╯
"""