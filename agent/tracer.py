"""
agent/tracer.py

Lightweight execution tracing for NanoCode.

The tracer records important events during an agent run.
Later, these traces can be used for debugging, evaluation,
and recursive self-improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TraceEvent:
    """Represents one event during an agent execution."""

    name: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Collects and displays events from an agent run."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.events: list[TraceEvent] = []

    def record(
        self,
        name: str,
        **data: Any,
    ) -> None:
        """Record a tracing event."""

        if not self.enabled:
            return

        event = TraceEvent(
            name=name,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            data=data,
        )

        self.events.append(event)

        self._print_event(event)

    def _print_event(self, event: TraceEvent) -> None:
        """Print one trace event to the terminal."""

        print(
            f"[TRACE] {event.timestamp} | "
            f"{event.name}"
        )

        if event.data:
            for key, value in event.data.items():
                print(f"         {key}: {value}")

    def clear(self) -> None:
        """Remove all recorded events."""

        self.events.clear()

    def get_events(self) -> list[TraceEvent]:
        """Return all recorded events."""

        return list(self.events)