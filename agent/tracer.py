from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any


@dataclass
class TraceEvent:
    """Represents one event during an agent execution."""

    name: str
    timestamp: str
    component: str
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None
    error: str | None = None


class Tracer:
    """Collects and displays events from an agent run."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.events: list[TraceEvent] = []

    def record(
        self,
        name: str,
        component: str = "agent",
        duration_ms: float | None = None,
        error: str | None = None,
        **data: Any,
    ) -> None:
        """Record a structured tracing event."""

        if not self.enabled:
            return

        event = TraceEvent(
            name=name,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            component=component,
            data=data,
            duration_ms=duration_ms,
            error=error,
        )

        self.events.append(event)
        self._print_event(event)

    # ADD span() HERE
    @contextmanager
    def span(
        self,
        name: str,
        component: str = "agent",
        **data: Any,
    ) -> Iterator[None]:
        """
        Measure the duration of an operation and automatically
        record started, completed, or failed events.
        """

        start = perf_counter()

        self.record(
            f"{name}.started",
            component=component,
            **data,
        )

        try:
            yield

        except Exception as exc:
            duration_ms = (perf_counter() - start) * 1000

            self.record(
                f"{name}.failed",
                component=component,
                duration_ms=duration_ms,
                error=str(exc),
                **data,
            )

            raise

        else:
            duration_ms = (perf_counter() - start) * 1000

            self.record(
                f"{name}.completed",
                component=component,
                duration_ms=duration_ms,
                **data,
            )

    def _print_event(self, event: TraceEvent) -> None:
        """Print one structured trace event."""

        print(
            f"[TRACE] {event.timestamp} | "
            f"{event.component} | "
            f"{event.name}"
        )

        if event.duration_ms is not None:
            print(
                f"         duration: "
                f"{event.duration_ms:.2f} ms"
            )

        if event.error is not None:
            print(
                f"         error: "
                f"{event.error}"
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