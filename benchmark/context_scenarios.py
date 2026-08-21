"""Deterministic context-budget scenarios.

Synthetic conversations of known size, measured through the real
ContextBudgetManager. No model, no network: this measures the context layer
itself, not an agent run.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import config.settings as settings
from agent.context_budget import ContextBudgetManager, estimate_tokens
from tools.output_limit import DEFAULT_MAX_OUTPUT_CHARS


@dataclass(frozen=True)
class ContextScenario:
    """One synthetic conversation shape."""

    name: str
    exchanges: int
    output_chars: int

    def messages(self) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": "You are nanocode, a terminal coding agent."},
            {"role": "user", "content": "Find the authentication bug across the project."},
        ]

        for index in range(self.exchanges):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": '{"pattern": "token"}',
                            },
                        }
                    ],
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{index}",
                    "content": "R" * self.output_chars,
                }
            )

        return messages


SCENARIOS: tuple[ContextScenario, ...] = (
    ContextScenario(name="small", exchanges=2, output_chars=500),
    ContextScenario(name="moderate", exchanges=10, output_chars=4_000),
    ContextScenario(name="large", exchanges=30, output_chars=20_000),
    # The original failure: 80 exchanges of tool output already capped at the
    # Phase 6.6 limit, which together passed the model's window.
    ContextScenario(
        name="regression-371k",
        exchanges=80,
        output_chars=DEFAULT_MAX_OUTPUT_CHARS,
    ),
)


@dataclass
class ContextScenarioResult:
    """What the budget layer did to one scenario."""

    name: str
    original_estimated_tokens: int
    final_estimated_tokens: int
    compressed: bool
    compression_passes: int
    entries_compressed: int
    entries_dropped: int
    hard_capped: bool
    duration_seconds: float

    @property
    def compression_ratio(self) -> float:
        if not self.original_estimated_tokens:
            return 1.0

        return round(
            self.final_estimated_tokens / self.original_estimated_tokens, 4
        )

    @property
    def within_budget(self) -> bool:
        return self.final_estimated_tokens <= settings.CONTEXT_TOKEN_BUDGET

    @property
    def exceeded_model_limit(self) -> bool:
        """Whether the uncompressed context would have overflowed the model."""

        return self.original_estimated_tokens > settings.MODEL_CONTEXT_LIMIT


def run_scenario(scenario: ContextScenario) -> ContextScenarioResult:
    messages = scenario.messages()

    manager = ContextBudgetManager()

    started = perf_counter()

    result = manager.analyze(messages)

    duration = perf_counter() - started

    return ContextScenarioResult(
        name=scenario.name,
        original_estimated_tokens=result.original_estimated_tokens,
        final_estimated_tokens=result.final_estimated_tokens,
        compressed=result.compressed,
        compression_passes=result.compression_passes,
        entries_compressed=result.entries_compressed,
        entries_dropped=result.entries_dropped,
        hard_capped=result.hard_capped,
        duration_seconds=round(duration, 4),
    )


def run_context_benchmark(scenarios=SCENARIOS) -> list[ContextScenarioResult]:
    return [run_scenario(scenario) for scenario in scenarios]


def render_context_report(results: list[ContextScenarioResult]) -> str:
    width = 78

    lines = [
        "=" * width,
        "NanoCode Context Budget",
        "=" * width,
        f"Configured budget: {settings.CONTEXT_TOKEN_BUDGET:,} estimated tokens "
        f"(model limit {settings.MODEL_CONTEXT_LIMIT:,}, "
        f"margin {settings.CONTEXT_SAFETY_MARGIN:.0%})",
        "-" * width,
        f"{'Scenario':<18}{'Before':>10}{'After':>10}{'Ratio':>9}"
        f"{'Passes':>8}{'Within':>9}{'Sec':>8}",
        "-" * width,
    ]

    for result in results:
        lines.append(
            f"{result.name:<18}"
            f"{result.original_estimated_tokens:>10,}"
            f"{result.final_estimated_tokens:>10,}"
            f"{result.compression_ratio:>9}"
            f"{result.compression_passes:>8}"
            f"{('yes' if result.within_budget else 'NO'):>9}"
            f"{result.duration_seconds:>8}"
        )

    lines.append("-" * width)
    lines.append(
        "Before/After are estimated tokens, not exact counts. "
        "No model was called."
    )
    lines.append("=" * width)

    return "\n".join(lines)


if __name__ == "__main__":
    print(render_context_report(run_context_benchmark()))
