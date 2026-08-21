from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import RateLimitError

from rlm.call import RLMCallHandler
from rlm.context import RLMContext
from rlm.result import RLMResult
from agent.agent import NanoCodeAgent

# Error classifications recorded in RLMResult.metadata["error_type"].
ERROR_TYPE_RATE_LIMIT = "rate_limit"
ERROR_TYPE_ERROR = "error"

# Bounded application-level retry for rate limits only. This is per child and
# never scales with the number of children: three children retrying once each
# is three extra requests, not nine.
DEFAULT_RATE_LIMIT_RETRIES = 1
DEFAULT_RATE_LIMIT_DELAY = 5.0


def is_rate_limit_error(error: BaseException) -> bool:
    """True only for real provider rate limits, never for other failures."""

    return isinstance(error, RateLimitError)


def classify_error(error: BaseException) -> str:
    """The error_type recorded for a failed child."""

    return (
        ERROR_TYPE_RATE_LIMIT
        if is_rate_limit_error(error)
        else ERROR_TYPE_ERROR
    )


@dataclass(frozen=True)
class NanoCodeRequest:
    """Information passed from an RLM child context to NanoCode."""

    task: str
    content: str
    depth: int
    metadata: dict[str, Any]

def create_nanocode_agent() -> NanoCodeAgent:
    """
    Create a fresh NanoCodeAgent for an RLM child execution.

    Children run the normal NanoCode pipeline: rlm_enabled=False stops a child
    from routing back into the RLM path.
    """

    return NanoCodeAgent(
            console_trace=False,
            rlm_enabled=False,
    )


class NanoCodeCallHandler(RLMCallHandler):
    """
    Adapter between the RLM runtime and NanoCodeAgent.

    The adapter keeps RLM-specific context separate from the existing
    NanoCodeAgent interface.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.agent_factory = agent_factory
        self.last_request: NanoCodeRequest | None = None

        # `sleep` is injectable so tests can prove the retry happened without
        # actually waiting.
        self.rate_limit_retries = max(0, rate_limit_retries)
        self.rate_limit_delay = rate_limit_delay
        self.sleep = sleep

    def call(self, context: RLMContext) -> RLMResult:
        """Execute a child task using NanoCodeAgent."""

        agent = self.agent_factory()

        request = NanoCodeRequest(
            task=context.task,
            content=context.content,
            depth=context.depth,
            metadata=dict(context.metadata),
        )

        self.last_request = request

        attempts = 0

        while True:
            attempts += 1

            try:
                # NanoCodeAgent.run() currently accepts a string task.
                response = agent.run(request.task)

            except Exception as error:
                retryable = (
                    is_rate_limit_error(error)
                    and attempts <= self.rate_limit_retries
                )

                if retryable:
                    # Only rate limits are retried: a deterministic failure
                    # would fail again identically and just burn budget.
                    self.sleep(self.rate_limit_delay)

                    # A fresh agent per attempt keeps attempts isolated, the
                    # same way each child gets its own agent.
                    agent = self.agent_factory()

                    continue

                # One child failing must not abort its siblings or corrupt the
                # parent context. It becomes an unsuccessful result, which is
                # the failure shape RLMSynthesizer already knows how to handle.
                return self._to_failure(
                    context=context,
                    error=error,
                    attempts=attempts,
                )

            return self._to_rlm_result(
                context=context,
                response=response,
                attempts=attempts,
            )

    def _to_failure(
        self,
        context: RLMContext,
        error: BaseException,
        attempts: int,
    ) -> RLMResult:
        """Convert a child exception into the existing failed-result shape."""

        error_type = classify_error(error)

        return RLMResult(
            answer="",
            success=False,
            depth=context.depth,
            metadata={
                "error": str(error),
                "error_type": error_type,
                "rate_limited": error_type == ERROR_TYPE_RATE_LIMIT,
                "attempts": attempts,
                "task": context.task,
            },
        )

    def create_nanocode_agent() -> NanoCodeAgent:
        """
        Create a fresh NanoCodeAgent for an RLM child execution.

        A fresh instance prevents child executions from accidentally
        sharing mutable agent state with the parent.
        """

        return NanoCodeAgent(
            console_trace=False,
        )

    def _to_rlm_result(
        self,
        context: RLMContext,
        response: Any,
        attempts: int = 1,
    ) -> RLMResult:
        """Convert a NanoCode response into an RLMResult."""

        if isinstance(response, RLMResult):
            return response

        if isinstance(response, str):
            return RLMResult(
                answer=response,
                success=True,
                depth=context.depth,
                metadata={"attempts": attempts},
            )

        if response is None:
            return RLMResult(
                answer="",
                success=False,
                depth=context.depth,
            )

        answer = getattr(response, "answer", None)

        if answer is not None:
            success = getattr(response, "success", True)

            return RLMResult(
                answer=str(answer),
                success=bool(success),
                depth=context.depth,
            )

        return RLMResult(
            answer=str(response),
            success=True,
            depth=context.depth,
        )

