from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rlm.call import RLMCallHandler
from rlm.context import RLMContext
from rlm.result import RLMResult
from agent.agent import NanoCodeAgent


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
    ) -> None:
        self.agent_factory = agent_factory
        self.last_request: NanoCodeRequest | None = None

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

        try:
            # NanoCodeAgent.run() currently accepts a string task.
            response = agent.run(request.task)
        except Exception as error:
            # One child crashing must not abort its siblings or corrupt the
            # parent context. It becomes an unsuccessful result, which is the
            # failure shape RLMSynthesizer already knows how to ignore.
            return RLMResult(
                answer="",
                success=False,
                depth=context.depth,
                metadata={"error": str(error)},
            )

        return self._to_rlm_result(
            context=context,
            response=response,
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
    ) -> RLMResult:
        """Convert a NanoCode response into an RLMResult."""

        if isinstance(response, RLMResult):
            return response

        if isinstance(response, str):
            return RLMResult(
                answer=response,
                success=True,
                depth=context.depth,
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

