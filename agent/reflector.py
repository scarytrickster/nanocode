from dataclasses import dataclass

from agent.evaluator import EvaluationResult
from agent.state import AgentState
from agent.tracer import Tracer
from langfuse import get_client


@dataclass
class ReflectionResult:
    """Result of reflecting on one agent execution."""

    should_improve: bool
    diagnosis: str
    improvement: str


class Reflector:
    """Analyzes an agent execution and suggests improvements."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()

    def reflect(
        self,
        state: AgentState,
        evaluation: EvaluationResult,
    ) -> ReflectionResult:
        """Reflect on the execution result."""

        langfuse = get_client()

        with langfuse.start_as_current_observation(
            as_type="span",
            name="reflection",
            input={
                "success": evaluation.success,
                "reason": evaluation.reason,
            },
        ) as obs:
            result = self._reflect(state, evaluation)

            obs.update(
                output={
                    "should_improve": result.should_improve,
                    "diagnosis": result.diagnosis,
                    "improvement": result.improvement,
                }
            )

        return result

    def _reflect(
        self,
        state: AgentState,
        evaluation: EvaluationResult,
    ) -> ReflectionResult:
        # -------------------------------------------------
        # Successful execution
        # -------------------------------------------------

        if evaluation.success:

            result = ReflectionResult(
                should_improve=False,
                diagnosis="The task was completed successfully.",
                improvement="No improvement required.",
            )

        # -------------------------------------------------
        # Failed execution
        # -------------------------------------------------

        else:

            result = ReflectionResult(
                should_improve=True,
                diagnosis=evaluation.reason,
                improvement=(
                    "Review the execution trace, identify the failure, "
                    "and adjust the strategy before the next attempt."
                ),
            )

        # -------------------------------------------------
        # Record reflection
        # -------------------------------------------------

        self.tracer.record(
            "reflection.completed",
            component="reflector",
            should_improve=result.should_improve,
            diagnosis=result.diagnosis,
            improvement=result.improvement,
        )

        return result