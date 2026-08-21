from dataclasses import dataclass

from agent.state import AgentState, AgentStatus
from agent.tracer import Tracer
from langfuse import get_client


@dataclass
class EvaluationResult:
    """Result of evaluating one agent execution."""

    success: bool
    score: float
    reason: str


class Evaluator:
    """Evaluates the outcome of an agent execution."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()

    def evaluate(self, state: AgentState) -> EvaluationResult:
        """Evaluate the final state of an agent run."""

        langfuse = get_client()

        with langfuse.start_as_current_observation(
            as_type="span",
            name="evaluator",
            input={
                "status": state.status,
                "final_response": state.final_response,
            },
        ) as obs:
            result = self._evaluate(state)

            obs.update(
                output={
                    "success": result.success,
                    "score": result.score,
                    "reason": result.reason,
                }
            )

        return result

    def _evaluate(self, state: AgentState) -> EvaluationResult:
        # -------------------------------------------------
        # Completed execution
        # -------------------------------------------------

        if state.status == AgentStatus.COMPLETED:

            if state.final_response.strip():

                result = EvaluationResult(
                    success=True,
                    score=1.0,
                    reason="Task completed successfully.",
                )

            else:

                result = EvaluationResult(
                    success=False,
                    score=0.0,
                    reason="Agent completed without a final response.",
                )

        # -------------------------------------------------
        # Failed execution
        # -------------------------------------------------

        elif state.status == AgentStatus.FAILED:

            result = EvaluationResult(
                success=False,
                score=0.0,
                reason="Agent execution failed.",
            )

        # -------------------------------------------------
        # Maximum iterations
        # -------------------------------------------------

        elif state.status == AgentStatus.MAX_ITERATIONS:

            result = EvaluationResult(
                success=False,
                score=0.0,
                reason="Agent reached the maximum number of iterations.",
            )

        # -------------------------------------------------
        # Unknown / unexpected status
        # -------------------------------------------------

        else:

            result = EvaluationResult(
                success=False,
                score=0.0,
                reason=f"Agent ended with status: {state.status}",
            )

        # -------------------------------------------------
        # Trace evaluation
        # -------------------------------------------------

        self.tracer.record(
            "evaluation.completed",
            component="evaluator",
            success=result.success,
            score=result.score,
            reason=result.reason,
        )

        return result   