from rlm.result import RLMResult

# Metadata keys describing how complete a synthesis actually was. A partial
# investigation that reports itself as a plain success is the dangerous case:
# the caller cannot tell that two of three perspectives were never examined.
PARTIAL_KEY = "partial"
CHILDREN_KEY = "children"
SUCCESSFUL_KEY = "successful_children"
FAILED_KEY = "failed_children"
RATE_LIMITED_KEY = "rate_limited_children"
FAILURES_KEY = "failures"


def _failure_details(results: list[RLMResult]) -> list[dict]:
    """One compact record per failed child, preserving why it failed."""

    return [
        {
            "task": result.metadata.get("task", ""),
            "error_type": result.metadata.get("error_type", "error"),
            "error": result.metadata.get("error", ""),
            "rate_limited": bool(result.metadata.get("rate_limited", False)),
        }
        for result in results
    ]


class RLMSynthesizer:
    """Combines results from recursive workers."""

    def synthesize(
        self,
        results: list[RLMResult],
    ) -> RLMResult:

        if not results:
            return RLMResult(
                answer="",
                success=False,
                metadata=self._completeness(
                    successful=[],
                    failed=[],
                ),
            )

        successful = [
            result
            for result in results
            if result.success
        ]

        failed = [
            result
            for result in results
            if not result.success
        ]

        metadata = self._completeness(
            successful=successful,
            failed=failed,
        )

        if not successful:
            return RLMResult(
                answer="",
                success=False,
                children_created=len(results),
                metadata=metadata,
            )

        answer = "\n\n".join(
            result.answer
            for result in successful
            if result.answer
        )

        return RLMResult(
            answer=answer,
            success=True,
            depth=max(
                result.depth
                for result in successful
            ),
            children_created=len(results),
            metadata=metadata,
        )

    def _completeness(
        self,
        successful: list[RLMResult],
        failed: list[RLMResult],
    ) -> dict:
        """Describe how much of the intended work actually completed.

        Failure information is carried forward rather than discarded, so the
        caller can say what was *not* investigated.
        """

        rate_limited = [
            result
            for result in failed
            if result.metadata.get("rate_limited")
        ]

        return {
            CHILDREN_KEY: len(successful) + len(failed),
            SUCCESSFUL_KEY: len(successful),
            FAILED_KEY: len(failed),
            RATE_LIMITED_KEY: len(rate_limited),
            PARTIAL_KEY: bool(failed),
            FAILURES_KEY: _failure_details(failed),
        }
