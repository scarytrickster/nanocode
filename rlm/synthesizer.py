from rlm.result import RLMResult


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
            )

        successful = [
            result
            for result in results
            if result.success
        ]

        if not successful:
            return RLMResult(
                answer="",
                success=False,
                children_created=len(results),
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
        )