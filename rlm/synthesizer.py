from rlm.evidence import CONFIRMED, CONFLICTING, analyze
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

# Metadata keys describing what the children actually established. Concatenated
# answers give a speculation the same weight as a verified defect; these carry
# the difference through to the caller.
REPORT_KEY = "report"
FINDINGS_KEY = "findings"
FINDINGS_COUNT_KEY = "findings_count"
CONFIRMED_COUNT_KEY = "confirmed_findings"
CONFLICTING_COUNT_KEY = "conflicting_findings"
PRIMARY_KEY = "primary_finding"
CONFIDENCE_KEY = "confidence"


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


def render_report(groups, primary) -> str:
    """Render ranked findings as one coherent answer.

    Claims are reproduced verbatim from the children, so nothing a child
    reported is lost or reworded into something it did not say.
    """

    if not groups:
        return ""

    lines: list[str] = []

    if primary is not None:
        lines.append("Primary finding:")
        lines.append(primary.claim)
        lines.append("")

        supporters = primary.supporting_children

        if len(supporters) > 1:
            lines.append(
                "Why: "
                f"{len(supporters)} independent child analyses "
                f"({', '.join(f'child {child}' for child in supporters)}) "
                "identified this issue."
            )
        elif primary.evidence:
            lines.append(
                f"Why: child {supporters[0]} identified it with "
                "supporting evidence."
            )
        else:
            lines.append(f"Why: reported by child {supporters[0]}.")

        lines.append("")

        if primary.evidence:
            lines.append("Evidence:")

            for item in primary.evidence:
                lines.append(f"- {item}")

            lines.append("")

        # Grouping chooses one headline wording; the other children's own
        # phrasing is kept so no reported detail is lost.
        supporting = [
            (child, claim)
            for child, claim in primary.statements
            if claim != primary.claim
        ]

        if supporting:
            lines.append("Also reported as:")

            for child, claim in supporting:
                lines.append(f"- {claim} (child {child})")

            lines.append("")
    else:
        lines.append(
            "No finding was supported well enough to be treated as primary."
        )
        lines.append("")

    others = [group for group in groups if group is not primary]

    if others:
        lines.append("Other findings:")

        for group in others:
            for child, claim in group.statements:
                lines.append(f"- [{group.status}] {claim} (child {child})")

        lines.append("")

    conflicts = [group for group in groups if group.conflict_reason]

    if conflicts:
        lines.append("Conflicting analysis:")

        for group in conflicts:
            lines.append(f"- {group.claim} -- {group.conflict_reason}")

        lines.append("")

    confidence = primary.confidence if primary is not None else "low"

    lines.append(f"Confidence: {confidence}")

    return "\n".join(lines).strip() + "\n"


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

        # The joined answer is the existing RLMResult contract and is left
        # exactly as it was. The evidence analysis is additive: it travels in
        # metadata, and the caller decides how to present it.
        answer = "\n\n".join(
            result.answer
            for result in successful
            if result.answer
        )

        metadata.update(self._evidence(results))

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

    def _evidence(self, results: list[RLMResult]) -> dict:
        """Analyze the successful children's answers into ranked findings.

        Only successful children are read: a failed or rate-limited child
        produced no evidence, and reading its silence as agreement would
        overstate the result.
        """

        answers = [
            (
                index,
                result.answer or "",
                str(result.metadata.get("task", "")),
            )
            for index, result in enumerate(results, start=1)
            if result.success and (result.answer or "").strip()
        ]

        groups, primary = analyze(answers)

        return {
            FINDINGS_KEY: [group.to_dict() for group in groups],
            FINDINGS_COUNT_KEY: len(groups),
            CONFIRMED_COUNT_KEY: sum(
                1 for group in groups if group.status == CONFIRMED
            ),
            CONFLICTING_COUNT_KEY: sum(
                1 for group in groups if group.status == CONFLICTING
            ),
            PRIMARY_KEY: primary.claim if primary is not None else "",
            CONFIDENCE_KEY: primary.confidence if primary is not None else "low",
            REPORT_KEY: render_report(groups, primary),
        }

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
