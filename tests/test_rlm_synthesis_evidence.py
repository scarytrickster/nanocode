"""Tests for evidence-based RLM synthesis.

Deterministic: real RLMResult objects in, real synthesizer out. No OpenRouter,
no LLM, no network.

Child results are built with the actual RLMResult shape the orchestrator
already passes to the synthesizer -- no parallel abstraction.
"""

import pytest

from rlm.evidence import (
    CONFIRMED,
    CONFLICTING,
    FOLLOW_UP,
    HYPOTHESIS,
    LIKELY,
    POSSIBLE,
    UNSUPPORTED,
    are_related,
    classify_category,
    classify_kind,
    collect_evidence,
    FUNCTIONAL,
    SECURITY,
)
from rlm.result import RLMResult
from rlm.synthesizer import (
    CONFIDENCE_KEY,
    CONFIRMED_COUNT_KEY,
    CONFLICTING_COUNT_KEY,
    FAILED_KEY,
    FINDINGS_COUNT_KEY,
    FINDINGS_KEY,
    PARTIAL_KEY,
    PRIMARY_KEY,
    RATE_LIMITED_KEY,
    REPORT_KEY,
    RLMSynthesizer,
    SUCCESSFUL_KEY,
)


def child(answer: str, success: bool = True, task: str = "", **metadata) -> RLMResult:
    """A child result in the shape the runtime actually produces."""

    data = {"task": task, **metadata}

    return RLMResult(
        answer=answer,
        success=success,
        depth=1,
        metadata=data,
    )


def rate_limited(task: str = "") -> RLMResult:
    return RLMResult(
        answer="",
        success=False,
        depth=1,
        metadata={
            "task": task,
            "error": "Error code: 429 - rate-limited upstream",
            "error_type": "rate_limit",
            "rate_limited": True,
            "attempts": 2,
        },
    )


def failed(task: str = "") -> RLMResult:
    return RLMResult(
        answer="",
        success=False,
        depth=1,
        metadata={
            "task": task,
            "error": "child exploded",
            "error_type": "error",
            "rate_limited": False,
            "attempts": 1,
        },
    )


def synthesize(results):
    return RLMSynthesizer().synthesize(results)


def findings_of(result):
    return result.metadata[FINDINGS_KEY]


def finding_matching(result, needle: str):
    for finding in findings_of(result):
        if needle.lower() in finding["claim"].lower():
            return finding

    raise AssertionError(f"no finding matching {needle!r}")


# ---------------------------------------------------------------------------
# 1. A single clear finding with concrete evidence
# ---------------------------------------------------------------------------

def test_one_child_with_concrete_evidence_is_confirmed():

    result = synthesize(
        [
            child(
                "auth.py validate_token() divides expires_at by 1000, "
                "so the expiry comparison is wrong."
            )
        ]
    )

    finding = findings_of(result)[0]

    assert finding["status"] == CONFIRMED
    assert finding["confidence"] == "high"
    assert finding["supporting_children"] == [1]
    assert finding["evidence"]

    assert result.metadata[PRIMARY_KEY] == finding["claim"]
    assert result.metadata[CONFIDENCE_KEY] == "high"


def test_a_bare_assertion_without_evidence_is_not_confirmed():

    result = synthesize([child("Authentication is broken somewhere.")])

    finding = findings_of(result)[0]

    assert finding["status"] != CONFIRMED
    assert finding["evidence"] == []


# ---------------------------------------------------------------------------
# 2 & 11. The same finding from several children
# ---------------------------------------------------------------------------

def test_the_same_finding_from_two_children_is_grouped_once():

    result = synthesize(
        [
            child("auth.py validate_token divides the timestamp incorrectly."),
            child("validate_token compares timestamp units incorrectly."),
        ]
    )

    assert result.metadata[FINDINGS_COUNT_KEY] == 1

    finding = findings_of(result)[0]

    assert finding["supporting_children"] == [1, 2]
    assert finding["status"] == CONFIRMED


def test_independent_confirmation_raises_confidence():

    single = synthesize([child("The token expiry comparison is incorrect.")])

    agreed = synthesize(
        [
            child("The token expiry comparison is incorrect."),
            child("The token expiry comparison is incorrect."),
        ]
    )

    lone = findings_of(single)[0]
    shared = findings_of(agreed)[0]

    assert lone["supporting_children"] == [1]
    assert shared["supporting_children"] == [1, 2]

    # Two independent children beat one asserting the same thing alone.
    assert lone["status"] == UNSUPPORTED
    assert shared["status"] == LIKELY


def test_grouped_findings_keep_every_child_wording():

    result = synthesize(
        [
            child("auth.py validate_token divides the timestamp incorrectly."),
            child("validate_token compares timestamp units incorrectly."),
        ]
    )

    report = result.metadata[REPORT_KEY]

    assert "auth.py validate_token divides the timestamp incorrectly." in report
    assert "validate_token compares timestamp units incorrectly." in report

    assert "2 independent child analyses" in report


# ---------------------------------------------------------------------------
# 3. Different findings stay separate
# ---------------------------------------------------------------------------

def test_unrelated_findings_are_not_merged():

    result = synthesize(
        [
            child("auth.py validate_token mis-converts the expiry timestamp."),
            child("SECRET_KEY is hardcoded in config.py."),
        ]
    )

    assert result.metadata[FINDINGS_COUNT_KEY] == 2

    claims = [finding["claim"] for finding in findings_of(result)]

    assert any("validate_token" in claim for claim in claims)
    assert any("SECRET_KEY" in claim for claim in claims)


def test_the_report_separates_primary_from_other_findings():

    result = synthesize(
        [
            child("auth.py validate_token mis-converts the expiry timestamp."),
            child("SECRET_KEY is hardcoded in config.py."),
        ]
    )

    report = result.metadata[REPORT_KEY]

    assert "Primary finding:" in report
    assert "Other findings:" in report

    primary_section, other_section = report.split("Other findings:")

    assert "validate_token" in primary_section
    assert "SECRET_KEY" in other_section


# ---------------------------------------------------------------------------
# 4 & 7. Contradictions
# ---------------------------------------------------------------------------

def test_contradictory_explanations_are_reported_not_resolved():

    result = synthesize(
        [
            child(
                "The expiry check in validate_token is wrong because of "
                "operator precedence."
            ),
            child(
                "The expiry check in validate_token is wrong because the "
                "timestamp units are mismatched."
            ),
        ]
    )

    assert result.metadata[CONFLICTING_COUNT_KEY] == 1

    finding = findings_of(result)[0]

    assert finding["status"] == CONFLICTING
    assert finding["confidence"] == "low"
    assert "disagree" in finding["conflict"]

    report = result.metadata[REPORT_KEY]

    assert "Conflicting analysis:" in report

    # Certainty is never claimed over a disagreement.
    assert "Confidence: low" in report


def test_a_conflict_does_not_silently_pick_one_side():

    result = synthesize(
        [
            child(
                "The expiry check in validate_token is wrong because of "
                "operator precedence."
            ),
            child(
                "The expiry check in validate_token is wrong because the "
                "timestamp units are mismatched."
            ),
        ]
    )

    report = result.metadata[REPORT_KEY]

    # Both explanations survive into the answer.
    assert "operator precedence" in report
    assert "units are mismatched" in report


def test_agreement_on_the_same_cause_is_not_a_conflict():

    result = synthesize(
        [
            child("validate_token mixes millisecond and second units."),
            child("validate_token compares seconds against milliseconds."),
        ]
    )

    assert result.metadata[CONFLICTING_COUNT_KEY] == 0

    # Independent agreement, but neither child cited code: likely, not proven.
    assert findings_of(result)[0]["status"] == LIKELY
    assert findings_of(result)[0]["supporting_children"] == [1, 2]


# ---------------------------------------------------------------------------
# 5. Speculation is not evidence
# ---------------------------------------------------------------------------

def test_speculation_is_not_treated_as_confirmed():

    result = synthesize(
        [child("config.py probably contains a hardcoded secret.")]
    )

    finding = findings_of(result)[0]

    assert finding["kind"] == HYPOTHESIS
    assert finding["status"] == POSSIBLE
    assert finding["confidence"] == "low"


def test_a_stated_fact_outranks_a_guess_about_the_same_area():

    result = synthesize(
        [
            child("config.py might contain a hardcoded secret."),
            child("config.py contains SECRET_KEY = 'dev-secret-key'."),
        ]
    )

    primary = result.metadata[PRIMARY_KEY]

    assert "SECRET_KEY = 'dev-secret-key'" in primary


@pytest.mark.parametrize(
    "statement,expected",
    [
        ("config.py probably contains a hardcoded secret", HYPOTHESIS),
        ("It seems the token expires immediately", HYPOTHESIS),
        ("config.py contains SECRET_KEY = 'dev'", "fact"),
        ("I need to inspect config.py and database.py", FOLLOW_UP),
        ("Further investigation of auth.py is required", FOLLOW_UP),
    ],
)
def test_statement_kinds_are_classified(statement, expected):

    assert classify_kind(statement) == expected


# ---------------------------------------------------------------------------
# 6 & 10. Incomplete investigations
# ---------------------------------------------------------------------------

def test_a_follow_up_request_is_not_a_finding():

    result = synthesize([child("I need to inspect config.py and database.py.")])

    finding = findings_of(result)[0]

    assert finding["kind"] == FOLLOW_UP
    assert finding["status"] == UNSUPPORTED

    # Nothing was established, so nothing is presented as primary.
    assert result.metadata[PRIMARY_KEY] == ""

    assert "No finding was supported well enough" in result.metadata[REPORT_KEY]


def test_success_does_not_by_itself_make_a_claim_true():

    # A successful child that only asked for more work.
    result = synthesize(
        [child("Further investigation of the login flow is required.")]
    )

    assert result.success is True
    assert result.metadata[CONFIRMED_COUNT_KEY] == 0
    assert result.metadata[PRIMARY_KEY] == ""


def test_a_completed_finding_outranks_an_incomplete_one():

    result = synthesize(
        [
            child("I need to inspect database.py."),
            child("auth.py validate_token() returns an expired token."),
        ]
    )

    assert "validate_token" in result.metadata[PRIMARY_KEY]


# ---------------------------------------------------------------------------
# 7-8. Failed and rate-limited children contribute no evidence
# ---------------------------------------------------------------------------

def test_a_failed_child_contributes_no_evidence():

    result = synthesize(
        [
            child("auth.py validate_token() mis-converts the expiry."),
            failed(task="child two"),
        ]
    )

    for finding in findings_of(result):
        assert 2 not in finding["supporting_children"]

    assert result.metadata[SUCCESSFUL_KEY] == 1
    assert result.metadata[FAILED_KEY] == 1


def test_a_rate_limited_child_contributes_no_evidence():

    result = synthesize(
        [
            child("auth.py validate_token() mis-converts the expiry."),
            rate_limited(task="child two"),
            child("validate_token compares the timestamp incorrectly."),
        ]
    )

    supporters = findings_of(result)[0]["supporting_children"]

    assert 2 not in supporters
    assert supporters == [1, 3]


def test_partial_metadata_is_unchanged_by_evidence_synthesis():

    result = synthesize(
        [
            child("auth.py validate_token() mis-converts the expiry."),
            rate_limited(),
            rate_limited(),
        ]
    )

    assert result.metadata[PARTIAL_KEY] is True
    assert result.metadata[SUCCESSFUL_KEY] == 1
    assert result.metadata[FAILED_KEY] == 2
    assert result.metadata[RATE_LIMITED_KEY] == 2

    # 2/3 never becomes 3/3.
    assert result.metadata["children"] == 3
    assert len(result.metadata["failures"]) == 2


def test_a_missing_child_is_not_counted_as_agreement():

    single = synthesize([child("The expiry comparison is incorrect.")])

    with_failures = synthesize(
        [child("The expiry comparison is incorrect."), failed(), rate_limited()]
    )

    assert (
        findings_of(single)[0]["supporting_children"]
        == findings_of(with_failures)[0]["supporting_children"]
    )


# ---------------------------------------------------------------------------
# 9. Primary bug selection
# ---------------------------------------------------------------------------

def test_a_functional_defect_outranks_a_security_observation():

    result = synthesize(
        [
            child("SECRET_KEY is hardcoded in config.py."),
            child(
                "auth.py validate_token() compares milliseconds against "
                "seconds, so tokens expire immediately."
            ),
        ]
    )

    assert "validate_token" in result.metadata[PRIMARY_KEY]

    report = result.metadata[REPORT_KEY]

    assert report.index("validate_token") < report.index("SECRET_KEY")


def test_a_functional_defect_outranks_a_style_observation():

    result = synthesize(
        [
            child("The auth module would benefit from a refactor for readability."),
            child("auth.py validate_token() returns a negative expiry."),
        ]
    )

    assert "validate_token" in result.metadata[PRIMARY_KEY]


@pytest.mark.parametrize(
    "statement,expected",
    [
        ("the expiry calculation is incorrect", FUNCTIONAL),
        ("SECRET_KEY is hardcoded", SECURITY),
        ("tokens expire immediately because units mismatch", FUNCTIONAL),
        ("passwords are stored in plaintext", SECURITY),
    ],
)
def test_categories_are_classified(statement, expected):

    assert classify_category(statement) == expected


# ---------------------------------------------------------------------------
# 10. Evidence preservation
# ---------------------------------------------------------------------------

def test_file_and_code_evidence_survives_synthesis():

    result = synthesize(
        [
            child(
                "In auth.py line 42, validate_token() computes "
                "expires_at = now + 3600000 which mixes units."
            )
        ]
    )

    evidence = findings_of(result)[0]["evidence"]

    joined = " ".join(evidence)

    assert "auth.py" in joined
    assert "line 42" in joined
    assert any("expires_at" in item for item in evidence)

    report = result.metadata[REPORT_KEY]

    assert "Evidence:" in report
    assert "auth.py" in report


def test_test_results_count_as_evidence():

    result = synthesize(
        [child("test_login fails with an assertion on the expiry value.")]
    )

    assert findings_of(result)[0]["evidence"]
    assert findings_of(result)[0]["status"] == CONFIRMED


def test_evidence_is_never_invented():

    statement = "Authentication is broken."

    assert collect_evidence(statement) == ()

    result = synthesize([child(statement)])

    assert findings_of(result)[0]["evidence"] == []


# ---------------------------------------------------------------------------
# 12-13. Degenerate inputs
# ---------------------------------------------------------------------------

def test_an_empty_successful_answer_is_handled():

    result = synthesize([child(""), child("   ")])

    assert result.success is True
    assert result.metadata[FINDINGS_COUNT_KEY] == 0
    assert result.metadata[PRIMARY_KEY] == ""
    assert result.metadata[REPORT_KEY] == ""


def test_all_children_failing_keeps_the_existing_failure_shape():

    result = synthesize([rate_limited(), rate_limited(), failed()])

    assert result.success is False
    assert result.answer == ""
    assert result.children_created == 3

    assert result.metadata[SUCCESSFUL_KEY] == 0
    assert result.metadata[RATE_LIMITED_KEY] == 2

    # No evidence keys are fabricated for a run that produced none.
    assert FINDINGS_KEY not in result.metadata


def test_no_results_keeps_the_existing_empty_behavior():

    result = synthesize([])

    assert result.success is False
    assert result.answer == ""
    assert result.metadata[SUCCESSFUL_KEY] == 0


# ---------------------------------------------------------------------------
# 14. Existing contract compatibility
# ---------------------------------------------------------------------------

def test_the_joined_answer_contract_is_unchanged():

    results = [
        child("First finding in auth.py."),
        failed(),
        child("Second finding in config.py."),
    ]

    result = synthesize(results)

    # RLMResult.answer is still exactly the successful answers, joined.
    assert result.answer == (
        "First finding in auth.py.\n\nSecond finding in config.py."
    )


def test_evidence_metadata_is_additive():

    result = synthesize([child("auth.py validate_token() is wrong.")])

    # Both the completeness metadata (Phase 6) and the evidence metadata
    # (Phase 8) are present together.
    for key in (PARTIAL_KEY, SUCCESSFUL_KEY, FAILED_KEY, RATE_LIMITED_KEY):
        assert key in result.metadata

    for key in (FINDINGS_KEY, FINDINGS_COUNT_KEY, PRIMARY_KEY, REPORT_KEY):
        assert key in result.metadata


# ---------------------------------------------------------------------------
# 15. Determinism
# ---------------------------------------------------------------------------

def test_synthesis_is_deterministic():

    def build():
        return [
            child("auth.py validate_token divides the timestamp incorrectly."),
            child("SECRET_KEY is hardcoded in config.py."),
            child("validate_token compares timestamp units incorrectly."),
            rate_limited(),
        ]

    first = synthesize(build())
    second = synthesize(build())

    assert first.answer == second.answer
    assert first.metadata[REPORT_KEY] == second.metadata[REPORT_KEY]
    assert first.metadata[FINDINGS_KEY] == second.metadata[FINDINGS_KEY]


def test_grouping_is_symmetric():

    assert are_related(
        "auth.py validate_token divides the timestamp incorrectly",
        "validate_token compares timestamp units incorrectly",
    ) is are_related(
        "validate_token compares timestamp units incorrectly",
        "auth.py validate_token divides the timestamp incorrectly",
    )


def test_unrelated_claims_are_not_related():

    assert not are_related(
        "SECRET_KEY is hardcoded in config.py",
        "validate_token compares timestamp units incorrectly",
    )


# ---------------------------------------------------------------------------
# End-to-end through the orchestrator
# ---------------------------------------------------------------------------

def test_the_orchestrator_returns_the_evidence_report():

    from rlm.budget import RLMBudget
    from rlm.nanocode_handler import NanoCodeCallHandler
    from rlm.orchestrator import RLMOrchestrator
    from rlm.runtime import RLMRuntime

    answers = [
        "auth.py validate_token() divides expires_at by 1000, so the expiry is wrong.",
        "validate_token compares timestamp units incorrectly.",
        "SECRET_KEY is hardcoded in config.py.",
    ]

    class ScriptedAgent:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: str) -> str:
            answer = answers[self.calls]
            self.calls += 1
            return answer

    agent = ScriptedAgent()

    orchestrator = RLMOrchestrator(
        runtime=RLMRuntime(
            call_handler=NanoCodeCallHandler(agent_factory=lambda: agent),
            budget=RLMBudget(max_depth=2, max_children=5, max_iterations=10),
        ),
    )

    answer = orchestrator.run("Find the authentication bug across the project.")

    assert answer.startswith("Primary finding:")
    assert "validate_token" in answer
    assert "Confidence: high" in answer

    # The security observation is reported, but not as the requested bug.
    assert "SECRET_KEY" in answer
    assert answer.index("validate_token") < answer.index("SECRET_KEY")

    metadata = orchestrator.last_result.metadata

    assert metadata[CONFIRMED_COUNT_KEY] >= 1
    assert metadata[PRIMARY_KEY]


def test_a_partial_run_keeps_its_notice_above_the_report():

    from rlm.budget import RLMBudget
    from rlm.nanocode_handler import NanoCodeCallHandler
    from rlm.orchestrator import RLMOrchestrator
    from rlm.runtime import RLMRuntime

    class HalfFailingAgent:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, task: str) -> str:
            self.calls += 1

            if self.calls == 1:
                return "auth.py validate_token() mis-converts the expiry timestamp."

            raise RuntimeError("child exploded")

    agent = HalfFailingAgent()

    orchestrator = RLMOrchestrator(
        runtime=RLMRuntime(
            call_handler=NanoCodeCallHandler(agent_factory=lambda: agent),
            budget=RLMBudget(max_depth=2, max_children=5, max_iterations=10),
        ),
    )

    answer = orchestrator.run("Find the authentication bug across the project.")

    assert answer.startswith("Partial investigation:")
    assert "1 of 3 child analyses completed successfully." in answer
    assert "Primary finding:" in answer
    assert "validate_token" in answer
