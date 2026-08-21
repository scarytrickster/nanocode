"""Deterministic evidence model for RLM synthesis.

Concatenating child answers presents a speculation and a verified defect as
equally true. This module turns child answers into structured findings so
synthesis can say *how well supported* each claim is, which children agreed,
and where they contradicted each other.

Everything here is deterministic string analysis: no LLM, no embeddings, no
external dependency. Evidence is always a substring of what a child actually
wrote -- nothing is invented, and a claim is never promoted beyond what its
own wording supports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Statuses and confidence
# ---------------------------------------------------------------------------

CONFIRMED = "confirmed"
LIKELY = "likely"
POSSIBLE = "possible"
CONFLICTING = "conflicting"
UNSUPPORTED = "unsupported"

HIGH = "high"
MEDIUM = "medium"
LOW = "low"

CONFIDENCE_BY_STATUS = {
    CONFIRMED: HIGH,
    LIKELY: MEDIUM,
    CONFLICTING: LOW,
    POSSIBLE: LOW,
    UNSUPPORTED: LOW,
}

# Statuses strong enough to headline an answer.
REPORTABLE_STATUSES = (CONFIRMED, LIKELY, CONFLICTING)

# What kind of statement a child made.
FACT = "fact"
HYPOTHESIS = "hypothesis"
FOLLOW_UP = "follow_up"

# What the claim is about, used to rank a functional defect above an
# incidental security or style observation when both are reported.
FUNCTIONAL = "functional"
SECURITY = "security"
QUALITY = "quality"
OBSERVATION = "observation"

CATEGORY_RANK = {FUNCTIONAL: 0, SECURITY: 1, QUALITY: 2, OBSERVATION: 3}
STATUS_RANK = {CONFIRMED: 0, CONFLICTING: 1, LIKELY: 2, POSSIBLE: 3, UNSUPPORTED: 4}


# ---------------------------------------------------------------------------
# Lexicons. Small and explicit on purpose: this is string analysis, not NLP.
# ---------------------------------------------------------------------------

SPECULATION_MARKERS = (
    "probably", "might", "maybe", "may be", "could be", "possibly", "perhaps",
    "seems", "appears", "suspect", "suspected", "likely", "unclear", "not sure",
    "i think", "presumably", "potentially",
)

FOLLOW_UP_MARKERS = (
    "need to inspect", "needs inspection", "need to check", "needs to be checked",
    "should be inspected", "should be reviewed", "further investigation",
    "requires further", "next step", "i would check", "to be confirmed",
    "unable to", "could not read", "did not inspect", "worth inspecting",
)

FUNCTIONAL_MARKERS = (
    "incorrect", "wrong", "mismatch", "mismatched", "bug", "defect", "fails",
    "failing", "failure", "breaks", "broken", "negative", "off-by-one",
    "off by one", "expiry", "expiration", "returns", "exception", "crash",
    "always false", "always true", "never expires", "immediately expired",
    "divides", "division", "compares", "comparison", "conversion", "units",
    "regression", "does not work", "doesn't work",
)

SECURITY_MARKERS = (
    "hardcoded", "hard-coded", "secret", "password", "plaintext", "plain text",
    "credential", "credentials", "insecure", "unsafe", "injection", "leak",
    "exposed", "unencrypted", "weak hash",
)

QUALITY_MARKERS = (
    "refactor", "naming", "duplication", "duplicated", "style", "readability",
    "best practice", "docstring", "type hint", "cleanup", "maintainability",
)

# Distinct explanations for the same symptom. Two children citing different
# causes for one issue are disagreeing, not adding up.
CAUSE_TERMS: dict[str, tuple[str, ...]] = {
    "precedence": ("operator precedence", "precedence", "parenthes"),
    "units": ("unit", "units", "millisecond", "milliseconds", "seconds", "conversion"),
    "timezone": ("timezone", "time zone", "utc", "localtime"),
    "off_by_one": ("off-by-one", "off by one", "boundary"),
    "type": ("type error", "wrong type", "string instead", "int instead", "cast"),
    "ordering": ("order of", "ordering", "sequence", "race"),
    "missing": ("missing check", "not validated", "no validation", "absent"),
}

NEGATION_MARKERS = (
    "is correct", "is not the cause", "not the bug", "no issue", "works correctly",
    "is fine", "is not broken",
)

STOPWORDS = frozenset(
    """
    the a an and or but if then that this these those there here is are was were
    be been being it its it's of in on at to from for with without by as into
    than when while which who whom what where why how not no yes can could should
    would may might must will shall do does did done has have had having i we you
    they he she them us our your their his her about after before between during
    over under again further more most other some such only own same so too very
    also because both each few nor own too via across per within
    """.split()
)

# A word that carries identity: a file name, a symbol, a snake_case name.
FILE_PATTERN = re.compile(
    r"\b[\w./\\-]*\w\.(?:py|js|jsx|ts|tsx|json|toml|cfg|ini|yaml|yml|md|txt|sql|env)\b"
)
SYMBOL_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\(\)|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b|\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
LINE_PATTERN = re.compile(r"\bline\s+\d+\b|\b\w+\.\w+:\d+\b")
CODE_PATTERN = re.compile(r"`[^`]+`|\b[\w.]+\s*=\s*[^\s,;]+|\b[A-Za-z_]\w*\([^)]*\)")
TEST_PATTERN = re.compile(
    r"\btest[s]?\s+(?:fail|failed|pass|passed)\b|\bassert\w*\b|\btest_\w+\b",
    re.IGNORECASE,
)

STATEMENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


@dataclass(frozen=True)
class Finding:
    """One claim made by one child, with the evidence that child gave for it."""

    claim: str
    child: int
    kind: str = FACT
    category: str = OBSERVATION
    evidence: tuple[str, ...] = ()
    task: str = ""

    @property
    def has_concrete_evidence(self) -> bool:
        """True when the child pointed at something, not just asserted it."""

        return bool(self.evidence)


@dataclass
class FindingGroup:
    """Findings from one or more children that describe the same issue."""

    findings: list[Finding] = field(default_factory=list)
    status: str = UNSUPPORTED
    confidence: str = LOW
    conflict_reason: str = ""

    @property
    def claim(self) -> str:
        """The best-supported wording of the shared claim."""

        return self.best_finding.claim

    @property
    def best_finding(self) -> Finding:
        """The member with the strongest support, deterministically chosen."""

        return sorted(
            self.findings,
            key=lambda finding: (
                0 if finding.kind == FACT else 1,
                0 if finding.has_concrete_evidence else 1,
                -len(finding.evidence),
                finding.child,
            ),
        )[0]

    @property
    def category(self) -> str:
        return min(
            (finding.category for finding in self.findings),
            key=lambda category: CATEGORY_RANK[category],
        )

    @property
    def supporting_children(self) -> tuple[int, ...]:
        return tuple(sorted({finding.child for finding in self.findings}))

    @property
    def evidence(self) -> tuple[str, ...]:
        seen: list[str] = []

        for finding in self.findings:
            for item in finding.evidence:
                if item not in seen:
                    seen.append(item)

        return tuple(seen)

    @property
    def statements(self) -> tuple[tuple[int, str], ...]:
        """Every distinct wording in this group, with the child that wrote it.

        Grouping must not discard what a child actually said: the group has one
        headline claim, but each child's own phrasing is preserved.
        """

        seen: list[tuple[int, str]] = []
        claims: set[str] = set()

        for finding in self.findings:
            if finding.claim in claims:
                continue

            claims.add(finding.claim)
            seen.append((finding.child, finding.claim))

        return tuple(seen)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "status": self.status,
            "confidence": self.confidence,
            "category": self.category,
            "kind": self.best_finding.kind,
            "supporting_children": list(self.supporting_children),
            "evidence": list(self.evidence),
            "conflict": self.conflict_reason,
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _contains(text: str, markers) -> bool:
    lowered = text.lower()

    return any(marker in lowered for marker in markers)


def collect_evidence(statement: str) -> tuple[str, ...]:
    """Concrete references the statement itself contains.

    Only substrings of the child's own text are returned, so evidence can
    never be stronger than what the child actually wrote.
    """

    evidence: list[str] = []

    for pattern in (CODE_PATTERN, LINE_PATTERN, FILE_PATTERN, TEST_PATTERN):
        for match in pattern.findall(statement):
            item = match.strip().strip("`").strip()

            if item and item not in evidence:
                evidence.append(item)

    return tuple(evidence)


def classify_kind(statement: str) -> str:
    """Whether the child stated a fact, guessed, or asked for more work."""

    if _contains(statement, FOLLOW_UP_MARKERS):
        return FOLLOW_UP

    if _contains(statement, SPECULATION_MARKERS):
        return HYPOTHESIS

    return FACT


def classify_category(statement: str) -> str:
    """What the claim is about."""

    if _contains(statement, FUNCTIONAL_MARKERS):
        return FUNCTIONAL

    if _contains(statement, SECURITY_MARKERS):
        return SECURITY

    if _contains(statement, QUALITY_MARKERS):
        return QUALITY

    return OBSERVATION


def split_statements(answer: str) -> list[str]:
    """Split one child answer into individual claims, punctuation preserved."""

    statements = []

    for raw in STATEMENT_SPLIT.split(answer or ""):
        statement = raw.strip().lstrip("-*•").strip()

        if statement:
            statements.append(statement)

    return statements


def extract_findings(answer: str, child: int, task: str = "") -> list[Finding]:
    """Turn one child's answer into findings."""

    return [
        Finding(
            claim=statement,
            child=child,
            kind=classify_kind(statement),
            category=classify_category(statement),
            evidence=collect_evidence(statement),
            task=task,
        )
        for statement in split_statements(answer)
    ]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def significant_tokens(text: str) -> frozenset[str]:
    return frozenset(
        word.lower()
        for word in WORD_PATTERN.findall(text)
        if len(word) > 2 and word.lower() not in STOPWORDS
    )


def file_tokens(text: str) -> frozenset[str]:
    """File names mentioned in a claim."""

    return frozenset(match.lower() for match in FILE_PATTERN.findall(text))


def symbol_tokens(text: str) -> frozenset[str]:
    """Function and constant names: the strongest handle on "the same thing"."""

    return frozenset(
        match.lower().rstrip("()")
        for match in SYMBOL_PATTERN.findall(text)
        if match
    )


def identity_tokens(text: str) -> frozenset[str]:
    """File names and symbols: the words that pin a claim to real code."""

    return file_tokens(text) | symbol_tokens(text)


def are_related(first: str, second: str) -> bool:
    """True when two claims describe the same underlying issue.

    Three deterministic signals, in decreasing strength:
      - they name the same symbol (two children discussing validate_token are
        discussing the same function),
      - they name the same file and share wording,
      - their wording overlaps heavily.
    """

    first_words = significant_tokens(first)
    second_words = significant_tokens(second)

    if not first_words or not second_words:
        return False

    if symbol_tokens(first) & symbol_tokens(second):
        return True

    shared_words = first_words & second_words

    if file_tokens(first) & file_tokens(second) and len(shared_words) >= 2:
        return True

    union = first_words | second_words

    return len(shared_words) / len(union) >= 0.5


def group_findings(findings: list[Finding]) -> list[FindingGroup]:
    """Collapse findings that describe the same issue into one group."""

    groups: list[FindingGroup] = []

    for finding in findings:
        for group in groups:
            if any(
                are_related(finding.claim, member.claim)
                for member in group.findings
            ):
                group.findings.append(finding)
                break
        else:
            groups.append(FindingGroup(findings=[finding]))

    for group in groups:
        _score_group(group)

    return groups


# ---------------------------------------------------------------------------
# Agreement, conflict and confidence
# ---------------------------------------------------------------------------

def cause_categories(text: str) -> frozenset[str]:
    lowered = text.lower()

    return frozenset(
        name
        for name, terms in CAUSE_TERMS.items()
        if any(term in lowered for term in terms)
    )


def detect_conflict(group: FindingGroup) -> str:
    """Describe a disagreement inside a group, or return an empty string.

    Only children that actually asserted something can disagree: a hypothesis
    or a follow-up request is not a competing explanation.
    """

    asserted = [
        finding for finding in group.findings if finding.kind == FACT
    ]

    if len({finding.child for finding in asserted}) < 2:
        return ""

    causes = {
        finding.child: cause_categories(finding.claim)
        for finding in asserted
    }

    named = {child: cause for child, cause in causes.items() if cause}

    if len(named) >= 2:
        distinct = set()

        for cause in named.values():
            distinct |= cause

        # Disjoint explanations: no child named a cause any other child named.
        if len(distinct) > 1 and not set.intersection(*(set(c) for c in named.values())):
            return (
                "children disagree on the cause: "
                + ", ".join(sorted(distinct))
            )

    contradicted = [
        finding for finding in asserted if _contains(finding.claim, NEGATION_MARKERS)
    ]

    if contradicted and len(contradicted) < len(asserted):
        return "one child reports no issue where another reports a defect"

    return ""


def _score_group(group: FindingGroup) -> None:
    """Assign the group's status and confidence from its members' support."""

    conflict = detect_conflict(group)

    if conflict:
        group.conflict_reason = conflict
        group.status = CONFLICTING
        group.confidence = CONFIDENCE_BY_STATUS[CONFLICTING]
        return

    facts = [finding for finding in group.findings if finding.kind == FACT]
    supporters = len({finding.child for finding in facts})

    with_evidence = [finding for finding in facts if finding.has_concrete_evidence]

    if with_evidence:
        # The child pointed at real code, a line, or a test result.
        group.status = CONFIRMED

    elif supporters >= 2:
        # Independent agreement is worth something, but without evidence it is
        # not a confirmation.
        group.status = LIKELY

    elif facts:
        # One child asserting something with nothing to back it up.
        group.status = UNSUPPORTED

    elif any(finding.kind == HYPOTHESIS for finding in group.findings):
        # Speculation stays speculation, however concrete its wording.
        group.status = POSSIBLE

    else:
        group.status = UNSUPPORTED

    group.confidence = CONFIDENCE_BY_STATUS[group.status]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_groups(groups: list[FindingGroup]) -> list[FindingGroup]:
    """Order findings so the best-supported functional defect comes first."""

    return sorted(
        groups,
        key=lambda group: (
            CATEGORY_RANK[group.category],
            STATUS_RANK[group.status],
            -len(group.supporting_children),
            -len(group.evidence),
            group.claim.lower(),
        ),
    )


def select_primary(groups: list[FindingGroup]) -> FindingGroup | None:
    """The finding that best explains the reported problem, if any does.

    A run that produced only speculation and follow-up requests has no primary
    finding: saying so is more useful than promoting a guess.
    """

    for group in groups:
        if group.status in REPORTABLE_STATUSES:
            return group

    return None


def analyze(answers: list[tuple[int, str, str]]) -> tuple[list[FindingGroup], FindingGroup | None]:
    """Analyze successful child answers.

    `answers` is (child_index, answer, task) for successful children only --
    a failed child contributes no evidence.
    """

    findings: list[Finding] = []

    for child, answer, task in answers:
        findings.extend(extract_findings(answer, child=child, task=task))

    groups = rank_groups(group_findings(findings))

    return groups, select_primary(groups)
