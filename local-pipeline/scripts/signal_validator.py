#!/usr/bin/env python3
"""SelfTrace deterministic validation helpers v1.8.

Why this file exists
--------------------
The VLM/LLM is good at semantic understanding, but final profile evidence needs a
more deterministic last gate.  This module validates and calibrates model output
without asking another model to "judge the judge".

The key rule is: classify a signal primarily from its *claim*, not from keywords
that happen to appear in its evidence.  This prevents mistakes such as an
EXPRESSION_STYLE claim becoming INTEREST merely because its evidence contains
"fascinated".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable


VALID_DIMENSIONS = {
    "INTEREST",
    "KNOWLEDGE",
    "ACTIVITY",
    "THINKING_PATTERN",
    "EXPRESSION_STYLE",
    "GOAL_ORIENTATION",
}

# Patterns are deliberately conservative.  They target obvious semantic cues,
# not every possible phrase the model could generate.
DIMENSION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    (
        "EXPRESSION_STYLE",
        (
            r"\bcommunicat(?:e|es|ion)\b",
            r"\bexpression style\b",
            r"\bconcise(?:ly)?\b",
            r"\bnote[- ]like\b",
            r"\bbilingual\b",
            r"\bquestion[- ]driven\b",
            r"\bterse\b",
            r"\bphrasing\b",
            r"\bwording\b",
            r"\bwrites? in\b",
            r"\buses? bullet",
        ),
    ),
    (
        "GOAL_ORIENTATION",
        (
            r"\bintend(?:s|ed|tion)?\b",
            r"\bplans? to\b",
            r"\bfuture intention\b",
            r"\bfuture plan\b",
            r"\bgoal\b",
            r"\bwould like to\b",
            r"\bwants? to\b",
            r"\baims? to\b",
            r"\bhope(?:s)? to\b",
            r"\bsystematically study\b",
        ),
    ),
    (
        "INTEREST",
        (
            r"\bfascinat(?:ed|ion|ing)\b",
            r"\binterested in\b",
            r"\binterest in\b",
            r"\bcurios(?:ity|ous)\b",
            r"\bappeal(?:s|ing)?\b",
            r"\bdrawn to\b",
            r"\bpreference for\b",
        ),
    ),
    (
        "THINKING_PATTERN",
        (
            r"\bcompar(?:e|es|ing|ison)\b",
            r"\babstract(?:s|ed|ion|ing)?\b",
            r"\banalog(?:y|ical)\b",
            r"\bsame core\b",
            r"\bshared core\b",
            r"\bcommon pattern\b",
            r"\bpattern[- ]link",
            r"\bmechanism[- ]seek",
            r"\bconnects? .* concepts\b",
            r"\btransfers? .* insight\b",
        ),
    ),
    (
        "ACTIVITY",
        (
            r"\bimplement(?:ed|ing|s)?\b",
            r"\bexperiment(?:ed|ing|s)?\b",
            r"\bstud(?:y|ied|ying)\b",
            r"\bread(?:ing)?\b",
            r"\blearn(?:ed|ing|s)?\b",
            r"\btested?\b",
            r"\bworking on\b",
            r"\bbuilt\b",
            r"\bruns? experiments?\b",
            r"\bdocument(?:ed|ing|s)?\b",
        ),
    ),
    (
        "KNOWLEDGE",
        (
            r"\bdemonstrates? understanding\b",
            r"\bshows? understanding\b",
            r"\bunderstands?\b",
            r"\bexplains?\b",
            r"\bdistinguishes?\b",
            r"\bidentifies?\b",
            r"\btechnical knowledge\b",
        ),
    ),
]

MEDIA_ONLY_PATTERNS = (
    r"\bincluded media\b",
    r"\battached media\b",
    r"\battached (?:an )?image\b",
    r"\bincluded (?:an )?image\b",
    r"\bmedia showing\b",
    r"\bimage showing\b",
    r"\bscreenshot showing\b",
)

# These are profile dimensions SelfTrace intentionally does not infer.  A post
# may of course contain these words; the validator only rejects *inference
# claims* that try to assign such a sensitive attribute/state to the author.
SENSITIVE_INFERENCE_PATTERNS = (
    r"\bthe author (?:is|has|suffers from) (?:depressed|anxious|bipolar|autistic)\b",
    r"\bpolitical (?:belief|ideology|orientation)\b",
    r"\breligious (?:belief|identity|affiliation)\b",
    r"\bsexual orientation\b",
    r"\bethnic(?:ity| identity)\b",
    r"\bdiagnos(?:is|ed)\b",
    r"\bromantic (?:interest|feeling|relationship)\b",
)


@dataclass
class ValidationEvent:
    """One deterministic validator action, recorded for auditability."""

    action: str
    reason: str
    claim: str
    before_dimension: str | None = None
    after_dimension: str | None = None

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "claim": self.claim,
            "before_dimension": self.before_dimension,
            "after_dimension": self.after_dimension,
        }


@dataclass
class ValidationReport:
    """Audit report returned alongside the final signals."""

    input_count: int = 0
    kept_count: int = 0
    corrected_count: int = 0
    filtered_count: int = 0
    duplicate_count: int = 0
    events: list[ValidationEvent] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "input_count": self.input_count,
            "kept_count": self.kept_count,
            "corrected_count": self.corrected_count,
            "filtered_count": self.filtered_count,
            "duplicate_count": self.duplicate_count,
            "events": [event.as_dict() for event in self.events],
        }


def normalize_text(text: str) -> str:
    """Normalize a claim for duplicate comparison."""

    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def classify_claim_dimension(claim: str) -> str | None:
    """Return an obvious dimension from the CLAIM ONLY.

    Evidence is deliberately excluded.  Evidence may contain topic words that
    are irrelevant to the semantic type of the claim.
    """

    text = claim.lower()
    for dimension, patterns in DIMENSION_PATTERNS:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return dimension
    return None


def is_media_only_claim(claim: str) -> bool:
    text = claim.lower()
    return any(re.search(pattern, text) for pattern in MEDIA_ONLY_PATTERNS)


def is_sensitive_inference(claim: str) -> bool:
    text = claim.lower()
    return any(re.search(pattern, text) for pattern in SENSITIVE_INFERENCE_PATTERNS)


def claims_are_near_duplicates(a: str, b: str) -> bool:
    """Conservative fuzzy duplicate test."""

    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.88


def validate_signal_dicts(signals: Iterable[dict], max_signals: int = 4) -> tuple[list[dict], ValidationReport, list[str]]:
    """Validate signal dictionaries and return (kept, report, rejected_reasons)."""

    raw = list(signals)
    report = ValidationReport(input_count=len(raw))
    kept: list[dict] = []
    rejected: list[str] = []

    for source in raw:
        signal = dict(source)
        claim = str(signal.get("claim") or "").strip()
        evidence = [str(x).strip() for x in signal.get("evidence", []) if str(x).strip()]
        dimension = str(signal.get("dimension") or "").strip()

        if not claim:
            reason = "Signal filtered: empty claim"
            report.filtered_count += 1
            report.events.append(ValidationEvent("FILTER", reason, claim))
            rejected.append(reason)
            continue

        if not evidence:
            reason = f"Signal filtered: no traceable evidence | {claim}"
            report.filtered_count += 1
            report.events.append(ValidationEvent("FILTER", reason, claim, dimension, None))
            rejected.append(reason)
            continue

        if is_media_only_claim(claim):
            reason = f"Signal filtered: media-description-only claim | {claim}"
            report.filtered_count += 1
            report.events.append(ValidationEvent("FILTER", reason, claim, dimension, None))
            rejected.append(reason)
            continue

        if is_sensitive_inference(claim):
            reason = f"Signal filtered: disallowed sensitive inference | {claim}"
            report.filtered_count += 1
            report.events.append(ValidationEvent("FILTER", reason, claim, dimension, None))
            rejected.append(reason)
            continue

        # Correct only when the claim itself contains an obvious cue.
        inferred = classify_claim_dimension(claim)
        if inferred and inferred != dimension:
            report.corrected_count += 1
            report.events.append(ValidationEvent(
                "CORRECT_DIMENSION",
                "Claim semantics map clearly to another dimension",
                claim,
                dimension,
                inferred,
            ))
            dimension = inferred

        if dimension not in VALID_DIMENSIONS:
            reason = f"Signal filtered: invalid dimension '{dimension}' | {claim}"
            report.filtered_count += 1
            report.events.append(ValidationEvent("FILTER", reason, claim, dimension, None))
            rejected.append(reason)
            continue

        # Keep the scope narrow even if a future model returns something else.
        signal["dimension"] = dimension
        signal["claim"] = claim
        signal["evidence"] = evidence
        signal["temporal_scope"] = "THIS_POST_ONLY"
        if signal.get("confidence") not in {"HIGH", "MEDIUM"}:
            signal["confidence"] = "MEDIUM"

        duplicate_of = next(
            (existing for existing in kept if existing["dimension"] == dimension and claims_are_near_duplicates(existing["claim"], claim)),
            None,
        )
        if duplicate_of:
            reason = f"Signal filtered: near duplicate of '{duplicate_of['claim']}' | {claim}"
            report.duplicate_count += 1
            report.events.append(ValidationEvent("DEDUP", reason, claim, dimension, dimension))
            rejected.append(reason)
            continue

        kept.append(signal)

    # Strong explicit signals first, then preserve generation order.
    kept.sort(key=lambda x: 0 if x.get("confidence") == "HIGH" else 1)
    if len(kept) > max_signals:
        extras = kept[max_signals:]
        kept = kept[:max_signals]
        for extra in extras:
            reason = f"Signal filtered: exceeds per-post max {max_signals} | {extra['claim']}"
            report.filtered_count += 1
            report.events.append(ValidationEvent("FILTER", reason, extra["claim"], extra["dimension"], None))
            rejected.append(reason)

    report.kept_count = len(kept)
    return kept, report, rejected


def normalize_post_intents(raw_text: str, intents: list[str]) -> tuple[list[str], list[dict]]:
    """Apply a few conservative intent corrections using explicit author text."""

    text = (raw_text or "").lower()
    result = list(dict.fromkeys(intents))
    events: list[dict] = []

    # A future learning intention is better represented with LEARN/RECORD_IDEA
    # than SHARE_RESULT unless the post actually reports a result.
    future_learning = bool(re.search(r"would like to study|want to study|plan to study|learn .* future|study it systematically", text))
    result_language = bool(re.search(r"result|achieved|completed|experiment (?:shows|showed)|we found|i found", text))

    if future_learning and "LEARN" not in result:
        result.append("LEARN")
        events.append({"action": "ADD_INTENT", "intent": "LEARN", "reason": "Explicit future learning intention"})

    if future_learning and "SHARE_RESULT" in result and not result_language:
        result.remove("SHARE_RESULT")
        events.append({"action": "REMOVE_INTENT", "intent": "SHARE_RESULT", "reason": "No explicit result/report language"})

    if not result:
        result = ["OTHER"]
    return result, events
