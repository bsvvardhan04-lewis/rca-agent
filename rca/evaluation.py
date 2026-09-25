"""Scoring an RCA report against recorded ground truth.

Pure functions, no I/O and no model calls, so the scoring rules themselves can
be unit tested. `scripts/evaluate.py` is the runner.

The checks are chosen to catch the failure that matters most: an agent that
always blames the most recent deploy. That heuristic scores perfectly on an
incident a deploy caused, and is actively harmful on the ones it did not - it
sends an engineer to revert innocent code while the real fault continues. So
`cause` and `decoys` are scored separately, and naming a decoy is a failure
even when the headline happens to be right.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rca.models import RCAReport

PR_PATTERN = re.compile(r"\b(PR-\d+|#\d{3,})\b", re.IGNORECASE)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    weight: float = 1.0


@dataclass
class ScoreCard:
    incident_id: str
    checks: list[Check] = field(default_factory=list)
    duration_seconds: float = 0.0
    tool_calls: int = 0
    tokens: int = 0
    error: str = ""

    def add(self, name: str, passed: bool, detail: str, weight: float = 1.0) -> None:
        self.checks.append(Check(name=name, passed=passed, detail=detail, weight=weight))

    @property
    def score(self) -> float:
        if self.error or not self.checks:
            return 0.0
        earned = sum(c.weight for c in self.checks if c.passed)
        total = sum(c.weight for c in self.checks)
        return earned / total if total else 0.0

    @property
    def passed(self) -> bool:
        """A run only counts as passing if it got the cause AND dismissed the decoys."""
        if self.error:
            return False
        critical = {"cause", "decoys"}
        return all(c.passed for c in self.checks if c.name in critical)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "score": round(self.score, 3),
            "passed": self.passed,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


def _haystack(report: RCAReport) -> str:
    """Everything the report says, lowercased, for substring checks."""
    a = report.analysis
    parts = [a.headline, a.blast_radius, *a.timeline, *a.immediate_actions, *a.evidence_gaps]
    for h in a.hypotheses:
        parts += [h.statement, h.reasoning, h.suspect_change, *h.verification_steps]
    return " ".join(parts).lower()


def _top_text(report: RCAReport) -> str:
    top = report.top_hypothesis
    if top is None:
        return report.analysis.headline.lower()
    return f"{report.analysis.headline} {top.statement} {top.reasoning}".lower()


def score_report(report: RCAReport, truth: dict[str, Any]) -> ScoreCard:
    card = ScoreCard(
        incident_id=report.incident.id,
        duration_seconds=report.duration_seconds,
        tool_calls=len(report.investigation_log),
        tokens=report.usage.total,
    )

    top = report.top_hypothesis
    if top is None:
        card.add("cause", False, "No hypothesis was produced at all.", weight=3.0)
        return card

    top_text = _top_text(report)
    everything = _haystack(report)
    root_cause = (truth.get("root_cause_id") or "").strip()
    expect_none = bool(truth.get("expect_no_recent_change"))

    # -- 1. the cause ------------------------------------------------------
    if expect_none:
        # The correct answer is "no deploy did this". Naming any PR as the
        # suspect is the exact failure mode we are testing for.
        named = PR_PATTERN.findall(f"{top.suspect_change} {top.statement}")
        blamed_code = top.category == "code_change" or bool(named)
        card.add(
            "cause",
            not blamed_code,
            "correctly did not blame a deploy"
            if not blamed_code
            else f"blamed {named or top.category} when no change was responsible",
            weight=3.0,
        )
    else:
        hit = root_cause.lower() in f"{top.suspect_change} {top.statement}".lower()
        card.add(
            "cause",
            hit,
            f"top hypothesis names {root_cause}"
            if hit
            else f"expected {root_cause}, got suspect_change={top.suspect_change!r}",
            weight=3.0,
        )

    # -- 2. the category ---------------------------------------------------
    expected_category = truth.get("category", "")
    card.add(
        "category",
        top.category == expected_category,
        f"{top.category} (expected {expected_category})",
    )

    # -- 3. the mechanism --------------------------------------------------
    must_mention = [m.lower() for m in truth.get("must_mention", [])]
    missing = [m for m in must_mention if m not in top_text]
    card.add(
        "mechanism",
        not missing,
        "explained" if not missing else f"never mentions {', '.join(missing)}",
        weight=2.0,
    )

    # -- 4. the decoys -----------------------------------------------------
    decoys = truth.get("must_rule_out", [])
    if decoys:
        # A decoy is handled if it is discussed and *not* the top suspect.
        mishandled = []
        for decoy in decoys:
            low = decoy.lower()
            if low in f"{top.suspect_change} {top.statement}".lower():
                mishandled.append(f"{decoy} named as the cause")
            elif low not in everything:
                mishandled.append(f"{decoy} never considered")
        card.add(
            "decoys",
            not mishandled,
            "decoys dismissed" if not mishandled else "; ".join(mishandled),
            weight=2.0,
        )

    # -- 5. calibration ----------------------------------------------------
    cause_check = next((c for c in card.checks if c.name == "cause"), None)
    got_it_right = bool(cause_check and cause_check.passed)
    overconfident = not got_it_right and top.confidence > 0.85
    card.add(
        "calibration",
        not overconfident,
        f"confidence {top.confidence:.0%}"
        + ("" if not overconfident else " while wrong - overconfident"),
    )

    # -- 6. alternatives ---------------------------------------------------
    card.add(
        "alternatives",
        len(report.analysis.hypotheses) >= 2,
        f"{len(report.analysis.hypotheses)} hypothesis(es)",
    )

    # -- 7. citations ------------------------------------------------------
    known = {item.id for item in report.evidence}
    cited = {
        eid
        for h in report.analysis.hypotheses
        for eid in h.supporting_evidence + h.contradicting_evidence
    }
    dangling = cited - known
    card.add(
        "citations",
        not dangling and bool(cited),
        "all resolve" if cited and not dangling else
        ("no citations at all" if not cited else f"dangling: {sorted(dangling)}"),
    )

    # -- 8. actionability --------------------------------------------------
    card.add(
        "actionable",
        bool(report.analysis.immediate_actions) and bool(top.verification_steps),
        f"{len(report.analysis.immediate_actions)} action(s), "
        f"{len(top.verification_steps)} verification step(s)",
    )

    return card


def summarise(cards: list[ScoreCard]) -> dict[str, Any]:
    if not cards:
        return {"runs": 0}
    by_check: dict[str, list[bool]] = {}
    for card in cards:
        for check in card.checks:
            by_check.setdefault(check.name, []).append(check.passed)

    return {
        "runs": len(cards),
        "passed": sum(1 for c in cards if c.passed),
        "mean_score": round(sum(c.score for c in cards) / len(cards), 3),
        "mean_duration": round(sum(c.duration_seconds for c in cards) / len(cards), 1),
        "total_tokens": sum(c.tokens for c in cards),
        "per_check": {
            name: f"{sum(results)}/{len(results)}" for name, results in by_check.items()
        },
    }
