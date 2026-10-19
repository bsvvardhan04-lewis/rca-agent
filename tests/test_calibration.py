"""The confidence guard.

It is a rule, not a judgement: it refuses "very likely" where the report's own
structure contradicts it, and says so. The two cases that motivated it came from
running a real 3B model on the same incident.
"""

from __future__ import annotations

from rca.calibration import CEILING, VERY_LIKELY, cap_overconfidence
from rca.models import Hypothesis, RCAAnalysis


def hypothesis(confidence=0.9, supporting=("L1",), contradicting=(), statement="Cause"):
    return Hypothesis(
        statement=statement,
        category="code_change",
        confidence=confidence,
        reasoning="Because.",
        supporting_evidence=list(supporting),
        contradicting_evidence=list(contradicting),
        suspect_change="",
        verification_steps=["Check it"],
    )


def report(*hypotheses: Hypothesis) -> RCAAnalysis:
    return RCAAnalysis(
        headline="h",
        timeline=[],
        hypotheses=list(hypotheses),
        blast_radius="b",
        immediate_actions=[],
        evidence_gaps=[],
    )


def test_a_lone_hypothesis_cannot_be_very_likely():
    """Live run, guided strategy: one hypothesis, 0.90, and it blamed a decoy PR.
    With nothing else considered there is nothing it was ranked against."""
    a = report(hypothesis(0.9))
    notes = cap_overconfidence(a)
    assert a.hypotheses[0].confidence == CEILING
    assert len(notes) == 1 and "no alternative" in notes[0]


def test_evidence_cited_against_a_hypothesis_forbids_very_likely():
    a = report(hypothesis(0.9, contradicting=["L3"]), hypothesis(0.3, statement="Other"))
    cap_overconfidence(a)
    assert a.hypotheses[0].confidence == CEILING
    assert a.hypotheses[1].confidence == 0.3, "only the overconfident one is touched"


def test_a_hypothesis_with_no_support_forbids_very_likely():
    a = report(hypothesis(0.95, supporting=()), hypothesis(0.2, statement="Other"))
    notes = cap_overconfidence(a)
    assert a.hypotheses[0].confidence == CEILING
    assert "no supporting evidence" in notes[0]


def test_a_well_founded_high_confidence_is_left_alone():
    """The guard must not turn every report into a hedge. Two hypotheses, the
    top one supported and uncontradicted: 0.9 stands."""
    a = report(hypothesis(0.9), hypothesis(0.2, statement="Other", contradicting=["L2"]))
    assert cap_overconfidence(a) == []
    assert a.hypotheses[0].confidence == 0.9


def test_the_reasons_are_all_reported_and_written_into_the_reasoning():
    a = report(hypothesis(0.92, supporting=(), contradicting=["L1"]))
    note = cap_overconfidence(a)[0]
    for reason in ("no alternative", "no supporting evidence", "against it"):
        assert reason in note
    assert note in a.hypotheses[0].reasoning, "the reader of the report must see why"


def test_confidence_at_or_below_the_line_is_never_touched():
    a = report(hypothesis(VERY_LIKELY, supporting=(), contradicting=["L1"]))
    assert cap_overconfidence(a) == []
    assert a.hypotheses[0].confidence == VERY_LIKELY


def test_an_empty_analysis_is_fine():
    assert cap_overconfidence(report()) == []


def test_the_ceiling_is_below_the_very_likely_line():
    """Otherwise capping would not change the band the reader sees."""
    assert CEILING <= VERY_LIKELY
