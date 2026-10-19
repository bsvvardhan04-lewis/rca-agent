"""A guard on stated confidence.

The synthesis prompt tells the model to reserve confidence above 0.85 for cases
where the mechanism is spelled out and an alternative was actively ruled out. A
3B model ignores that. Two live runs on the same incident both came back at 90%:
one correct but vague, one *wrong* - it blamed a decoy pull request, as the only
hypothesis, with evidence cited against it.

A prompt is a request. This is a rule. It does not make the model right and it
cannot judge whether a mechanism is well explained; it only refuses "very
likely" in the situations where the report's own structure contradicts it. A
confident wrong answer is the worst thing an incident tool can show someone
under time pressure, and these three conditions are checkable without a model.
"""

from __future__ import annotations

from rca.models import RCAAnalysis

VERY_LIKELY = 0.85
CEILING = 0.70


def cap_overconfidence(analysis: RCAAnalysis) -> list[str]:
    """Lower any hypothesis rated above 0.85 whose own evidence does not earn it.

    Mutates `analysis` and returns one human-readable note per hypothesis
    changed (empty if none was). The note is also appended to that hypothesis's
    reasoning, so the reader of the report sees why the number is what it is.
    """
    notes: list[str] = []
    for hypothesis in analysis.hypotheses:
        if hypothesis.confidence <= VERY_LIKELY:
            continue

        reasons: list[str] = []
        if len(analysis.hypotheses) < 2:
            reasons.append("no alternative explanation was considered")
        if not hypothesis.supporting_evidence:
            reasons.append("it cites no supporting evidence")
        if hypothesis.contradicting_evidence:
            reasons.append("evidence was cited against it")
        if not reasons:
            continue

        before = hypothesis.confidence
        hypothesis.confidence = CEILING
        note = (
            f"Confidence lowered from {before:.2f} to {CEILING:.2f} automatically: "
            + "; ".join(reasons)
            + "."
        )
        hypothesis.reasoning = f"{hypothesis.reasoning} [{note}]"
        notes.append(note)
    return notes
