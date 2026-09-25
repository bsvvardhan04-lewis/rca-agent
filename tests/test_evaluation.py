"""Tests for the scorer.

An eval you have not tested is just a number that makes you feel good. These
pin down the behaviour that matters most: a report that blames the newest PR
when no deploy was responsible must FAIL, even though it looks confident and
well-cited.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rca.evaluation import score_report, summarise
from rca.models import (
    EvidenceItem,
    Hypothesis,
    IncidentReport,
    RCAAnalysis,
    RCAReport,
    TriagedIncident,
    iso,
)
from tests.conftest import ANCHOR


def make_report(
    *,
    statement="Risk-scoring call inside tx_authorize exhausts the connection pool.",
    category="code_change",
    confidence=0.9,
    suspect="PR-4821",
    reasoning="The deploy precedes onset and the transaction holds a pooled connection.",
    supporting=("L1", "C1"),
    extra_hypotheses=(),
    headline="PR-4821 put a network call inside the payments DB transaction.",
    actions=("Roll back PR-4821",),
    verification=("Check pool utilisation",),
    evidence_ids=("L1", "C1", "A1"),
    gaps=(),
) -> RCAReport:
    hypotheses = [
        Hypothesis(
            statement=statement,
            category=category,
            confidence=confidence,
            reasoning=reasoning,
            supporting_evidence=list(supporting),
            contradicting_evidence=[],
            suspect_change=suspect,
            verification_steps=list(verification),
        ),
        *extra_hypotheses,
    ]
    return RCAReport(
        incident=IncidentReport(
            id="INC-1042", title="t", description="d", reported_at=ANCHOR
        ),
        triage=TriagedIncident(
            summary="s",
            affected_services=["payments-api"],
            symptoms=["x"],
            error_signatures=["500"],
            search_keywords=["pay"],
            window_start=iso(ANCHOR - timedelta(hours=3)),
            window_end=iso(ANCHOR),
            severity="SEV2",
            triage_notes="n",
        ),
        analysis=RCAAnalysis(
            headline=headline,
            timeline=["13:38 - deploy"],
            hypotheses=hypotheses,
            blast_radius="some checkout traffic",
            immediate_actions=list(actions),
            evidence_gaps=list(gaps),
        ),
        evidence=[
            EvidenceItem(id=eid, kind="log", summary=f"{eid} detail", source="s")
            for eid in evidence_ids
        ],
        investigation_log=["log_volume(...)", "search_logs(...)"],
    )


CODE_TRUTH = {
    "category": "code_change",
    "root_cause_id": "PR-4821",
    "expect_no_recent_change": False,
    "must_mention": ["pool", "transaction"],
    "must_rule_out": ["PR-4817"],
}

NO_CHANGE_TRUTH = {
    "category": "external_provider",
    "root_cause_id": "",
    "expect_no_recent_change": True,
    "must_mention": ["acquirer", "503"],
    "must_rule_out": ["PR-4901"],
}


def check(card, name):
    return next(c for c in card.checks if c.name == name)


# -- the deploy-caused case ----------------------------------------------


def test_a_correct_report_passes():
    report = make_report(
        extra_hypotheses=[
            Hypothesis(
                statement="The festive banner PR-4817 changed checkout.",
                category="code_change",
                confidence=0.1,
                reasoning="Ruled out: it is frontend-only and deployed 110 minutes earlier.",
                supporting_evidence=[],
                contradicting_evidence=["C1"],
                suspect_change="",
                verification_steps=["none needed"],
            )
        ]
    )
    card = score_report(report, CODE_TRUTH)
    assert card.passed
    assert card.score == 1.0


def test_naming_the_wrong_change_fails_the_cause_check():
    card = score_report(make_report(suspect="PR-4817", statement="The banner PR broke it."),
                        CODE_TRUTH)
    assert not check(card, "cause").passed
    assert not card.passed


def test_missing_the_mechanism_is_caught_even_when_the_id_is_right():
    card = score_report(
        make_report(
            statement="PR-4821 caused the outage.",
            reasoning="It was deployed recently and the timing lines up.",
        ),
        CODE_TRUTH,
    )
    assert check(card, "cause").passed
    assert not check(card, "mechanism").passed
    assert "pool" in check(card, "mechanism").detail


def test_a_decoy_never_considered_fails_the_decoy_check():
    # PR-4817 is nowhere in the report - the agent never weighed it.
    card = score_report(make_report(), CODE_TRUTH)
    assert not check(card, "decoys").passed
    assert "never considered" in check(card, "decoys").detail
    assert not card.passed, "getting the cause right is not enough on its own"


# -- the no-deploy case, which is the point of the whole harness ----------


def test_blaming_a_pr_when_no_deploy_was_responsible_fails():
    """The failure this harness exists to catch."""
    confident_but_wrong = make_report(
        headline="PR-4901 caused the payment failures.",
        statement="PR-4901 broke the settlement path.",
        category="code_change",
        suspect="PR-4901",
        confidence=0.92,
    )
    card = score_report(confident_but_wrong, NO_CHANGE_TRUTH)

    assert not card.passed
    assert not check(card, "cause").passed
    assert "blamed" in check(card, "cause").detail
    assert not check(card, "calibration").passed, "92% confident and wrong"


def test_correctly_reporting_that_nothing_shipped_passes():
    report = make_report(
        headline="The upstream acquirer is returning 503s. No deploy is responsible.",
        statement="The acquirer-gateway provider is returning 503 for most authorisations.",
        category="external_provider",
        suspect="",
        reasoning="Errors are all upstream 503 from the acquirer. PR-4901 is "
        "observability-only on another service and cannot reach this path.",
        confidence=0.88,
        actions=("Open a ticket with northstar-acquiring", "Enable the fallback acquirer"),
    )
    card = score_report(report, NO_CHANGE_TRUTH)
    assert card.passed
    assert check(card, "cause").passed
    assert check(card, "decoys").passed


def test_naming_a_pr_only_in_the_statement_still_counts_as_blaming_it():
    report = make_report(
        headline="Something upstream broke.",
        statement="The change in PR-4901 is responsible for the 503s from the acquirer.",
        category="external_provider",  # category says otherwise, but it named a PR
        suspect="",
        confidence=0.5,
    )
    card = score_report(report, NO_CHANGE_TRUTH)
    assert not check(card, "cause").passed


# -- the remaining checks -------------------------------------------------


def test_a_single_hypothesis_fails_the_alternatives_check():
    card = score_report(make_report(), CODE_TRUTH)
    assert not check(card, "alternatives").passed


def test_dangling_citations_are_caught():
    card = score_report(make_report(supporting=("L1", "L999")), CODE_TRUTH)
    assert not check(card, "citations").passed
    assert "L999" in check(card, "citations").detail


def test_a_report_with_no_citations_at_all_fails():
    card = score_report(make_report(supporting=()), CODE_TRUTH)
    assert not check(card, "citations").passed
    assert "no citations" in check(card, "citations").detail


def test_being_wrong_but_appropriately_unsure_is_not_a_calibration_failure():
    card = score_report(make_report(suspect="PR-9999", confidence=0.3), CODE_TRUTH)
    assert not check(card, "cause").passed
    assert check(card, "calibration").passed, "low confidence while wrong is honest"


def test_a_report_with_no_actions_is_not_actionable():
    card = score_report(make_report(actions=(), verification=()), CODE_TRUTH)
    assert not check(card, "actionable").passed


def test_an_empty_hypothesis_list_scores_zero():
    report = make_report()
    report.analysis.hypotheses = []
    card = score_report(report, CODE_TRUTH)
    assert card.score == 0.0
    assert not card.passed


# -- aggregation ----------------------------------------------------------


def test_summary_counts_passes_and_per_check_rates():
    good = score_report(
        make_report(
            extra_hypotheses=[
                Hypothesis(
                    statement="PR-4817 banner change.",
                    category="code_change",
                    confidence=0.1,
                    reasoning="Ruled out, frontend only.",
                    supporting_evidence=[],
                    contradicting_evidence=["C1"],
                    suspect_change="",
                    verification_steps=["-"],
                )
            ]
        ),
        CODE_TRUTH,
    )
    bad = score_report(make_report(suspect="PR-4817"), CODE_TRUTH)

    summary = summarise([good, bad])
    assert summary["runs"] == 2
    assert summary["passed"] == 1
    assert summary["per_check"]["cause"] == "1/2"


def test_a_crashed_run_scores_zero_and_does_not_pass():
    from rca.evaluation import ScoreCard

    card = ScoreCard(incident_id="INC-X", error="LLMError: server unreachable")
    assert card.score == 0.0
    assert not card.passed
    assert summarise([card])["passed"] == 0
