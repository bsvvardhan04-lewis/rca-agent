"""End-to-end pipeline tests against a scripted client.

No server, no credentials - but the tool calls run the *real* tools against
the *real* seeded data, so everything except the model's judgement is covered.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from rca.agent import RCAAgent, RCAError
from rca.models import Hypothesis, IncidentReport, RCAAnalysis, TriagedIncident, iso
from rca.report import to_markdown
from tests.conftest import ANCHOR
from tests.fakes import RefusingToolLLM, ScriptedLLM


@pytest.fixture()
def incident() -> IncidentReport:
    return IncidentReport(
        id="INC-1042",
        title="Checkout failing for a chunk of customers",
        description="Customers click Pay and get a generic error. Not everyone.",
        reported_at=ANCHOR,
        reporter="nisha.support",
        service_hint="checkout-web",
    )


@pytest.fixture()
def triage() -> TriagedIncident:
    return TriagedIncident(
        summary="Payment authorization failing intermittently.",
        affected_services=["payments-api", "checkout-web"],
        symptoms=["generic error on Pay", "intermittent"],
        error_signatures=["500", "TimeoutError"],
        search_keywords=["payment", "authorize"],
        window_start=iso(ANCHOR - timedelta(hours=3)),
        window_end=iso(ANCHOR + timedelta(minutes=5)),
        severity="SEV2",
        triage_notes="Widened to 3h because a deploy usually precedes symptoms.",
    )


def analysis(supporting: list[str], contradicting: list[str] | None = None) -> RCAAnalysis:
    return RCAAnalysis(
        headline="PR-4821 put a network call inside the payments DB transaction.",
        timeline=["13:38 - PR-4821 deployed", "13:45 - first pool timeout"],
        hypotheses=[
            Hypothesis(
                statement="Risk-service call inside tx_authorize exhausts the 10-connection pool.",
                category="code_change",
                confidence=0.9,
                reasoning="Deploy precedes onset by 7 minutes and the error names the pool bound.",
                supporting_evidence=supporting,
                contradicting_evidence=contradicting or [],
                suspect_change="PR-4821",
                verification_steps=["Roll back PR-4821", "Check pool utilisation"],
            )
        ],
        blast_radius="A share of checkout traffic.",
        immediate_actions=["Roll back PR-4821"],
        evidence_gaps=["No access to the diff itself."],
    )


# Two assistant turns: a parallel batch, then a follow-up.
SCRIPT = [
    [
        ("log_volume", {"levels": "ERROR", "bucket_minutes": 15}),
        ("list_alerts", {}),
    ],
    [
        ("search_logs", {"query": '"pool exhausted"', "limit": 5}),
        ("list_recent_changes", {}),
    ],
    [("get_change_detail", {"change_id": "PR-4821"})],
]

FINDINGS = "Errors begin ~13:45. PR-4821 deployed at 13:38 (C1). Pool bound is in L1."


def make_agent(sources, settings, triage, analysis_obj, *, script=SCRIPT, events=None,
               client_cls=ScriptedLLM, strategy="agentic"):
    client = client_cls(
        parsed={"TriagedIncident": triage, "RCAAnalysis": analysis_obj},
        tool_script=[list(batch) for batch in script] if script is not None else [],
        final_findings=FINDINGS,
    )
    agent = RCAAgent(
        sources,
        settings=replace(settings, strategy=strategy) if strategy else settings,
        client=client,
        on_event=(lambda k, m: events.append((k, m))) if events is not None else None,
    )
    return agent, client


# -- agentic strategy -----------------------------------------------------


def test_agentic_pipeline_produces_a_report(sources, settings, incident, triage):
    events: list[tuple[str, str]] = []
    agent, client = make_agent(
        sources, settings, triage, analysis(["L1", "C1", "A1"]), events=events
    )
    report = agent.investigate(incident)

    assert client.calls[0] == "structured:TriagedIncident"
    assert client.calls[-1] == "structured:RCAAnalysis"
    assert report.top_hypothesis.suspect_change == "PR-4821"
    assert len([m for k, m in events if k == "phase"]) == 3


def test_parallel_tool_calls_in_one_turn_all_execute(sources, settings, incident, triage):
    agent, _ = make_agent(sources, settings, triage, analysis(["L1"]))
    report = agent.investigate(incident)

    # 5 calls across 3 turns, including two parallel batches.
    assert len(report.investigation_log) == 5
    assert {item.kind for item in report.evidence} == {"log", "change", "alert"}
    assert any("PR-4821" in item.summary for item in report.evidence)


def test_the_model_is_offered_every_tool(sources, settings, incident, triage):
    agent, client = make_agent(sources, settings, triage, analysis(["L1"]))
    agent.investigate(incident)
    assert set(client.tools_offered[0]) == {
        "list_services",
        "log_volume",
        "search_logs",
        "list_recent_changes",
        "get_change_detail",
        "list_alerts",
    }


def test_an_unknown_tool_name_is_reported_to_the_model_not_raised(
    sources, settings, incident, triage
):
    agent, _ = make_agent(
        sources,
        settings,
        triage,
        analysis([]),
        script=[[("no_such_tool", {})], [("list_alerts", {})]],
    )
    report = agent.investigate(incident)
    assert report.analysis.headline  # it completed
    assert any(item.kind == "alert" for item in report.evidence)


def test_bad_arguments_do_not_crash_the_investigation(sources, settings, incident, triage):
    agent, _ = make_agent(
        sources,
        settings,
        triage,
        analysis([]),
        # Unknown keys are dropped; a bad timestamp falls back to the window.
        script=[[("search_logs", {"query": "ERROR", "nonsense": 1, "start": "whenever"})]],
    )
    report = agent.investigate(incident)
    assert report.investigation_log


def test_iteration_ceiling_is_enforced(sources, settings, incident, triage):
    capped = replace(settings, max_tool_iterations=2, strategy="agentic")
    events: list[tuple[str, str]] = []
    client = ScriptedLLM(
        parsed={"TriagedIncident": triage, "RCAAnalysis": analysis([])},
        tool_script=[[("list_alerts", {})] for _ in range(10)],
        final_findings=FINDINGS,
    )
    agent = RCAAgent(
        sources, settings=capped, client=client, on_event=lambda k, m: events.append((k, m))
    )
    report = agent.investigate(incident)

    assert len(report.investigation_log) == 2
    assert any("ceiling" in m for k, m in events if k == "warn")


def test_usage_is_accumulated_across_phases(sources, settings, incident, triage):
    agent, _ = make_agent(sources, settings, triage, analysis(["L1"]))
    report = agent.investigate(incident)
    assert report.usage.input_tokens > 0
    assert report.usage.output_tokens > 0


# -- guided strategy ------------------------------------------------------


def test_guided_strategy_runs_the_whole_sequence_without_tool_calling(
    sources, settings, incident, triage
):
    agent, client = make_agent(
        sources, settings, triage, analysis(["L1", "C1", "A1"]), script=None, strategy="guided"
    )
    report = agent.investigate(incident)

    # The model was never offered tools, yet all three sources were consulted.
    assert client.tools_offered == []
    assert {item.kind for item in report.evidence} == {"log", "change", "alert"}

    called = " ".join(report.investigation_log)
    assert "log_volume" in called
    assert "search_logs" in called
    assert "list_recent_changes" in called
    assert "get_change_detail" in called
    assert "list_alerts" in called


def test_guided_finds_the_onset_and_zooms_in(sources, settings, incident, triage):
    agent, _ = make_agent(
        sources, settings, triage, analysis([]), script=None, strategy="guided"
    )
    report = agent.investigate(incident)
    buckets = [c for c in report.investigation_log if "log_volume" in c]
    # A coarse pass then a fine one - that is the whole point of the sequence.
    assert any("bucket=15m" in c for c in buckets)
    assert any("bucket=2m" in c for c in buckets)


def test_guided_reads_the_change_that_landed_before_onset(sources, settings, incident, triage):
    agent, _ = make_agent(
        sources, settings, triage, analysis([]), script=None, strategy="guided"
    )
    report = agent.investigate(incident)
    details = [c for c in report.investigation_log if "get_change_detail" in c]
    assert any("PR-4821" in c for c in details)


# -- auto strategy --------------------------------------------------------


def test_auto_falls_back_to_guided_when_the_model_ignores_tools(
    sources, settings, incident, triage
):
    events: list[tuple[str, str]] = []
    agent, client = make_agent(
        sources,
        settings,
        triage,
        analysis([]),
        script=None,
        events=events,
        client_cls=RefusingToolLLM,
        strategy="auto",
    )
    report = agent.investigate(incident)

    assert any("Falling back to the guided sequence" in m for k, m in events if k == "warn")
    # Fallback still produced a full investigation.
    assert {item.kind for item in report.evidence} == {"log", "change", "alert"}


def test_auto_stays_agentic_when_the_model_uses_tools(sources, settings, incident, triage):
    events: list[tuple[str, str]] = []
    agent, _ = make_agent(
        sources, settings, triage, analysis([]), events=events, strategy="auto"
    )
    report = agent.investigate(incident)
    assert not any("Falling back" in m for k, m in events if k == "warn")
    assert len(report.investigation_log) == 5


# -- report integrity -----------------------------------------------------


def test_unresolvable_citations_are_dropped(sources, settings, incident, triage):
    events: list[tuple[str, str]] = []
    agent, _ = make_agent(
        sources,
        settings,
        triage,
        analysis(["L1", "L999", "MADE-UP"], contradicting=["C1", "A404"]),
        events=events,
    )
    report = agent.investigate(incident)

    h = report.top_hypothesis
    assert h.supporting_evidence == ["L1"]
    assert h.contradicting_evidence == ["C1"]
    assert any("Dropped 3 citation" in m for k, m in events if k == "warn")


def test_every_surviving_citation_resolves_in_the_rendered_report(
    sources, settings, incident, triage
):
    agent, _ = make_agent(sources, settings, triage, analysis(["L1", "C1", "A1"]))
    report = agent.investigate(incident)

    ids = {item.id for item in report.evidence}
    for h in report.analysis.hypotheses:
        for eid in h.supporting_evidence + h.contradicting_evidence:
            assert eid in ids

    markdown = to_markdown(report)
    assert "# RCA: INC-1042" in markdown
    assert "Roll back PR-4821" in markdown


def test_a_broken_triage_window_falls_back_instead_of_crashing(
    sources, settings, incident, triage
):
    triage.window_start = "not a timestamp"
    events: list[tuple[str, str]] = []
    agent, _ = make_agent(sources, settings, triage, analysis(["L1"]), events=events)
    report = agent.investigate(incident)

    assert any("falling back" in m for k, m in events if k == "warn")
    assert report.investigation_log
