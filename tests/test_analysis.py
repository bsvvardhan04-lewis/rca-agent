"""Computed facts handed to the report writer.

Pure arithmetic and string handling, so these are exact. The agent-level tests at
the bottom check the facts actually reach the model and cannot break a report.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from rca.analysis import change_table, describe_change, dominant_errors, error_services
from rca.models import CodeChange
from tests.fakes import ScriptedLLM
from tests.test_agent import analysis, incident, make_agent, triage  # noqa: F401 - fixtures

UTC = timezone.utc
ONSET = datetime(2026, 9, 25, 8, 3, tzinfo=UTC)


def change(cid="PR-4821", deployed=ONSET - timedelta(minutes=12), services=("payments-api",), desc="Adds risk scoring."):
    return CodeChange(
        id=cid, title="Add merchant risk scoring", author="priya", merged_at=deployed - timedelta(minutes=30),
        deployed_at=deployed, services=list(services), description=desc,
    )


def logline(ts, level, service, message):
    return f"2026-09-25T{ts}Z {level:<5} [{service}] trace=6c0a404d6e2f883a {message}"


# -- error wording -----------------------------------------------------------


def test_repeated_errors_collapse_into_one_counted_line():
    """'waited 5001ms' and 'waited 5012ms' are the same error."""
    lines = [logline("08:12:04", "ERROR", "payments-api", f"TimeoutError: pool exhausted waited {5000 + i}ms") for i in range(5)]
    out = dominant_errors(lines)
    assert len(out) == 1
    assert out[0].startswith("x5  ERROR [payments-api]:")


def test_routine_info_lines_are_not_errors():
    lines = [logline("08:12:13", "INFO", "payments-api", "GET /healthz 200 in 71ms"),
             logline("08:12:14", "ERROR", "payments-api", "boom")]
    assert len(dominant_errors(lines)) == 1
    assert error_services(lines) == ["payments-api"]


def test_most_common_error_comes_first():
    lines = [logline("08:00:00", "WARN", "a", "slow query")] + \
            [logline("08:01:0%d" % i, "ERROR", "b", "pool exhausted") for i in range(3)]
    assert "pool exhausted" in dominant_errors(lines)[0]


def test_unparseable_summaries_are_skipped_not_fatal():
    assert dominant_errors(["free text", ""]) == []
    assert error_services(["free text"]) == []


def test_a_very_long_message_is_shortened():
    out = dominant_errors([logline("08:00:00", "ERROR", "a", "x" * 400)])
    assert len(out[0]) < 220 and out[0].endswith("...")


# -- change timing -------------------------------------------------------------


def test_a_change_before_onset_says_how_many_minutes():
    text = describe_change(change(), ONSET, ["payments-api"], "C1")
    assert text.startswith("[C1] PR-4821")
    assert "12 min BEFORE the first errors" in text
    assert "these also logged errors" in text


def test_a_change_after_onset_is_marked_as_unable_to_have_caused_it():
    """The newest-PR trap: a deploy that landed after the errors began."""
    late = change("PR-4817", deployed=ONSET + timedelta(minutes=20))
    text = describe_change(late, ONSET, ["payments-api"])
    assert "20 min AFTER the first errors began" in text
    assert "cannot have caused them" in text


def test_services_that_logged_no_errors_are_called_out():
    other = change("PR-4830", services=("notifications",))
    assert "none of these logged errors" in describe_change(other, ONSET, ["payments-api"])


def test_unknown_services_are_stated_not_guessed():
    assert "services unknown" in describe_change(change(services=()), ONSET, ["payments-api"])


def test_without_an_onset_the_timing_is_not_invented():
    assert "onset not established" in describe_change(change(), None, [])


def test_an_undeployed_change_says_so():
    c = CodeChange(id="PR-9", title="t", author="a", merged_at=ONSET - timedelta(hours=1), description="")
    assert "merged, no deploy time recorded" in describe_change(c, ONSET, [])


def test_description_is_included_and_bounded():
    text = describe_change(change(desc="y" * 900), ONSET, [])
    assert "Description: " in text and len(text) < 700 and text.endswith("...")


def test_table_is_newest_deploy_first():
    early, late = change("PR-1", ONSET - timedelta(hours=2)), change("PR-2", ONSET - timedelta(minutes=5))
    rows = change_table([early, late], ONSET, [], {"PR-1": "C1", "PR-2": "C2"})
    assert rows[0].startswith("[C2] PR-2") and rows[1].startswith("[C1] PR-1")


# -- wired into synthesis ------------------------------------------------------


class Capturing(ScriptedLLM):
    """Records the user message of every structured (synthesis) call."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.payloads: list[str] = []

    def chat(self, **kwargs):
        if kwargs.get("json_schema") is not None and kwargs.get("messages"):
            self.payloads.append(kwargs["messages"][-1]["content"])
        return super().chat(**kwargs)


def synthesis_payload(client) -> str:
    return next(p for p in client.payloads if "Evidence ledger" in p)


def test_the_writer_is_handed_the_candidate_changes_and_their_timing(sources, settings, incident, triage):
    agent, client = make_agent(sources, replace(settings, computed_facts=True), triage, analysis(["L1"]), client_cls=Capturing)
    agent.investigate(incident)
    payload = synthesis_payload(client)

    assert "## Candidate changes" in payload and "address EVERY one" in payload
    assert "PR-4821" in payload
    assert "BEFORE the first errors" in payload or "AFTER the first errors" in payload
    assert "## Dominant warning/error messages" in payload


def test_the_candidate_changes_become_citable_evidence(sources, settings, incident, triage):
    """A change the writer is told to address must be one it is allowed to cite."""
    agent, _ = make_agent(sources, replace(settings, computed_facts=True), triage, analysis(["L1"]))
    report = agent.investigate(incident)
    change_items = [e for e in report.evidence if e.kind == "change"]
    assert len(change_items) >= 2, "the decoy must be in the ledger too, not only the culprit"
    assert any("PR-4821" in e.summary for e in change_items)


def test_a_failure_computing_the_facts_degrades_to_no_extra_facts(sources, settings, incident, triage):
    """Extra facts are a help, never a dependency: the report must still be produced."""
    events: list = []
    agent, client = make_agent(sources, replace(settings, computed_facts=True), triage, analysis(["L1"]), client_cls=Capturing, events=events)

    def boom(_ctx):
        raise RuntimeError("log source down")

    agent._find_onset = boom   # only the synthesis step calls this in the agentic strategy
    report = agent.investigate(incident)

    assert report.analysis.headline
    assert "## Candidate changes" not in synthesis_payload(client)
    assert any(k == "warn" and "candidate-change facts" in m for k, m in events)


# -- off by default ------------------------------------------------------------
#
# Measured on llama3.2:3b over the four gold scenarios, enabling these facts made
# the model blame the look-alike PR in 4 of 4 (0 of 4 passed, mean 27% against
# 46% without). They are opt-in; these tests pin that the default is the one that
# was measured to be better.


def synthesis_system(client) -> str:
    return next(s for s in client.systems if "reporting step" in s)


def test_the_computed_facts_are_off_by_default(sources, settings, incident, triage):
    assert settings.computed_facts is False
    agent, client = make_agent(sources, settings, triage, analysis(["L1"]), client_cls=Capturing)
    agent.investigate(incident)

    payload = synthesis_payload(client)
    assert "## Candidate changes" not in payload
    assert "## Dominant warning/error messages" not in payload
    assert "address EVERY one" not in payload


def test_the_default_prompt_carries_no_addendum(sources, settings, incident, triage):
    agent, client = make_agent(sources, settings, triage, analysis(["L1"]), client_cls=Capturing)
    agent.investigate(incident)
    assert "Candidate changes" not in synthesis_system(client)


def test_enabling_the_facts_also_enables_their_instructions(sources, settings, incident, triage):
    """The prompt must not mention a section that is not in the payload, or the
    reverse."""
    agent, client = make_agent(sources, replace(settings, computed_facts=True), triage, analysis(["L1"]), client_cls=Capturing)
    agent.investigate(incident)
    assert "Candidate changes" in synthesis_system(client)
    assert "## Candidate changes" in synthesis_payload(client)


def test_the_candidate_changes_are_not_added_to_the_ledger_when_off(sources, settings, incident, triage):
    """With the facts off the decoy only enters the ledger if the investigation
    itself retrieved it - the report must not change behind the caller's back."""
    agent_off, _ = make_agent(sources, settings, triage, analysis(["L1"]))
    agent_on, _ = make_agent(sources, replace(settings, computed_facts=True), triage, analysis(["L1"]))
    off = len([e for e in agent_off.investigate(incident).evidence if e.kind == "change"])
    on = len([e for e in agent_on.investigate(incident).evidence if e.kind == "change"])
    assert on >= off
