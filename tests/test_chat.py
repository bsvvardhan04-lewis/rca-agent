"""Follow-up chat over a finished report.

Two things here are actually worth testing, and they are both about grounding:

* the context handed to the model contains the evidence, the rejected
  alternatives and the gaps - so a correct answer is *possible*;
* a citation the ledger cannot resolve never reaches the reader looking like a
  citation.

The model is scripted. What is under test is the grounding and the lifecycle,
not the generation.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from rca.chat import ReportChat, suggested_questions
from rca.llm.base import LLMResponse, Usage
from rca.models import (
    ChatTurn,
    EvidenceItem,
    Hypothesis,
    IncidentReport,
    RCAAnalysis,
    RCAReport,
    TriagedIncident,
    iso,
    utcnow,
)
from tests.test_agent import analysis

ANSWER = "PR-4821 deployed at 13:31 [C1], and errors began at 13:34 [L1]."


class EchoLLM:
    """Records every call and replies with a canned answer."""

    def __init__(self, reply: str = ANSWER):
        self.model = "scripted-model"
        self.reply = reply
        self.systems: list[str] = []
        self.messages: list[list[dict]] = []
        self.kwargs: list[dict] = []

    def health(self) -> tuple[bool, str]:
        return True, "scripted"

    def chat(self, *, system, messages, **kwargs) -> LLMResponse:
        self.systems.append(system)
        self.messages.append(messages)
        self.kwargs.append(kwargs)
        return LLMResponse(text=self.reply, usage=Usage(prompt_tokens=800, completion_tokens=90))


def build_report(**overrides) -> RCAReport:
    now = utcnow()
    defaults: dict = dict(
        incident=IncidentReport(
            id="INC-1042",
            title="Checkout failing",
            description="Customers see 500s on pay.",
            reported_at=now,
        ),
        triage=TriagedIncident(
            summary="Checkout 500s",
            affected_services=["payments-api"],
            symptoms=["500 on /pay"],
            error_signatures=["500"],
            search_keywords=["checkout"],
            window_start=iso(now - timedelta(hours=3)),
            window_end=iso(now),
            severity="SEV2",
            triage_notes="payments-api owns /pay",
        ),
        analysis=analysis(["L1", "C1"]),
        evidence=[
            EvidenceItem(id="L1", kind="log", summary="ERROR pool exhausted", source="logs/pay"),
            EvidenceItem(id="C1", kind="change", summary="PR-4821 connection reuse", source="git"),
            EvidenceItem(id="A1", kind="alert", summary="ALERT-9 latency high", source="mon"),
        ],
        investigation_log=[f"step {i}" for i in range(40)],
        model="qwen2.5:7b",
    )
    defaults.update(overrides)
    return RCAReport(**defaults)


@pytest.fixture()
def report() -> RCAReport:
    return build_report()


@pytest.fixture()
def chat(report: RCAReport) -> ReportChat:
    return ReportChat(report, client=EchoLLM())


# -- grounding -------------------------------------------------------------


def test_context_carries_every_citable_id(chat: ReportChat):
    """The model can only cite what it was shown. An evidence item missing
    from the context makes a correct citation to it impossible."""
    context = chat.context()
    for eid in ("L1", "C1", "A1"):
        assert eid in context


def test_context_includes_what_was_not_checked(chat: ReportChat, report: RCAReport):
    """'What did you not look at?' is among the first questions a developer
    asks, and evidence_gaps is the only place that answer lives."""
    context = chat.context()
    assert "Evidence gaps" in context
    for gap in report.analysis.evidence_gaps:
        assert gap in context


def test_context_includes_the_reasoning_and_the_confidence(chat: ReportChat, report: RCAReport):
    """Answering 'how sure are you?' needs the number, and answering 'why?'
    needs the reasoning - neither is reconstructible from the headline."""
    context = chat.context()
    top = report.top_hypothesis
    assert top is not None
    assert top.reasoning in context
    assert f"{top.confidence:.2f}" in context


def test_long_investigation_log_is_tailed_not_dumped(chat: ReportChat):
    """40 steps of tool trace would crowd out the evidence it exists to
    explain. The recent steps are the ones that produced the conclusion."""
    context = chat.context()
    assert "step 39" in context
    assert "step 0\n" not in context
    assert "omitted" in context


def test_context_is_built_once(chat: ReportChat):
    assert chat.context() is chat.context()


def test_history_is_replayed_but_bounded(chat: ReportChat):
    history = [
        ChatTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
        for i in range(20)
    ]
    chat.answer("and the other one?", history)

    contents = [m["content"] for m in chat.client.messages[-1]]
    assert contents[-1] == "and the other one?"
    assert "turn 19" in contents
    assert "turn 0" not in contents, "old turns must fall out of the window"


def test_the_chat_is_offered_no_tools(chat: ReportChat):
    """A reader, not a second investigator. If it could call tools it could
    return findings that contradict the report on the developer's screen."""
    chat.answer("why?", [])
    assert not chat.client.kwargs[-1].get("tools")


def test_the_report_is_sent_as_context_not_as_the_question(chat: ReportChat):
    """The question goes last. A model handed 4KB of report *after* the
    question tends to summarise the report instead of answering."""
    messages = chat.client.messages[-1] if chat.client.messages else None
    chat.answer("was it the config?", [])
    messages = chat.client.messages[-1]
    assert "Evidence ledger" in messages[0]["content"]
    assert messages[-1]["content"] == "was it the config?"


def test_usage_is_accumulated_when_a_counter_is_passed(chat: ReportChat):
    total = Usage()
    chat.answer("why?", [], usage=total)
    chat.answer("and?", [], usage=total)
    assert total.prompt_tokens == 1600
    assert total.completion_tokens == 180


# -- citation hygiene ------------------------------------------------------


def test_real_citations_survive(chat: ReportChat):
    text, cited, unresolved = chat.verify("The deploy [C1] preceded the errors [L1].")
    assert cited == ["L1", "C1"]
    assert unresolved == []
    assert "[C1]" in text


def test_invented_citation_is_marked_not_deleted(chat: ReportChat):
    """Silently dropping `[L9]` leaves the sentence asserting a fact with no
    visible backing. The reader must see which claim to distrust."""
    text, cited, unresolved = chat.verify("Errors began at 13:34 [L9] after the deploy [C1].")
    assert unresolved == ["L9"]
    assert cited == ["C1"]
    assert "L9" not in text
    assert "[unverified]" in text
    assert "Errors began at 13:34" in text, "the claim itself stays"


def test_citations_are_deduplicated_and_ordered(chat: ReportChat):
    _, cited, _ = chat.verify("[A1] then [C1] then [L1] then [C1] again.")
    assert cited == ["L1", "C1", "A1"]


def test_an_empty_answer_is_not_a_crash(chat: ReportChat):
    assert chat.verify("") == ("", [], [])


def test_answer_reports_its_unresolved_citations(report: RCAReport):
    chat = ReportChat(report, client=EchoLLM("Caused by [C7], see [L1]."))
    result = chat.answer("what caused it?", [])
    assert result.unresolved == ["C7"]
    assert result.citations == ["L1"]
    assert "[unverified]" in result.text


def test_resolve_turns_ids_back_into_evidence(chat: ReportChat):
    rows = chat.resolve(["C1", "NOPE"])
    assert [r["id"] for r in rows] == ["C1"]
    assert rows[0]["summary"] == "PR-4821 connection reuse"


def test_empty_question_is_rejected_before_a_generation(chat: ReportChat):
    with pytest.raises(ValueError):
        chat.answer("   ", [])
    assert chat.client.messages == [], "no tokens should be spent on an empty question"


def test_a_report_with_no_evidence_permits_no_citations():
    """An investigation that retrieved nothing must not acquire authority in
    the chat that it did not earn in the report."""
    empty = ReportChat(build_report(evidence=[]), client=EchoLLM("Blame [C1]."))
    result = empty.answer("what happened?", [])
    assert result.citations == []
    assert result.unresolved == ["C1"]
    assert "you may cite nothing" in empty.context()


# -- suggestions -----------------------------------------------------------


def test_suggestions_come_from_this_report(report: RCAReport):
    """A static list of openers gets ignored. These have to name the actual
    suspect and the actual gap, or they are no better than a blank box."""
    questions = suggested_questions(report)
    assert 0 < len(questions) <= 4
    joined = " ".join(questions)
    top = report.top_hypothesis
    assert top is not None
    assert top.suspect_change in joined
    assert report.analysis.evidence_gaps[0] in joined


def test_suggestions_name_the_rejected_alternative_when_there_is_one(report: RCAReport):
    runner_up = Hypothesis(
        statement="A provider-side outage at the card processor.",
        category="external_provider",
        confidence=0.3,
        reasoning="Timing is close but no provider alert fired.",
        supporting_evidence=[],
        contradicting_evidence=["A1"],
        suspect_change="",
        verification_steps=["Check the provider status page"],
    )
    report.analysis.hypotheses.append(runner_up)
    assert any(runner_up.statement in q for q in suggested_questions(report))


def test_suggestions_survive_a_report_with_no_hypotheses():
    """An investigation can legitimately conclude nothing. The console still
    has to render."""
    thin = build_report(
        analysis=RCAAnalysis(
            headline="No cause identified.",
            timeline=[],
            hypotheses=[],
            blast_radius="Unknown.",
            immediate_actions=[],
            evidence_gaps=[],
        )
    )
    assert suggested_questions(thin) == ["What should I do first?"]
    ReportChat(thin, client=EchoLLM()).context()  # must not raise


# -- invented IDs with prefixes that do not exist ---------------------------


def test_an_id_with_an_unknown_prefix_is_still_flagged(chat: ReportChat):
    """Found on the live model: `Sources: [I1]`. The ledger only has L/C/A
    prefixes, so a scanner that only looked for those let `[I1]` through as if
    it were a citation. Any bracketed ID-shaped token must resolve or be marked."""
    text, cited, unresolved = chat.verify("Roll back the change.\nSources: [I1]")
    assert unresolved == ["I1"]
    assert cited == []
    assert "[I1]" not in text
    assert "[unverified]" in text


def test_a_sources_line_mixing_real_and_invented_ids_is_split(chat: ReportChat):
    text, cited, unresolved = chat.verify("Because of the deploy.\nSources: [C1] [E4] [L1]")
    assert cited == ["L1", "C1"]
    assert unresolved == ["E4"]


def test_sources_none_is_left_alone(chat: ReportChat):
    text, cited, unresolved = chat.verify("Roll it back.\nSources: none")
    assert text.endswith("Sources: none")
    assert cited == [] and unresolved == []


# -- the prompt rules that live testing showed are load-bearing --------------


def test_the_chat_prompt_keeps_its_scope_and_sources_rules():
    """Two rules were added only after running a real model: it answered
    'capital of France' with 'Paris', and it ignored an inline citation rule
    entirely. Both fixes live in prose, so pin them - a tidy-up edit must not
    silently drop them."""
    from rca.prompts import CHAT_SYSTEM

    flat = " ".join(CHAT_SYSTEM.split())  # the prompt is hard-wrapped; compare flowed text
    assert "I can only answer questions about this incident report." in flat
    assert "Sources:" in flat
    assert "Sources: none" in flat


# -- times the cited evidence does not contain -------------------------------


@pytest.fixture()
def timed() -> RCAReport:
    """Evidence that carries real timestamps - the shape the live report has."""
    return build_report(
        evidence=[
            EvidenceItem(
                id="A1", kind="alert", source="mon",
                summary="A-9019 [warning] LedgerSettlementRetryRate 2026-09-25T08:04:00Z -> ONGOING",
            ),
            EvidenceItem(
                id="C1", kind="change", source="git",
                summary="PR-4821 merged=2026-09-25T07:18:00Z deployed=2026-09-25T07:51:00Z",
            ),
        ]
    )


def test_a_time_the_cited_evidence_lacks_is_flagged(timed: RCAReport):
    """The exact live failure: the deploy time (07:51, evidence C1) stated as an
    alert time while citing only the alert (A1)."""
    reply = "The alert fired at 2026-09-25T08:04:00Z and again at 2026-09-25T07:51:00Z.\nSources: [A1]"
    result = ReportChat(timed, client=EchoLLM(reply)).answer("when did the alert fire?")

    assert result.unsupported_times == ["2026-09-25T07:51:00Z"]
    assert "[Check:" in result.text
    assert "2026-09-25T07:51:00Z" in result.text.split("[Check:")[1]


def test_a_time_the_cited_evidence_contains_is_not_flagged(timed: RCAReport):
    reply = "The alert fired at 2026-09-25T08:04:00Z.\nSources: [A1]"
    result = ReportChat(timed, client=EchoLLM(reply)).answer("when?")
    assert result.unsupported_times == []
    assert "[Check:" not in result.text


def test_citing_the_right_item_makes_the_same_time_valid(timed: RCAReport):
    """Same sentence, different citation: the check is about backing, not about
    the time being unknown to the report."""
    reply = "It deployed at 2026-09-25T07:51:00Z.\nSources: [C1]"
    assert ReportChat(timed, client=EchoLLM(reply)).answer("when?").unsupported_times == []


def test_a_time_with_no_citation_at_all_is_flagged(timed: RCAReport):
    reply = "It happened at 2026-09-25T08:04:00Z."
    assert ReportChat(timed, client=EchoLLM(reply)).answer("when?").unsupported_times == [
        "2026-09-25T08:04:00Z"
    ]


def test_an_answer_with_no_times_is_untouched(timed: RCAReport):
    result = ReportChat(timed, client=EchoLLM("Roll it back.\nSources: [C1]")).answer("what now?")
    assert result.unsupported_times == []
    assert result.text == "Roll it back.\nSources: [C1]"


def test_context_states_the_totals_so_only_one_change_can_be_counted(chat: ReportChat):
    """Live: asked 'was it the only change that day?' a 3B model answered 'No,
    there was another change' against a ledger holding exactly one. Counting is
    something it will do if the count is on the page."""
    context = chat.context()
    assert "Totals: 1 log line(s), 1 code change(s), 1 alert(s)" in context
    assert "no evidence of any other change" in context
