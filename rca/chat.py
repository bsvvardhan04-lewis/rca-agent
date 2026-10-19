"""Follow-up Q&A over a finished RCA report.

The agent pushes a hypothesis at the developer before they start looking. The
first thing a developer does with a hypothesis is argue with it: *why not the
other PR? how sure are you? what did you not check?* Answering that is a
different job from investigating, and it deserves a different component.

What this is **not**: a second investigator. There are no tools here and no
source access. The evidence was collected once, during the investigation, and
this module answers strictly from that ledger. That is a deliberate limit -

* an answer that re-queried the sources could contradict the report the
  developer is reading, which is worse than no answer;
* every claim stays traceable to an ID the report already prints;
* a follow-up costs one generation instead of a fresh twenty-step loop.

If a question genuinely needs new evidence, the honest answer is "that was not
checked" - and the right next step is to file a new incident, not to let the
model improvise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rca.config import Settings, settings as default_settings
from rca.llm import LLMClient, Usage, build_client
from rca.models import ChatTurn, RCAReport, iso
from rca.prompts import CHAT_SYSTEM
from rca.report import band

# `L4`, `[C1]`, `A12` - the shapes a model actually emits when citing.
CITATION = re.compile(r"\[?\b([LCA]\d{1,3})\b\]?")
# Any bracketed token shaped like an evidence ID is treated as a citation - not
# just the L/C/A prefixes that exist. A model that writes `[I1]` or `[E4]` has
# invented an ID, and an invented ID that passes through unflagged looks exactly
# like a real one.
# Full ISO timestamps, the form every evidence summary carries.
TIMESTAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?\b")

BRACKETED = re.compile(r"\[([A-Z]{1,2}\d{1,4})\]")

# How much of the tail of the investigation log to show. The log is a trace of
# tool calls; the last few are the ones that produced the conclusion.
LOG_TAIL = 25

# Turns of prior conversation to replay. Long enough for "and what about the
# other one?" to resolve, short enough that context stays mostly evidence.
HISTORY_TURNS = 8


@dataclass
class ChatAnswer:
    text: str
    citations: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    unsupported_times: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class ReportChat:
    """Answers questions about one finished `RCAReport`.

    Stateless with respect to the conversation: history is passed in by the
    caller, so the same instance serves any number of conversations and the
    transcript lives wherever the caller keeps its jobs.
    """

    def __init__(
        self,
        report: RCAReport,
        *,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ):
        self.report = report
        self.settings = settings or default_settings
        self.client = client or build_client(self.settings)
        self._context: str | None = None
        self._ids = {item.id for item in report.evidence}

    # -- grounding ---------------------------------------------------------

    def known_ids(self) -> set[str]:
        return set(self._ids)

    def context(self) -> str:
        """The whole report, flattened into the one message the model reads.

        Built once and cached: for a given report it never changes, and it is
        by far the largest part of every request.
        """
        if self._context is not None:
            return self._context

        r = self.report
        a = r.analysis
        out: list[str] = [
            "## Incident",
            r.incident.as_prompt(),
            "",
            "## Triage",
            f"Summary: {r.triage.summary}",
            f"Severity: {r.triage.severity}",
            f"Services investigated: {', '.join(r.triage.affected_services) or '(none)'}",
            f"Window: {r.triage.window_start} .. {r.triage.window_end}",
            f"Notes: {r.triage.triage_notes}",
            "",
            "## Conclusion",
            f"Headline: {a.headline}",
            f"Blast radius: {a.blast_radius}",
            "",
        ]

        if a.timeline:
            out.append("## Timeline")
            out.extend(f"- {entry}" for entry in a.timeline)
            out.append("")

        out.append("## Hypotheses, ranked")
        if not a.hypotheses:
            out.append("(the investigation produced no hypothesis)")
        for rank, h in enumerate(a.hypotheses, start=1):
            out.extend(
                [
                    f"### {rank}. {h.statement}",
                    f"Category: {h.category}",
                    f"Confidence: {h.confidence:.2f} ({band(h.confidence)})",
                    f"Reasoning: {h.reasoning}",
                    f"Supports: {', '.join(h.supporting_evidence) or '(none cited)'}",
                    f"Contradicts: {', '.join(h.contradicting_evidence) or '(none cited)'}",
                    f"Suspect change: {h.suspect_change or '(none)'}",
                ]
            )
            if h.verification_steps:
                out.append("Verification steps:")
                out.extend(f"  - {step}" for step in h.verification_steps)
            out.append("")

        if a.immediate_actions:
            out.append("## Immediate actions recommended")
            out.extend(f"- {item}" for item in a.immediate_actions)
            out.append("")

        if a.evidence_gaps:
            out.append("## Evidence gaps - what was NOT checked")
            out.extend(f"- {item}" for item in a.evidence_gaps)
            out.append("")

        out.append("## Evidence ledger - these are the ONLY citable IDs")
        if r.evidence:
            out.extend(item.one_line() for item in r.evidence)
        else:
            out.append("(no evidence was retrieved - you may cite nothing)")
        out.append("")

        if r.investigation_log:
            tail = r.investigation_log[-LOG_TAIL:]
            elided = len(r.investigation_log) - len(tail)
            out.append("## Investigation trace (what the agent actually did)")
            if elided:
                out.append(f"({elided} earlier step(s) omitted)")
            out.extend(f"- {line}" for line in tail)
            out.append("")

        out.append(
            f"Report generated {iso(r.generated_at)} by {r.model or 'unknown model'} "
            f"in {r.duration_seconds}s."
        )

        self._context = "\n".join(out)
        return self._context

    # -- answering ---------------------------------------------------------

    def answer(
        self,
        question: str,
        history: list[ChatTurn] | None = None,
        *,
        usage: Usage | None = None,
    ) -> ChatAnswer:
        question = question.strip()
        if not question:
            raise ValueError("The question is empty.")

        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    "Here is the finished RCA report you are answering questions "
                    f"about.\n\n{self.context()}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I have read the report and the evidence ledger. Ask me about it "
                    "and I will answer from the evidence, citing IDs."
                ),
            },
        ]
        for turn in (history or [])[-HISTORY_TURNS:]:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": question})

        response = self.client.chat(
            system=CHAT_SYSTEM,
            messages=messages,
            # Slightly above zero: a follow-up is a conversation, and a fully
            # greedy decode tends to repeat the report's own phrasing verbatim
            # instead of answering the question that was asked.
            temperature=0.2,
            max_tokens=min(self.settings.max_tokens, 1024),
        )
        if usage is not None:
            usage.add(response.usage)

        text, cited, unresolved = self.verify(response.text)

        wrong_times = self.unsupported_times(text, cited)
        if wrong_times:
            text += (
                "\n\n[Check: the time(s) " + ", ".join(wrong_times) + " do not appear in "
                "the evidence cited above. Verify them before relying on this answer.]"
            )
        return ChatAnswer(
            text=text,
            citations=cited,
            unresolved=unresolved,
            unsupported_times=wrong_times,
            usage=response.usage,
        )

    # -- citation hygiene --------------------------------------------------

    def verify(self, text: str) -> tuple[str, list[str], list[str]]:
        """Split an answer's citations into resolvable and invented.

        A hallucinated `[L9]` is more damaging than no citation at all: it
        reads as checkable and sends the developer looking for a log line that
        was never retrieved. So rather than silently dropping it - which would
        leave the surrounding sentence asserting a fact with no visible
        backing - it is rewritten in place to `[unverified]`, and the sentence
        stays. The reader can see exactly which claim to distrust.
        """
        text = (text or "").strip()
        if not text:
            return "", [], []

        unresolved: list[str] = []

        def rewrite(match: re.Match[str]) -> str:
            eid = match.group(1)
            if eid in self._ids:
                return match.group(0)
            if eid not in unresolved:
                unresolved.append(eid)
            return "[unverified]"

        text = BRACKETED.sub(rewrite, text)

        cited: list[str] = []
        for match in CITATION.finditer(text):
            eid = match.group(1)
            if eid in self._ids and eid not in cited:
                cited.append(eid)

        order = {"L": 0, "C": 1, "A": 2}
        cited.sort(key=lambda e: (order.get(e[0], 9), int(e[1:])))
        return text, cited, unresolved

    def unsupported_times(self, text: str, cited: list[str]) -> list[str]:
        """Timestamps in the answer that the cited evidence does not contain.

        `verify` only proves a cited ID *exists*; it cannot tell that the
        sentence beside it is true. Seen on a live 3B model: "the alert fired
        again at 07:51:00Z", citing a real alert - 07:51 was the deploy time of
        a pull request, not an alert. Wrong times are the most damaging error an
        incident assistant can make, and the one that can be caught without
        another model call: a time the answer states must be one the evidence it
        cites actually carries.

        A heuristic with known limits - it sees full ISO timestamps only, not
        "around 8am", and it cannot judge a correct time attached to the wrong
        event. It narrows the gap; it does not close it.
        """
        by_id = {item.id: item.summary for item in self.report.evidence}
        backing = " ".join(by_id[eid] for eid in cited if eid in by_id)
        have = {t.rstrip("Z") for t in TIMESTAMP.findall(backing)}

        bad: list[str] = []
        for stamp in TIMESTAMP.findall(text):
            if stamp.rstrip("Z") not in have and stamp not in bad:
                bad.append(stamp)
        return bad

    def resolve(self, ids: list[str]) -> list[dict[str, str]]:
        """Turn cited IDs back into the evidence lines they point at."""
        by_id = {item.id: item for item in self.report.evidence}
        out = []
        for eid in ids:
            item = by_id.get(eid)
            if item is not None:
                out.append({"id": item.id, "kind": item.kind, "summary": item.summary})
        return out


def suggested_questions(report: RCAReport, limit: int = 4) -> list[str]:
    """Openers derived from this specific report, not a static list.

    A blank chat box in front of a report nobody has read yet gets no use. The
    questions worth surfacing are the ones the report itself invites: the
    rejected alternative, the confidence, the gap.
    """
    a = report.analysis
    out: list[str] = []
    top = report.top_hypothesis

    if top and top.suspect_change:
        out.append(f"How do you know {top.suspect_change} caused this?")
    if len(a.hypotheses) > 1:
        runner_up = a.hypotheses[1]
        out.append(f"Why is it not this instead: {runner_up.statement}")
    if top:
        out.append(f"What would prove this wrong? (currently {band(top.confidence)})")
    if a.evidence_gaps:
        out.append(f"You said you could not check: {a.evidence_gaps[0]} - does that matter?")
    out.append("What should I do first?")

    return out[:limit]
