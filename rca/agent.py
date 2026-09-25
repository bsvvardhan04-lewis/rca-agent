"""The RCA agent: triage -> investigate -> synthesise.

Three phases, deliberately separate:

* **Triage** turns the free-text ticket into search parameters. One structured
  call - there is nothing to explore yet.
* **Investigation** gathers evidence. Two strategies, see below.
* **Synthesis** ranks hypotheses over the evidence that was actually collected,
  so every citation resolves.

Splitting them matters: a single loop that both investigates and fills a report
schema stops investigating the moment it has enough to fill the schema.

## Two investigation strategies

`agentic` lets the model choose the next tool call. It is the better strategy
when the model can sustain a 15-step plan - a 32B/70B-class instruct model
served by vLLM. It can follow a thread we did not anticipate.

`guided` runs the query sequence in code - count, narrow, read, correlate
changes, check alert order - and asks the model to interpret the results in one
call. It cannot improvise, but it cannot flounder either, and it finishes in one
round trip instead of twenty.

The point is not that one is a fallback for the other. Small models cannot drive
a six-tool loop reliably, and on a slow host twenty sequential generations is
minutes of latency. `auto` picks agentic and drops to guided the moment the
model shows it is not using the tools.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from rca.config import Settings, settings as default_settings
from rca.evidence import EvidenceLedger
from rca.llm import LLMClient, LLMError, ToolSpec, Usage, build_client, complete_structured
from rca.models import (
    IncidentReport,
    RCAAnalysis,
    RCAReport,
    TimeWindow,
    TokenUsage,
    TriagedIncident,
    iso,
    parse_ts,
)
from rca.prompts import INVESTIGATOR_SYSTEM, SYNTHESIS_SYSTEM, TRIAGE_SYSTEM
from rca.sources.base import SourceBundle
from rca.tools import ToolContext, build_tools

EventHook = Callable[[str, str], None]


class RCAError(RuntimeError):
    """The investigation could not be completed."""


def _noop(kind: str, message: str) -> None:  # pragma: no cover - default hook
    pass


class RCAAgent:
    def __init__(
        self,
        sources: SourceBundle,
        *,
        settings: Settings | None = None,
        client: LLMClient | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.sources = sources
        self.settings = settings or default_settings
        self.client = client or build_client(self.settings)
        self.on_event: EventHook = on_event or _noop

    # -- phase 1 -----------------------------------------------------------

    def triage(self, incident: IncidentReport, usage: Usage) -> TriagedIncident:
        self.on_event("phase", "Triage - turning the report into search parameters")

        known = self.sources.logs.services()
        context = "\n".join(
            [
                incident.as_prompt(),
                "",
                "Services with logs available: " + (", ".join(known) or "(none)"),
                f"Current time: {iso(incident.reported_at)}",
            ]
        )

        triaged = complete_structured(
            self.client,
            system=TRIAGE_SYSTEM,
            user=context,
            model_cls=TriagedIncident,
            usage=usage,
            max_tokens=self.settings.max_tokens,
        )
        self.on_event(
            "detail",
            f"window {triaged.window_start} .. {triaged.window_end} | "
            f"services {', '.join(triaged.affected_services) or '-'} | {triaged.severity}",
        )
        return triaged

    # -- phase 2 -----------------------------------------------------------

    def _briefing(self, incident: IncidentReport, triage: TriagedIncident, ctx: ToolContext) -> str:
        return "\n".join(
            [
                "## Incident report (as filed)",
                incident.as_prompt(),
                "",
                "## Triage",
                f"Summary: {triage.summary}",
                f"Severity: {triage.severity}",
                f"Suspected services: {', '.join(triage.affected_services) or '(unknown)'}",
                f"Symptoms: {'; '.join(triage.symptoms) or '(unspecified)'}",
                f"Error signatures: {'; '.join(triage.error_signatures) or '(none given)'}",
                f"Keywords: {', '.join(triage.search_keywords) or '(none)'}",
                f"Investigation window: {ctx.window.describe()}",
                f"Triage notes: {triage.triage_notes}",
            ]
        )

    def investigate_loop(
        self,
        incident: IncidentReport,
        triage: TriagedIncident,
        ctx: ToolContext,
        usage: Usage,
    ) -> str:
        strategy = self.settings.strategy
        if strategy == "guided":
            return self._guided(incident, triage, ctx, usage)
        if strategy == "agentic":
            return self._agentic(incident, triage, ctx, usage)

        # auto
        try:
            findings = self._agentic(incident, triage, ctx, usage, allow_giveup=True)
        except _ModelNotUsingTools as exc:
            self.on_event("warn", f"{exc} Falling back to the guided sequence.")
            return self._guided(incident, triage, ctx, usage)
        return findings

    # -- agentic -----------------------------------------------------------

    def _agentic(
        self,
        incident: IncidentReport,
        triage: TriagedIncident,
        ctx: ToolContext,
        usage: Usage,
        allow_giveup: bool = False,
    ) -> str:
        self.on_event("phase", "Investigation (agentic) - the model drives the tools")

        tools = build_tools(ctx)
        by_name = {t.name: t for t in tools}
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._briefing(incident, triage, ctx)
                + "\n\nInvestigate using the tools, then report what you find.",
            }
        ]

        last_text = ""
        for iteration in range(1, self.settings.max_tool_iterations + 1):
            response = self.client.chat(
                system=INVESTIGATOR_SYSTEM,
                messages=messages,
                tools=tools,
                max_tokens=self.settings.max_tokens,
            )
            usage.add(response.usage)

            if response.text:
                last_text = response.text

            if not response.wants_tools:
                if iteration == 1 and allow_giveup and len(ctx.call_log) == 0:
                    raise _ModelNotUsingTools(
                        f"{self.client.model} answered without calling any tool."
                    )
                break

            messages.append(self._assistant_message(response))

            for call in response.tool_calls:
                spec = by_name.get(call.name)
                if spec is None:
                    result = (
                        f"Error: no tool named {call.name!r}. "
                        f"Available tools: {', '.join(by_name)}."
                    )
                else:
                    result = self._safe_call(spec, call.arguments)
                self.on_event("tool", f"{call.name}({_short(call.arguments)})")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        else:
            self.on_event(
                "warn",
                f"Hit the {self.settings.max_tool_iterations}-iteration ceiling; "
                "summarising what was gathered.",
            )
            last_text = last_text or self._wrap_up(ctx, usage)

        if not last_text:
            last_text = self._wrap_up(ctx, usage)

        self.on_event(
            "detail",
            f"{len(ctx.call_log)} tool call(s), evidence: {ctx.ledger.summary_line()}",
        )
        return last_text

    def _assistant_message(self, response: Any) -> dict[str, Any]:
        import json

        return {
            "role": "assistant",
            "content": response.text or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in response.tool_calls
            ],
        }

    @staticmethod
    def _safe_call(spec: ToolSpec, arguments: dict[str, Any]) -> str:
        """A bad tool call is a message to the model, not a crashed investigation."""
        try:
            return spec.call(arguments)
        except LLMError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced to the model
            return f"Error running {spec.name}: {type(exc).__name__}: {exc}"

    def _wrap_up(self, ctx: ToolContext, usage: Usage) -> str:
        """Ask for a written summary when the loop ended without one."""
        if not len(ctx.ledger):
            return ""
        ledger = "\n".join(item.one_line() for item in ctx.ledger.items())
        response = self.client.chat(
            system=INVESTIGATOR_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": "Here is the evidence gathered so far. Write your findings: "
                    "when it started, the most likely cause and its mechanism, what you "
                    "ruled out, and what you could not check.\n\n" + ledger,
                }
            ],
            max_tokens=self.settings.max_tokens,
        )
        usage.add(response.usage)
        return response.text

    # -- guided ------------------------------------------------------------

    def _guided(
        self,
        incident: IncidentReport,
        triage: TriagedIncident,
        ctx: ToolContext,
        usage: Usage,
    ) -> str:
        """Run the query sequence in code; the model reads the results once.

        This is the same method the investigator prompt describes, executed
        deterministically. Decisions (where is the spike?) read the sources
        directly; evidence collection goes through the tools so the ledger and
        the audit trail are identical to an agentic run.
        """
        self.on_event("phase", "Investigation (guided) - running the standard sequence")

        tools = {t.name: t for t in build_tools(ctx)}
        window = ctx.window
        services = ",".join(triage.affected_services[:3])
        transcript: list[str] = []

        def step(label: str, text: str) -> None:
            self.on_event("tool", label)
            transcript.append(f"### {label}\n{text}")

        # 1. Where are the errors, coarsely?
        step(
            "log_volume(levels=ERROR, bucket=15m)",
            tools["log_volume"].call({"levels": "ERROR", "bucket_minutes": 15}),
        )

        # 2. Locate the onset from the raw histogram, then zoom in.
        onset = self._find_onset(ctx)
        if onset is not None:
            zoom_start = onset - timedelta(minutes=20)
            zoom_end = min(onset + timedelta(minutes=20), window.end)
            step(
                f"log_volume(levels=ERROR, bucket=2m) around {iso(onset)}",
                tools["log_volume"].call(
                    {
                        "levels": "ERROR",
                        "bucket_minutes": 2,
                        "start": iso(zoom_start),
                        "end": iso(zoom_end),
                    }
                ),
            )
            # 3. Read a sample at the onset, and the warnings just before it.
            step(
                "search_logs(levels=ERROR) at onset",
                tools["search_logs"].call(
                    {"levels": "ERROR", "start": iso(onset), "end": iso(zoom_end), "limit": 12}
                ),
            )
            step(
                "search_logs(levels=WARN) just before onset",
                tools["search_logs"].call(
                    {
                        "levels": "WARN",
                        "start": iso(onset - timedelta(minutes=15)),
                        "end": iso(onset),
                        "limit": 10,
                    }
                ),
            )
        else:
            step(
                "search_logs(levels=ERROR,WARN) - no clear onset",
                tools["search_logs"].call({"levels": "ERROR,WARN", "limit": 15}),
            )

        # 4. What landed in production, and what does it actually do?
        changes_text = tools["list_recent_changes"].call({"services": services})
        if "No changes landed" in changes_text:
            changes_text = tools["list_recent_changes"].call({"lookback_hours": 48})
            step("list_recent_changes(lookback_hours=48) - nothing in window", changes_text)
        else:
            step(f"list_recent_changes(services={services!r})", changes_text)

        for change in self._candidate_changes(ctx, onset):
            step(
                f"get_change_detail({change!r})",
                tools["get_change_detail"].call({"change_id": change}),
            )

        # 5. Alert fire order gives the causal direction.
        step("list_alerts()", tools["list_alerts"].call({}))

        self.on_event(
            "detail",
            f"{len(ctx.call_log)} tool call(s), evidence: {ctx.ledger.summary_line()}",
        )

        prompt = "\n\n".join(
            [
                self._briefing(incident, triage, ctx),
                "## Evidence gathered",
                *transcript,
                "## Your task",
                "Write your findings: when the problem started and how you know; the single "
                "most likely cause with its mechanism spelled out; the alternatives you "
                "considered and what ruled each out; and what you could not check. "
                "Cite evidence IDs (L1, C2, A3) for every factual claim.",
            ]
        )
        response = self.client.chat(
            system=INVESTIGATOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.settings.max_tokens,
        )
        usage.add(response.usage)
        if not response.text.strip():
            raise RCAError("The model returned no findings from the guided investigation.")
        return response.text

    def _find_onset(self, ctx: ToolContext):
        """First bucket where errors go from quiet to busy."""
        buckets = ctx.sources.logs.histogram(
            start=ctx.window.start,
            end=ctx.window.end,
            bucket_minutes=5,
            levels=["ERROR"],
        )
        non_empty = [(ts, count) for ts, count in buckets if count]
        if not non_empty:
            return None
        peak = max(count for _, count in non_empty)
        # The onset is the first bucket carrying a real share of the peak, not
        # the first bucket with a single stray error in it.
        for ts, count in non_empty:
            if count >= max(2, peak * 0.2):
                return ts
        return non_empty[0][0]

    def _candidate_changes(self, ctx: ToolContext, onset) -> list[str]:
        """Changes worth reading in full: those that landed closest before onset."""
        changes = ctx.sources.changes.list_changes(
            start=ctx.window.start, end=ctx.window.end
        )
        if not changes:
            changes = ctx.sources.changes.list_changes(
                start=ctx.window.end - timedelta(hours=48), end=ctx.window.end
            )
        if onset is not None:
            before = [c for c in changes if (c.deployed_at or c.merged_at) <= onset]
            changes = before or changes
        return [c.id for c in changes[:3]]

    # -- phase 3 -----------------------------------------------------------

    def synthesise(
        self,
        incident: IncidentReport,
        triage: TriagedIncident,
        findings: str,
        ctx: ToolContext,
        usage: Usage,
    ) -> RCAAnalysis:
        self.on_event("phase", "Synthesis - ranking hypotheses and writing the report")

        ledger_lines = [item.one_line() for item in ctx.ledger.items()] or [
            "(no evidence was retrieved)"
        ]
        payload = "\n".join(
            [
                "## Original report",
                incident.as_prompt(),
                "",
                "## Triage",
                f"{triage.summary} (severity {triage.severity})",
                f"Window investigated: {ctx.window.describe()}",
                "",
                "## Evidence ledger - these are the ONLY citable IDs",
                *ledger_lines,
                "",
                "## Investigator findings",
                findings,
            ]
        )

        analysis = complete_structured(
            self.client,
            system=SYNTHESIS_SYSTEM,
            user=payload,
            model_cls=RCAAnalysis,
            usage=usage,
            max_tokens=self.settings.max_tokens,
        )

        # Drop citations the ledger cannot resolve rather than shipping a report
        # whose evidence IDs go nowhere.
        dropped = 0
        for hypothesis in analysis.hypotheses:
            for field_name in ("supporting_evidence", "contradicting_evidence"):
                ids = getattr(hypothesis, field_name)
                kept = [eid for eid in ids if ctx.ledger.get(eid) is not None]
                dropped += len(ids) - len(kept)
                setattr(hypothesis, field_name, kept)
        if dropped:
            self.on_event("warn", f"Dropped {dropped} citation(s) that did not resolve.")

        return analysis

    # -- orchestration -----------------------------------------------------

    def investigate(self, incident: IncidentReport) -> RCAReport:
        started = time.monotonic()
        usage = Usage()

        triage = self.triage(incident, usage)

        try:
            window = TimeWindow(
                start=parse_ts(triage.window_start), end=parse_ts(triage.window_end)
            )
        except ValueError:
            self.on_event("warn", "Triage returned an unparseable window; falling back to 4h.")
            window = TimeWindow(
                start=incident.reported_at - timedelta(hours=4),
                end=incident.reported_at + timedelta(minutes=10),
            )

        ctx = ToolContext(
            sources=self.sources,
            ledger=EvidenceLedger(),
            window=window,
            settings=self.settings,
        )

        findings = self.investigate_loop(incident, triage, ctx, usage)
        analysis = self.synthesise(incident, triage, findings, ctx, usage)

        return RCAReport(
            incident=incident,
            triage=triage,
            analysis=analysis,
            evidence=ctx.ledger.items(),
            investigation_log=list(ctx.call_log),
            model=self.settings.model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens
            ),
            duration_seconds=round(time.monotonic() - started, 1),
        )


class _ModelNotUsingTools(RuntimeError):
    """Internal signal for `auto`: this model is not driving the tool loop."""


def _short(arguments: dict[str, Any], limit: int = 70) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in arguments.items() if v not in ("", 0, None))
    return text[:limit] + ("..." if len(text) > limit else "")
