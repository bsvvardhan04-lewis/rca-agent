"""Domain models for the RCA agent.

Two layers live here on purpose:

* **Domain models** (`LogEntry`, `CodeChange`, `Alert`, ...) are what the
  source adapters return. They use real `datetime` objects.
* **LLM-facing models** (`TriagedIncident`, `Hypothesis`, `RCAAnalysis`) are
  what we hand to `messages.parse()` as a JSON schema. They keep timestamps as
  ISO strings and avoid exotic types so the generated schema stays strict-safe.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(UTC)


def parse_ts(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing 'Z' and naive input."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------


class Severity(str, Enum):
    sev1 = "SEV1"
    sev2 = "SEV2"
    sev3 = "SEV3"
    sev4 = "SEV4"


class IncidentReport(BaseModel):
    """The raw bug report / incident ticket as it arrives from a human or webhook."""

    id: str
    title: str
    description: str
    reported_at: datetime = Field(default_factory=utcnow)
    reporter: str = "unknown"
    service_hint: str | None = None
    severity_hint: Severity | None = None

    def as_prompt(self) -> str:
        lines = [
            f"Incident ID: {self.id}",
            f"Title: {self.title}",
            f"Reported at: {iso(self.reported_at)}",
            f"Reported by: {self.reporter}",
        ]
        if self.service_hint:
            lines.append(f"Service hint: {self.service_hint}")
        if self.severity_hint:
            lines.append(f"Severity hint: {self.severity_hint.value}")
        lines.append("")
        lines.append("Description:")
        lines.append(self.description.strip())
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Evidence sources
# --------------------------------------------------------------------------


class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    service: str
    message: str
    trace_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    def one_line(self) -> str:
        trace = f" trace={self.trace_id}" if self.trace_id else ""
        return f"{iso(self.timestamp)} {self.level:<5} [{self.service}]{trace} {self.message}"


class CodeChange(BaseModel):
    """A merged pull request or commit that could plausibly have caused an incident."""

    id: str
    title: str
    author: str
    merged_at: datetime
    services: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    url: str | None = None
    description: str = ""
    deployed_at: datetime | None = None

    def one_line(self) -> str:
        deploy = f" deployed={iso(self.deployed_at)}" if self.deployed_at else ""
        svc = ",".join(self.services) or "-"
        return (
            f"{self.id} {self.title!r} by {self.author} merged={iso(self.merged_at)}"
            f"{deploy} services={svc} (+{self.additions}/-{self.deletions}, "
            f"{len(self.files_changed)} files)"
        )


class Alert(BaseModel):
    id: str
    name: str
    service: str
    severity: str
    fired_at: datetime
    resolved_at: datetime | None = None
    value: float | None = None
    threshold: float | None = None
    labels: dict[str, str] = Field(default_factory=dict)

    def one_line(self) -> str:
        end = iso(self.resolved_at) if self.resolved_at else "ONGOING"
        measure = ""
        if self.value is not None and self.threshold is not None:
            measure = f" value={self.value} threshold={self.threshold}"
        return (
            f"{self.id} [{self.severity}] {self.name} on {self.service} "
            f"{iso(self.fired_at)} -> {end}{measure}"
        )


class TimeWindow(BaseModel):
    start: datetime
    end: datetime

    @classmethod
    def around(cls, moment: datetime, before: timedelta, after: timedelta) -> TimeWindow:
        return cls(start=moment - before, end=moment + after)

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end

    def describe(self) -> str:
        return f"{iso(self.start)} .. {iso(self.end)}"


# --------------------------------------------------------------------------
# LLM-facing schemas (strict JSON, ISO strings, no surprises)
# --------------------------------------------------------------------------


class TriagedIncident(BaseModel):
    """Phase 1 output: the free-text report turned into something queryable."""

    summary: str = Field(description="One-sentence restatement of what is broken.")
    affected_services: list[str] = Field(
        description="Service names most likely involved, best guess first."
    )
    symptoms: list[str] = Field(description="Observable symptoms, one per item.")
    error_signatures: list[str] = Field(
        description="Distinctive strings to grep logs for: exception names, "
        "status codes, error messages. Short and literal."
    )
    search_keywords: list[str] = Field(
        description="Broader keywords for searching changes and alerts."
    )
    window_start: str = Field(
        description="ISO-8601 UTC start of the window worth investigating."
    )
    window_end: str = Field(description="ISO-8601 UTC end of the window worth investigating.")
    severity: Literal["SEV1", "SEV2", "SEV3", "SEV4"]
    triage_notes: str = Field(description="Why this window and these services.")


class Hypothesis(BaseModel):
    """One candidate root cause."""

    statement: str = Field(description="The root cause, stated as a single claim.")
    category: Literal[
        "code_change",
        "configuration",
        "dependency_failure",
        "resource_exhaustion",
        "data_issue",
        "infrastructure",
        "external_provider",
        "unknown",
    ]
    confidence: float = Field(ge=0.0, le=1.0, description="0.0-1.0 calibrated confidence.")
    reasoning: str = Field(description="How the evidence leads to this conclusion.")
    supporting_evidence: list[str] = Field(
        description="Evidence IDs (e.g. 'L3', 'C1', 'A2') that support this."
    )
    contradicting_evidence: list[str] = Field(
        description="Evidence IDs that argue against it. Empty list if none."
    )
    suspect_change: str = Field(
        description="ID of the code change most implicated, or empty string."
    )
    verification_steps: list[str] = Field(
        description="Concrete things the on-call engineer should check to confirm or kill this."
    )


class RCAAnalysis(BaseModel):
    """Phase 3 output: the finished hypothesis set."""

    headline: str = Field(description="One line an on-call engineer can read in 3 seconds.")
    timeline: list[str] = Field(
        description="Ordered 'HH:MM:SS - what happened' entries reconstructed from evidence."
    )
    hypotheses: list[Hypothesis] = Field(description="Ranked most likely first.")
    blast_radius: str = Field(description="Who and what is affected, as far as evidence shows.")
    immediate_actions: list[str] = Field(
        description="What to do in the next 15 minutes: rollback, flag flip, scale up."
    )
    evidence_gaps: list[str] = Field(
        description="What could not be checked and would change the answer."
    )


# --------------------------------------------------------------------------
# Assembled report
# --------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """A single numbered fact the agent pulled, so hypotheses can cite it."""

    id: str
    kind: Literal["log", "change", "alert"]
    timestamp: datetime | None = None
    summary: str
    source: str

    def one_line(self) -> str:
        return f"[{self.id}] {self.summary}"


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens += (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class RCAReport(BaseModel):
    incident: IncidentReport
    triage: TriagedIncident
    analysis: RCAAnalysis
    evidence: list[EvidenceItem]
    investigation_log: list[str] = Field(
        default_factory=list,
        description="Human-readable trace of which tools the agent called.",
    )
    model: str = ""
    generated_at: datetime = Field(default_factory=utcnow)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_seconds: float = 0.0

    @property
    def top_hypothesis(self) -> Hypothesis | None:
        return self.analysis.hypotheses[0] if self.analysis.hypotheses else None
