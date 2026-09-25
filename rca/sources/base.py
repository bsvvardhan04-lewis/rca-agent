"""Source interfaces.

Each evidence source is a small protocol so the agent never learns where the
data physically lives. The demo ships file-backed implementations; swapping in
Loki, Datadog, or the GitHub API means writing one class, not touching the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from rca.models import Alert, CodeChange, LogEntry


@runtime_checkable
class LogSource(Protocol):
    name: str

    def services(self) -> list[str]:
        """Every service that has logs available."""

    def search(
        self,
        *,
        start: datetime,
        end: datetime,
        query: str = "",
        services: list[str] | None = None,
        levels: list[str] | None = None,
        limit: int = 50,
    ) -> list[LogEntry]:
        """Newest-first matching log lines inside [start, end]."""

    def histogram(
        self,
        *,
        start: datetime,
        end: datetime,
        bucket_minutes: int = 5,
        services: list[str] | None = None,
        levels: list[str] | None = None,
        query: str = "",
    ) -> list[tuple[datetime, int]]:
        """Counts per time bucket - lets the agent find *when* without reading lines."""


@runtime_checkable
class ChangeSource(Protocol):
    name: str

    def list_changes(
        self,
        *,
        start: datetime,
        end: datetime,
        services: list[str] | None = None,
        query: str = "",
    ) -> list[CodeChange]:
        """Changes merged or deployed inside [start, end], newest first."""

    def get_change(self, change_id: str) -> CodeChange | None:
        """Full detail for one change, including its file list."""


@runtime_checkable
class AlertSource(Protocol):
    name: str

    def list_alerts(
        self,
        *,
        start: datetime,
        end: datetime,
        services: list[str] | None = None,
        include_resolved: bool = True,
    ) -> list[Alert]:
        """Alerts that were firing at any point inside [start, end]."""


@dataclass
class SourceBundle:
    """Everything the investigator can reach, in one place."""

    logs: LogSource
    changes: ChangeSource
    alerts: AlertSource

    def describe(self) -> str:
        return (
            f"logs={self.logs.name} changes={self.changes.name} alerts={self.alerts.name}"
        )


def matches(text: str, query: str) -> bool:
    """Case-insensitive AND-match over whitespace-separated terms.

    Deliberately simple: an incident responder greps, they do not write DSL.
    Quoted phrases are honoured so 'pool exhausted' can be searched as a unit.
    """
    if not query:
        return True
    haystack = text.lower()
    terms: list[str] = []
    buf: list[str] = []
    in_quote = False
    for ch in query:
        if ch == '"':
            in_quote = not in_quote
            if not in_quote and buf:
                terms.append("".join(buf).strip().lower())
                buf = []
            continue
        if ch.isspace() and not in_quote:
            if buf:
                terms.append("".join(buf).strip().lower())
                buf = []
            continue
        buf.append(ch)
    if buf:
        terms.append("".join(buf).strip().lower())
    return all(term in haystack for term in terms if term)
