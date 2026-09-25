"""File-backed evidence sources: JSONL logs, JSON changes, JSON alerts.

These are the offline/demo implementations. They are also what the test suite
runs against, so the agent's behaviour is reproducible without network access.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from rca.models import Alert, CodeChange, LogEntry, parse_ts
from rca.sources.base import matches


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_no} is not valid JSON: {exc}") from exc
    return rows


class FileLogSource:
    """Reads `<service>.jsonl` files out of a directory."""

    name = "jsonl-files"

    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self._cache: tuple[LogEntry, ...] | None = None

    def _all(self) -> tuple[LogEntry, ...]:
        if self._cache is not None:
            return self._cache
        if not self.logs_dir.exists():
            self._cache = ()
            return self._cache
        entries: list[LogEntry] = []
        for path in sorted(self.logs_dir.glob("*.jsonl")):
            fallback_service = path.stem
            for row in _load_jsonl(path):
                entries.append(
                    LogEntry(
                        timestamp=parse_ts(row["timestamp"]),
                        level=str(row.get("level", "INFO")).upper(),
                        service=row.get("service", fallback_service),
                        message=row.get("message", ""),
                        trace_id=row.get("trace_id"),
                        attributes=row.get("attributes", {}),
                    )
                )
        entries.sort(key=lambda e: e.timestamp)
        self._cache = tuple(entries)
        return self._cache

    def services(self) -> list[str]:
        return sorted({e.service for e in self._all()})

    def _filter(
        self,
        *,
        start: datetime,
        end: datetime,
        query: str,
        services: list[str] | None,
        levels: list[str] | None,
    ) -> list[LogEntry]:
        wanted_services = {s.lower() for s in services} if services else None
        wanted_levels = {lv.upper() for lv in levels} if levels else None
        out: list[LogEntry] = []
        for entry in self._all():
            if not (start <= entry.timestamp <= end):
                continue
            if wanted_services and entry.service.lower() not in wanted_services:
                continue
            if wanted_levels and entry.level not in wanted_levels:
                continue
            if query:
                blob = f"{entry.message} {entry.service} {json.dumps(entry.attributes)}"
                if not matches(blob, query):
                    continue
            out.append(entry)
        return out

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
        hits = self._filter(
            start=start, end=end, query=query, services=services, levels=levels
        )
        # Newest first: during an incident the tail is what matters.
        hits.sort(key=lambda e: e.timestamp, reverse=True)
        return hits[:limit]

    def count(
        self,
        *,
        start: datetime,
        end: datetime,
        query: str = "",
        services: list[str] | None = None,
        levels: list[str] | None = None,
    ) -> int:
        return len(
            self._filter(start=start, end=end, query=query, services=services, levels=levels)
        )

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
        bucket = timedelta(minutes=max(1, bucket_minutes))
        hits = self._filter(
            start=start, end=end, query=query, services=services, levels=levels
        )
        counts: Counter[datetime] = Counter()
        for entry in hits:
            offset = (entry.timestamp - start) // bucket
            counts[start + offset * bucket] += 1

        out: list[tuple[datetime, int]] = []
        cursor = start
        while cursor < end:
            out.append((cursor, counts.get(cursor, 0)))
            cursor += bucket
        return out


class FileChangeSource:
    """Reads merged PRs / deploys out of `changes.json`."""

    name = "changes-json"

    def __init__(self, changes_dir: Path):
        self.changes_dir = Path(changes_dir)
        self._cache: tuple[CodeChange, ...] | None = None

    def _all(self) -> tuple[CodeChange, ...]:
        if self._cache is not None:
            return self._cache
        path = self.changes_dir / "changes.json"
        if not path.exists():
            self._cache = ()
            return self._cache
        rows = json.loads(path.read_text(encoding="utf-8"))
        changes = [
            CodeChange(
                id=row["id"],
                title=row["title"],
                author=row.get("author", "unknown"),
                merged_at=parse_ts(row["merged_at"]),
                deployed_at=parse_ts(row["deployed_at"]) if row.get("deployed_at") else None,
                services=row.get("services", []),
                files_changed=row.get("files_changed", []),
                additions=row.get("additions", 0),
                deletions=row.get("deletions", 0),
                url=row.get("url"),
                description=row.get("description", ""),
            )
            for row in rows
        ]
        changes.sort(key=lambda c: c.merged_at, reverse=True)
        self._cache = tuple(changes)
        return self._cache

    def list_changes(
        self,
        *,
        start: datetime,
        end: datetime,
        services: list[str] | None = None,
        query: str = "",
    ) -> list[CodeChange]:
        wanted = {s.lower() for s in services} if services else None
        out: list[CodeChange] = []
        for change in self._all():
            # A change is relevant if it *landed in production* in the window,
            # or was merged in the window and has not shipped yet.
            landed = change.deployed_at or change.merged_at
            if not (start <= landed <= end):
                continue
            # A change with no known services is *unattributed*, not
            # irrelevant - a root config or shared library can break any
            # service. Filtering it out silently during an incident is the
            # worse failure, so it survives the filter and the tool layer
            # flags it so the agent can weigh it for itself.
            if wanted and change.services and not ({s.lower() for s in change.services} & wanted):
                continue
            if query:
                blob = " ".join(
                    [change.title, change.description, change.author, *change.files_changed]
                )
                if not matches(blob, query):
                    continue
            out.append(change)
        out.sort(key=lambda c: c.deployed_at or c.merged_at, reverse=True)
        return out

    def get_change(self, change_id: str) -> CodeChange | None:
        needle = change_id.strip().lstrip("#").lower()
        for change in self._all():
            if change.id.lower() == needle or change.id.lower().lstrip("pr-") == needle:
                return change
        return None


class FileAlertSource:
    """Reads monitoring alerts out of `alerts.json`."""

    name = "alerts-json"

    def __init__(self, alerts_dir: Path):
        self.alerts_dir = Path(alerts_dir)
        self._cache: tuple[Alert, ...] | None = None

    def _all(self) -> tuple[Alert, ...]:
        if self._cache is not None:
            return self._cache
        path = self.alerts_dir / "alerts.json"
        if not path.exists():
            self._cache = ()
            return self._cache
        rows = json.loads(path.read_text(encoding="utf-8"))
        alerts = [
            Alert(
                id=row["id"],
                name=row["name"],
                service=row["service"],
                severity=row.get("severity", "warning"),
                fired_at=parse_ts(row["fired_at"]),
                resolved_at=parse_ts(row["resolved_at"]) if row.get("resolved_at") else None,
                value=row.get("value"),
                threshold=row.get("threshold"),
                labels=row.get("labels", {}),
            )
            for row in rows
        ]
        alerts.sort(key=lambda a: a.fired_at, reverse=True)
        self._cache = tuple(alerts)
        return self._cache

    def list_alerts(
        self,
        *,
        start: datetime,
        end: datetime,
        services: list[str] | None = None,
        include_resolved: bool = True,
    ) -> list[Alert]:
        wanted = {s.lower() for s in services} if services else None
        out: list[Alert] = []
        for alert in self._all():
            # Overlap test: the alert was firing at some point in the window.
            alert_end = alert.resolved_at or end
            if alert.fired_at > end or alert_end < start:
                continue
            if not include_resolved and alert.resolved_at is not None:
                continue
            if wanted and alert.service.lower() not in wanted:
                continue
            out.append(alert)
        out.sort(key=lambda a: a.fired_at, reverse=True)
        return out
