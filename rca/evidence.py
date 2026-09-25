"""The evidence ledger.

Every fact the agent pulls gets a short, stable ID (`L3`, `C1`, `A2`). The
synthesis phase cites those IDs, and the report renderer resolves them back to
the underlying record. That is what keeps the final hypothesis auditable
instead of a plausible-sounding paragraph.
"""

from __future__ import annotations

from datetime import datetime

from rca.models import Alert, CodeChange, EvidenceItem, LogEntry, iso


class EvidenceLedger:
    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}
        self._seen: dict[str, str] = {}  # natural key -> evidence id
        self._counters = {"L": 0, "C": 0, "A": 0}

    # -- internals ---------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]}"

    def _record(
        self,
        prefix: str,
        key: str,
        kind: str,
        summary: str,
        source: str,
        timestamp: datetime | None,
    ) -> str:
        if key in self._seen:
            return self._seen[key]
        evidence_id = self._next_id(prefix)
        self._items[evidence_id] = EvidenceItem(
            id=evidence_id,
            kind=kind,  # type: ignore[arg-type]
            timestamp=timestamp,
            summary=summary,
            source=source,
        )
        self._seen[key] = evidence_id
        return evidence_id

    # -- public ------------------------------------------------------------

    def add_log(self, entry: LogEntry) -> str:
        key = f"log|{iso(entry.timestamp)}|{entry.service}|{entry.message}"
        return self._record(
            "L", key, "log", entry.one_line(), f"logs/{entry.service}", entry.timestamp
        )

    def add_change(self, change: CodeChange) -> str:
        key = f"change|{change.id}"
        return self._record(
            "C",
            key,
            "change",
            change.one_line(),
            change.url or "changes",
            change.deployed_at or change.merged_at,
        )

    def add_alert(self, alert: Alert) -> str:
        key = f"alert|{alert.id}"
        return self._record("A", key, "alert", alert.one_line(), "monitoring", alert.fired_at)

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self._items.get(evidence_id.strip().upper())

    def items(self) -> list[EvidenceItem]:
        def sort_key(item: EvidenceItem) -> tuple[int, int]:
            order = {"L": 0, "C": 1, "A": 2}
            return order.get(item.id[0], 9), int(item.id[1:] or 0)

        return sorted(self._items.values(), key=sort_key)

    def by_kind(self, kind: str) -> list[EvidenceItem]:
        return [item for item in self.items() if item.kind == kind]

    def __len__(self) -> int:
        return len(self._items)

    def summary_line(self) -> str:
        return (
            f"{self._counters['L']} log, {self._counters['C']} change, "
            f"{self._counters['A']} alert"
        )
