"""A timeline built from the evidence, not written by the model.

The synthesis prompt asks the model to reconstruct the timeline "strictly from
evidence timestamps". A 3B model on the live run did not: it reported an alert
firing at 08:12:24 when the alert fired at 08:04:00 and 08:12:24 was an unrelated
log line, listed a successful 200 response as an incident event, and repeated one
entry twice. A timeline is a table of facts. Building it in code makes it correct
by construction and lets every line carry the ID of the evidence behind it.

What this does *not* do is explain anything. Causation is the model's job and
stays in the hypotheses; this only says what happened when.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rca.models import EvidenceItem, iso

MAX_ENTRIES = 14
LABEL_CHARS = 100

# Shapes of the one-line summaries the evidence ledger writes (rca/models.py).
CHANGE = re.compile(r"^(?P<id>\S+) '(?P<title>.*?)' by ")
ALERT = re.compile(r"^(?P<id>\S+) \[(?P<severity>[^\]]+)\] (?P<name>\S+) on (?P<service>\S+) ")
LOG = re.compile(r"^\S+\s+(?P<level>\w+)\s+\[(?P<service>[^\]]+)\](?: trace=\S+)? (?P<message>.*)$")

# Routine traffic. Once there are real WARN/ERROR lines, "GET /healthz 200" is
# not an incident event - it is what the log search happened to also return.
QUIET_LEVELS = {"INFO", "DEBUG", "TRACE"}

# Numbers and long hex ids differ between otherwise identical log lines.
VOLATILE = re.compile(r"\b[0-9a-f]{8,}\b|\d+")


@dataclass
class _Entry:
    item: EvidenceItem
    label: str
    key: tuple[str, str, str] | None = None  # set for logs, to collapse repeats
    repeats: int = 1


def _shorten(text: str, limit: int = LABEL_CHARS) -> str:
    text = " ".join(text.split())
    # Plain ASCII: a Unicode ellipsis prints as a replacement character on a
    # default Windows console, and the timeline is meant to be read there.
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _label(item: EvidenceItem) -> tuple[str, tuple[str, str, str] | None]:
    if item.kind == "change":
        m = CHANGE.match(item.summary)
        if m:
            deployed = "deployed=" in item.summary
            verb = "deployed" if deployed else "merged (no deploy time recorded)"
            return f"{m['id']} {verb}: {_shorten(m['title'], 70)}", None
    elif item.kind == "alert":
        m = ALERT.match(item.summary)
        if m:
            return f"alert {m['name']} fired on {m['service']} ({m['severity']})", None
    elif item.kind == "log":
        m = LOG.match(item.summary)
        if m:
            key = (m["level"], m["service"], VOLATILE.sub("#", m["message"]))
            return f"{m['level']} {m['service']}: {_shorten(m['message'])}", key
    return _shorten(item.summary), None


def build_timeline(evidence: list[EvidenceItem], max_entries: int = MAX_ENTRIES) -> list[str]:
    """Chronological `ISO-time - what happened [evidence id]` lines.

    Every change and alert is kept. Log lines are collapsed to one entry per
    distinct message shape ("x14" for repeats), routine INFO/DEBUG lines are
    dropped whenever warnings or errors exist, and if there are still too many
    the earliest are kept: the first errors are the onset, and the onset is the
    part of a timeline anyone reads.
    """
    dated = sorted(
        (item for item in evidence if item.timestamp is not None),
        key=lambda item: (item.timestamp, item.id),
    )

    entries: list[_Entry] = []
    by_shape: dict[tuple[str, str, str], _Entry] = {}
    for item in dated:
        label, key = _label(item)
        if key is not None and key in by_shape:
            by_shape[key].repeats += 1
            continue
        entry = _Entry(item=item, label=label, key=key)
        entries.append(entry)
        if key is not None:
            by_shape[key] = entry

    facts = [e for e in entries if e.key is None]
    logs = [e for e in entries if e.key is not None]
    if any(e.key[0].upper() not in QUIET_LEVELS for e in logs):  # type: ignore[index]
        logs = [e for e in logs if e.key[0].upper() not in QUIET_LEVELS]  # type: ignore[index]
    room = max(max_entries - len(facts), 4)
    kept = {id(e) for e in facts} | {id(e) for e in logs[:room]}

    lines = []
    for entry in entries:
        if id(entry) not in kept:
            continue
        suffix = f" (x{entry.repeats})" if entry.repeats > 1 else ""
        stamp = iso(entry.item.timestamp)  # type: ignore[arg-type] - dated above
        lines.append(f"{stamp} - {entry.label}{suffix} [{entry.item.id}]")
    return lines
