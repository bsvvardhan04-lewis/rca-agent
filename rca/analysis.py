"""Facts the report writer should not have to work out for itself.

The synthesis step is a language model, and a small one will happily skip two
things a human on-call engineer would never skip: the arithmetic of *when* each
candidate change landed relative to the first errors, and the exact wording of
the errors. Both are plain computation over evidence already retrieved. Doing
them here puts them in front of the model as given facts, so it spends its
effort on judgement - which candidate fits - rather than on guessing.

Nothing in this module decides what caused an incident. It lays out the
candidates and the timing, and leaves the verdict to the model (and, in the
report, to the engineer).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from rca.models import CodeChange, iso
from rca.timeline import LOG, VOLATILE

DESCRIPTION_CHARS = 260
QUIET = {"INFO", "DEBUG", "TRACE"}


def _parsed_logs(summaries: list[str]) -> list[tuple[str, str, str]]:
    """(level, service, message) for every ledger log summary that parses."""
    out = []
    for summary in summaries:
        m = LOG.match(summary)
        if m:
            out.append((m["level"].upper(), m["service"], m["message"]))
    return out


def error_services(log_summaries: list[str]) -> list[str]:
    """Services that logged a warning or an error, most frequent first."""
    counts = Counter(svc for level, svc, _ in _parsed_logs(log_summaries) if level not in QUIET)
    return [svc for svc, _ in counts.most_common()]


def dominant_errors(log_summaries: list[str], limit: int = 4) -> list[str]:
    """The distinct warning/error messages seen, with how often, most common first.

    Messages differing only in numbers or ids count as one ("waited 5001ms" and
    "waited 5012ms" are the same error).
    """
    shapes: Counter[tuple[str, str, str]] = Counter()
    example: dict[tuple[str, str, str], str] = {}
    for level, svc, message in _parsed_logs(log_summaries):
        if level in QUIET:
            continue
        key = (level, svc, VOLATILE.sub("#", message))
        shapes[key] += 1
        example.setdefault(key, message)
    lines = []
    for key, n in shapes.most_common(limit):
        level, svc, _ = key
        text = " ".join(example[key].split())
        text = text if len(text) <= 160 else text[:157] + "..."
        lines.append(f"x{n}  {level} [{svc}]: {text}")
    return lines


def _relative(change_time: datetime, onset: datetime | None) -> str:
    if onset is None:
        return "onset not established"
    minutes = round((onset - change_time).total_seconds() / 60)
    if minutes > 0:
        return f"{minutes} min BEFORE the first errors"
    if minutes < 0:
        return f"{-minutes} min AFTER the first errors began (cannot have caused them)"
    return "at the moment the errors began"


def describe_change(
    change: CodeChange,
    onset: datetime | None,
    err_services: list[str],
    evidence_id: str | None = None,
) -> str:
    when = change.deployed_at or change.merged_at
    stage = "deployed" if change.deployed_at else "merged, no deploy time recorded"
    if change.services:
        overlap = set(change.services) & set(err_services)
        if not err_services:
            svc_note = f"services {', '.join(change.services)}"
        elif overlap:
            svc_note = f"services {', '.join(change.services)} (these also logged errors)"
        else:
            svc_note = f"services {', '.join(change.services)} (none of these logged errors)"
    else:
        svc_note = "services unknown"
    desc = " ".join(change.description.split())
    desc = desc if len(desc) <= DESCRIPTION_CHARS else desc[: DESCRIPTION_CHARS - 3] + "..."
    tag = f"[{evidence_id}] " if evidence_id else ""
    return (
        f"{tag}{change.id} \"{change.title}\" by {change.author}: {stage} {iso(when)}, "
        f"{_relative(when, onset)}; {svc_note}."
        + (f" Description: {desc}" if desc else "")
    )


def change_table(
    changes: list[CodeChange],
    onset: datetime | None,
    err_services: list[str],
    ids: dict[str, str] | None = None,
) -> list[str]:
    """One line per candidate change, newest deploy first."""
    ids = ids or {}
    ordered = sorted(changes, key=lambda c: c.deployed_at or c.merged_at, reverse=True)
    return [describe_change(c, onset, err_services, ids.get(c.id)) for c in ordered]
