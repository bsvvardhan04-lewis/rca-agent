"""The tools the investigator agent can call.

Each tool is a thin, boring wrapper over a source adapter. Two rules shape them:

1. **Return text, with evidence IDs attached.** The agent reads the same
   `[L7] 14:03:11 ERROR [payments-api] ...` line the human will read in the
   report, so citations cannot drift from the underlying record.
2. **Make counting cheaper than reading.** `log_volume` exists so the agent can
   locate the onset of a problem across 2,000 lines without pulling 2,000 lines
   into context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from rca.config import Settings
from rca.evidence import EvidenceLedger
from rca.llm import ToolSpec, tool
from rca.models import TimeWindow, iso, parse_ts
from rca.sources.base import SourceBundle

MAX_LOG_LIMIT = 80


@dataclass
class ToolContext:
    """Everything the tool closures need, in one object."""

    sources: SourceBundle
    ledger: EvidenceLedger
    window: TimeWindow
    settings: Settings
    call_log: list[str] = field(default_factory=list)

    def note(self, line: str) -> None:
        self.call_log.append(line)


def _split(csv: str) -> list[str] | None:
    """'a, b' -> ['a', 'b']; '' -> None (meaning 'no filter')."""
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    return parts or None


def _resolve_window(ctx: ToolContext, start: str, end: str) -> tuple[datetime, datetime]:
    """Blank start/end fall back to the triaged incident window."""
    try:
        s = parse_ts(start) if start.strip() else ctx.window.start
    except ValueError:
        s = ctx.window.start
    try:
        e = parse_ts(end) if end.strip() else ctx.window.end
    except ValueError:
        e = ctx.window.end
    if s > e:
        s, e = e, s
    return s, e


def build_tools(ctx: ToolContext) -> list[ToolSpec]:
    """Return the decorated tool functions bound to this investigation."""

    @tool
    def list_services() -> str:
        """List every service that has logs available, and the overall log time range.

        Call this first if you are unsure which service names are valid.
        """
        ctx.note("list_services()")
        services = ctx.sources.logs.services()
        if not services:
            return "No log sources are configured."
        return (
            "Services with logs: "
            + ", ".join(services)
            + f"\nDefault investigation window: {ctx.window.describe()}"
        )

    @tool
    def log_volume(
        query: str = "",
        services: str = "",
        levels: str = "",
        start: str = "",
        end: str = "",
        bucket_minutes: int = 5,
    ) -> str:
        """Count matching log lines per time bucket. Use this to find WHEN something started.

        Far cheaper than reading lines. A flat run of zeros followed by a spike
        tells you the onset time; then search_logs that narrow window.

        Args:
            query: Space-separated terms that must all appear. Use "double quotes" for a phrase. Empty matches everything.
            services: Comma-separated service names to include. Empty means all services.
            levels: Comma-separated log levels, e.g. "ERROR,WARN". Empty means all levels.
            start: ISO-8601 UTC start. Empty uses the incident window.
            end: ISO-8601 UTC end. Empty uses the incident window.
            bucket_minutes: Size of each bucket in minutes.
        """
        s, e = _resolve_window(ctx, start, end)
        ctx.note(
            f"log_volume(query={query!r}, services={services!r}, levels={levels!r}, "
            f"bucket={bucket_minutes}m)"
        )
        buckets = ctx.sources.logs.histogram(
            start=s,
            end=e,
            bucket_minutes=bucket_minutes,
            services=_split(services),
            levels=_split(levels),
            query=query,
        )
        if not buckets:
            return f"No buckets in {iso(s)} .. {iso(e)}."

        peak = max(count for _, count in buckets) or 1
        total = sum(count for _, count in buckets)
        lines = [
            f"Matching lines per {bucket_minutes}m bucket, {iso(s)} .. {iso(e)} "
            f"(total {total}):"
        ]
        for ts, count in buckets:
            bar = "#" * int(round(24 * count / peak)) if count else ""
            lines.append(f"  {iso(ts)}  {count:>5}  {bar}")
        return "\n".join(lines)

    @tool
    def search_logs(
        query: str = "",
        services: str = "",
        levels: str = "",
        start: str = "",
        end: str = "",
        limit: int = 15,
    ) -> str:
        """Return matching log lines, newest first, each tagged with an evidence ID.

        Cite those IDs (e.g. L4) when you state a conclusion. Keep `limit` small
        and narrow the window instead of pulling hundreds of lines.

        Args:
            query: Space-separated terms that must all appear. Use "double quotes" for a phrase. Empty matches everything.
            services: Comma-separated service names to include. Empty means all services.
            levels: Comma-separated log levels, e.g. "ERROR". Empty means all levels.
            start: ISO-8601 UTC start. Empty uses the incident window.
            end: ISO-8601 UTC end. Empty uses the incident window.
            limit: Maximum lines to return (capped at 80).
        """
        s, e = _resolve_window(ctx, start, end)
        capped = max(1, min(int(limit or 15), MAX_LOG_LIMIT, ctx.settings.max_log_lines))
        ctx.note(
            f"search_logs(query={query!r}, services={services!r}, levels={levels!r}, "
            f"limit={capped})"
        )
        svc = _split(services)
        lv = _split(levels)
        total = ctx.sources.logs.count(start=s, end=e, query=query, services=svc, levels=lv)
        hits = ctx.sources.logs.search(
            start=s, end=e, query=query, services=svc, levels=lv, limit=capped
        )
        if not hits:
            return (
                f"No log lines matched query={query!r} in {iso(s)} .. {iso(e)}. "
                "Try fewer terms, a wider window, or drop the level filter."
            )

        header = f"{total} line(s) matched; showing the {len(hits)} most recent:"
        lines = [header]
        for entry in hits:
            eid = ctx.ledger.add_log(entry)
            lines.append(f"  [{eid}] {entry.one_line()}")
            if entry.attributes:
                shown = {
                    k: v for k, v in entry.attributes.items() if k not in {"latency_ms", "status"}
                }
                if shown:
                    lines.append(f"        attrs: {shown}")
        if total > len(hits):
            lines.append(
                f"  ... {total - len(hits)} older matching line(s) not shown. "
                "Narrow the window or raise limit if you need them."
            )
        return "\n".join(lines)

    @tool
    def list_recent_changes(
        start: str = "",
        end: str = "",
        services: str = "",
        query: str = "",
        lookback_hours: int = 0,
    ) -> str:
        """List pull requests / deploys that landed in a time range, newest first.

        A change is included when it *reached production* in the range (or was
        merged there and has not deployed yet). Deploy time is what correlates
        with an incident, not merge time.

        Args:
            start: ISO-8601 UTC start. Empty uses the incident window.
            end: ISO-8601 UTC end. Empty uses the incident window.
            services: Comma-separated service names. Empty means all services.
            query: Terms that must appear in the title, description, author or file paths.
            lookback_hours: If > 0, ignore `start` and look back this many hours from `end`.
        """
        s, e = _resolve_window(ctx, start, end)
        if lookback_hours and lookback_hours > 0:
            s = e - timedelta(hours=lookback_hours)
        ctx.note(
            f"list_recent_changes(services={services!r}, query={query!r}, "
            f"window={iso(s)}..{iso(e)})"
        )
        changes = ctx.sources.changes.list_changes(
            start=s, end=e, services=_split(services), query=query
        )
        if not changes:
            return (
                f"No changes landed between {iso(s)} and {iso(e)}. "
                "Try lookback_hours to widen the search."
            )
        lines = [f"{len(changes)} change(s) landed in {iso(s)} .. {iso(e)}:"]
        unattributed = 0
        for change in changes:
            cid = ctx.ledger.add_change(change)
            flag = ""
            if services and not change.services:
                # It survived a service filter only because we could not tell
                # which service it belongs to. Say so rather than letting it
                # look like a confirmed match.
                flag = "  <- UNATTRIBUTED: no service known, may not be relevant"
                unattributed += 1
            lines.append(f"  [{cid}] {change.one_line()}{flag}")
        if unattributed:
            lines.append(
                f"{unattributed} change(s) could not be attributed to a service and are "
                "listed anyway - a shared config or library can break any service."
            )
        lines.append("Use get_change_detail(<id>) for the file list and description.")
        return "\n".join(lines)

    @tool
    def get_change_detail(change_id: str) -> str:
        """Full detail for one change: description, files touched, timings.

        Args:
            change_id: The change ID as shown by list_recent_changes, e.g. "PR-4821".
        """
        ctx.note(f"get_change_detail({change_id!r})")
        change = ctx.sources.changes.get_change(change_id)
        if change is None:
            return f"No change found with id {change_id!r}."
        cid = ctx.ledger.add_change(change)
        deployed = iso(change.deployed_at) if change.deployed_at else "NOT DEPLOYED"
        lines = [
            f"[{cid}] {change.id}: {change.title}",
            f"  author:   {change.author}",
            f"  merged:   {iso(change.merged_at)}",
            f"  deployed: {deployed}",
            f"  services: {', '.join(change.services) or '(none declared)'}",
            f"  size:     +{change.additions} / -{change.deletions} "
            f"across {len(change.files_changed)} file(s)",
        ]
        if change.url:
            lines.append(f"  url:      {change.url}")
        if change.description:
            lines.append(f"  description: {change.description}")
        if change.files_changed:
            lines.append("  files:")
            lines.extend(f"    - {path}" for path in change.files_changed)
        return "\n".join(lines)

    @tool
    def list_alerts(
        start: str = "",
        end: str = "",
        services: str = "",
        include_resolved: bool = True,
    ) -> str:
        """List monitoring alerts that were firing during a time range.

        Alert *fire order* is a strong signal for causality: the service whose
        alert fired first is usually nearer the cause than the ones that
        followed it.

        Args:
            start: ISO-8601 UTC start. Empty uses the incident window.
            end: ISO-8601 UTC end. Empty uses the incident window.
            services: Comma-separated service names. Empty means all services.
            include_resolved: Whether to include alerts that have already resolved.
        """
        s, e = _resolve_window(ctx, start, end)
        ctx.note(f"list_alerts(services={services!r}, window={iso(s)}..{iso(e)})")
        alerts = ctx.sources.alerts.list_alerts(
            start=s, end=e, services=_split(services), include_resolved=include_resolved
        )
        if not alerts:
            return f"No alerts were firing between {iso(s)} and {iso(e)}."
        ordered = sorted(alerts, key=lambda a: a.fired_at)
        lines = [f"{len(ordered)} alert(s), oldest first (fire order matters):"]
        for alert in ordered:
            aid = ctx.ledger.add_alert(alert)
            lines.append(f"  [{aid}] {alert.one_line()}")
            if alert.labels:
                lines.append(f"        labels: {alert.labels}")
        return "\n".join(lines)

    return [
        list_services,
        log_volume,
        search_logs,
        list_recent_changes,
        get_change_detail,
        list_alerts,
    ]
