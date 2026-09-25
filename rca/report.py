"""Render an RCAReport for humans: Markdown for the ticket, Rich for the terminal."""

from __future__ import annotations

from rca.models import RCAReport, iso

CONFIDENCE_BANDS = [
    (0.85, "very likely"),
    (0.65, "likely"),
    (0.40, "possible"),
    (0.0, "speculative"),
]


def band(confidence: float) -> str:
    for threshold, label in CONFIDENCE_BANDS:
        if confidence >= threshold:
            return label
    return "speculative"


def _bar(confidence: float, width: int = 20) -> str:
    filled = int(round(confidence * width))
    return "#" * filled + "." * (width - filled)


def to_markdown(report: RCAReport) -> str:
    a = report.analysis
    out: list[str] = []

    out.append(f"# RCA: {report.incident.id} - {report.incident.title}")
    out.append("")
    out.append(f"> **{a.headline}**")
    out.append("")
    out.append(
        f"*Generated {iso(report.generated_at)} by `{report.model}` in "
        f"{report.duration_seconds}s. Automated hypothesis - verify before acting.*"
    )
    out.append("")

    out.append("## Reported")
    out.append("")
    out.append(f"- **Filed:** {iso(report.incident.reported_at)} by {report.incident.reporter}")
    out.append(f"- **Triaged severity:** {report.triage.severity}")
    out.append(
        f"- **Services investigated:** {', '.join(report.triage.affected_services) or '-'}"
    )
    out.append("")
    out.append(f"{report.incident.description.strip()}")
    out.append("")

    if a.timeline:
        out.append("## Timeline")
        out.append("")
        for entry in a.timeline:
            out.append(f"- {entry}")
        out.append("")

    out.append("## Hypotheses")
    out.append("")
    if not a.hypotheses:
        out.append("_No hypothesis could be formed from the available evidence._")
        out.append("")
    for i, h in enumerate(a.hypotheses, 1):
        marker = " **<- most likely**" if i == 1 else ""
        out.append(f"### {i}. {h.statement}{marker}")
        out.append("")
        out.append(
            f"`{_bar(h.confidence)}` **{h.confidence:.0%}** ({band(h.confidence)}) "
            f"- category: `{h.category}`"
        )
        out.append("")
        out.append(h.reasoning)
        out.append("")
        if h.suspect_change:
            out.append(f"- **Suspect change:** `{h.suspect_change}`")
        if h.supporting_evidence:
            out.append(f"- **Supported by:** {', '.join(h.supporting_evidence)}")
        if h.contradicting_evidence:
            out.append(f"- **Argued against by:** {', '.join(h.contradicting_evidence)}")
        if h.verification_steps:
            out.append("- **To confirm or kill this:**")
            for step in h.verification_steps:
                out.append(f"  - {step}")
        out.append("")

    out.append("## Blast radius")
    out.append("")
    out.append(a.blast_radius or "_Not established._")
    out.append("")

    out.append("## Do this now")
    out.append("")
    if a.immediate_actions:
        for action in a.immediate_actions:
            out.append(f"- [ ] {action}")
    else:
        out.append("_No immediate action identified._")
    out.append("")

    if a.evidence_gaps:
        out.append("## What was not checked")
        out.append("")
        for gap in a.evidence_gaps:
            out.append(f"- {gap}")
        out.append("")

    out.append("## Evidence")
    out.append("")
    for kind, title in (("change", "Code changes"), ("alert", "Alerts"), ("log", "Log lines")):
        items = [e for e in report.evidence if e.kind == kind]
        if not items:
            continue
        out.append(f"### {title}")
        out.append("")
        out.append("```")
        for item in items:
            out.append(item.one_line())
        out.append("```")
        out.append("")

    out.append("<details>")
    out.append("<summary>Investigation trail</summary>")
    out.append("")
    out.append("```")
    for i, call in enumerate(report.investigation_log, 1):
        out.append(f"{i:>2}. {call}")
    out.append("```")
    out.append("")
    out.append(
        f"Tokens: {report.usage.input_tokens} in / {report.usage.output_tokens} out"
        + (
            f" ({report.usage.cache_read_input_tokens} cached)"
            if report.usage.cache_read_input_tokens
            else ""
        )
    )
    out.append("")
    out.append("</details>")
    out.append("")

    return "\n".join(out)


def print_console(report: RCAReport) -> None:
    """Terminal rendering. Falls back to plain text if `rich` is unavailable."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:  # pragma: no cover
        print(to_markdown(report))
        return

    console = Console()
    a = report.analysis

    console.print()
    console.print(
        Panel(
            f"[bold]{a.headline}[/bold]",
            title=f"[bold]{report.incident.id}[/bold] - {report.incident.title}",
            subtitle=f"{report.model} | {report.duration_seconds}s | "
            f"{len(report.evidence)} evidence items",
            border_style="red" if report.triage.severity in {"SEV1", "SEV2"} else "yellow",
        )
    )

    if a.timeline:
        console.print()
        console.print("[bold]Timeline[/bold]")
        for entry in a.timeline:
            console.print(f"  [dim]|[/dim] {entry}")

    console.print()
    console.print("[bold]Hypotheses[/bold]")
    for i, h in enumerate(a.hypotheses, 1):
        colour = "green" if h.confidence >= 0.85 else "yellow" if h.confidence >= 0.5 else "dim"
        console.print()
        console.print(f"  [bold]{i}.[/bold] {h.statement}")
        console.print(
            f"     [{colour}]{_bar(h.confidence)}[/{colour}] "
            f"[bold]{h.confidence:.0%}[/bold] [dim]{band(h.confidence)} | {h.category}[/dim]"
        )
        console.print(f"     [dim]{h.reasoning}[/dim]")
        cites = ", ".join(h.supporting_evidence) or "-"
        console.print(f"     [dim]evidence:[/dim] {cites}")
        if h.contradicting_evidence:
            console.print(f"     [dim]against:[/dim]  {', '.join(h.contradicting_evidence)}")

    if a.immediate_actions:
        console.print()
        table = Table(title="Do this now", show_header=False, title_justify="left")
        table.add_column(style="bold")
        for action in a.immediate_actions:
            table.add_row(f"[ ] {action}")
        console.print(table)

    if a.evidence_gaps:
        console.print()
        console.print("[bold]Not checked[/bold]")
        for gap in a.evidence_gaps:
            console.print(f"  [dim]-[/dim] {gap}")

    console.print()
    console.print(
        f"[dim]tokens: {report.usage.input_tokens} in / {report.usage.output_tokens} out | "
        f"tool calls: {len(report.investigation_log)}[/dim]"
    )
