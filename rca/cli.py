"""Command line entry point.

    python -m rca.cli investigate INC-1042
    python -m rca.cli investigate --title "..." --description "..."
    python -m rca.cli incidents
    python -m rca.cli sources
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console

from rca.agent import RCAAgent, RCAError
from rca.config import settings
from rca.llm import build_client
from rca.models import IncidentReport, RCAReport, parse_ts, utcnow
from rca.report import print_console, to_markdown
from rca.sources import SourceBundle, build_sources

app = typer.Typer(
    add_completion=False,
    help="Automated root cause analysis for incidents.",
    no_args_is_help=True,
)
console = Console()


def load_incident(incident_id: str) -> IncidentReport:
    path = settings.data_dir / "incidents" / f"{incident_id}.json"
    if not path.exists():
        raise typer.BadParameter(
            f"No incident file at {path}. Run `python scripts/seed_demo.py` first, "
            "or pass --title/--description."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return IncidentReport(
        id=raw["id"],
        title=raw["title"],
        description=raw["description"],
        reported_at=parse_ts(raw["reported_at"]),
        reporter=raw.get("reporter", "unknown"),
        service_hint=raw.get("service_hint"),
        severity_hint=raw.get("severity_hint"),
    )


def make_event_printer(verbose: bool):
    icons = {"phase": "==>", "tool": "   ->", "detail": "    .", "warn": "  (!)"}
    styles = {"phase": "bold cyan", "tool": "dim", "detail": "dim", "warn": "yellow"}

    def on_event(kind: str, message: str) -> None:
        if kind == "tool" and not verbose:
            return
        console.print(f"[{styles.get(kind, '')}]{icons.get(kind, '   ')} {message}[/]")

    return on_event


@app.command()
def sources() -> None:
    """Show which evidence sources are wired up and what they can see."""
    bundle = build_sources()
    console.print(f"[bold]Sources:[/bold] {bundle.describe()}")
    console.print(f"  data dir: {settings.data_dir}")
    services = bundle.logs.services()
    console.print(f"  services with logs: {', '.join(services) or '(none)'}")
    now = utcnow()
    from datetime import timedelta

    day_ago = now - timedelta(days=1)
    console.print(f"  log lines (last 24h): {bundle.logs.count(start=day_ago, end=now)}")
    console.print(
        f"  changes (last 24h):   {len(bundle.changes.list_changes(start=day_ago, end=now))}"
    )
    console.print(
        f"  alerts (last 24h):    {len(bundle.alerts.list_alerts(start=day_ago, end=now))}"
    )
    console.print()
    console.print(f"  model:    [bold]{settings.model}[/bold]")
    console.print(f"  server:   {settings.llm_base_url}")
    console.print(f"  strategy: {settings.strategy}")
    reachable, detail = build_client(settings).health()
    mark = "[green]reachable[/green]" if reachable else "[red]unreachable[/red]"
    console.print(f"  status:   {mark} - {detail}")
    if not reachable:
        console.print()
        console.print("[yellow]No model server.[/yellow] Start one, then re-run:")
        console.print("  local dev : ollama serve")
        console.print("              RCA_BASE_URL=http://localhost:11434/v1")
        console.print("  production: vllm serve Qwen/Qwen3-32B")
        console.print("              RCA_BASE_URL=http://<gpu-host>:8000/v1")


@app.command()
def incidents() -> None:
    """List the incident files available to investigate."""
    folder = settings.data_dir / "incidents"
    if not folder.exists():
        console.print("[yellow]No incidents directory. Run scripts/seed_demo.py.[/yellow]")
        raise typer.Exit(1)
    found = sorted(folder.glob("*.json"))
    if not found:
        console.print("[yellow]No incident files found.[/yellow]")
        raise typer.Exit(1)
    for path in found:
        raw = json.loads(path.read_text(encoding="utf-8"))
        console.print(f"  [bold]{raw['id']}[/bold]  {raw['title']}")
        console.print(f"          [dim]filed {raw['reported_at']} by {raw.get('reporter','?')}[/dim]")


@app.command()
def investigate(
    incident_id: str = typer.Argument("", help="Incident ID from data/incidents/."),
    title: str = typer.Option("", help="Title, if not loading from a file."),
    description: str = typer.Option("", help="Free-text report, if not loading from a file."),
    reporter: str = typer.Option("cli", help="Who filed it."),
    service_hint: str = typer.Option("", help="Optional service the symptom was seen on."),
    out: Path = typer.Option(None, help="Where to write the Markdown report."),
    json_out: Path = typer.Option(None, help="Also write the full report as JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show every tool call."),
) -> None:
    """Run the full triage -> investigate -> synthesise pipeline on one incident."""
    if incident_id:
        incident = load_incident(incident_id)
    elif title and description:
        incident = IncidentReport(
            id=f"AD-HOC-{utcnow():%Y%m%d%H%M%S}",
            title=title,
            description=description,
            reporter=reporter,
            service_hint=service_hint or None,
        )
    else:
        raise typer.BadParameter(
            "Give an incident ID, or both --title and --description."
        )

    console.print(f"[bold]{incident.id}[/bold] - {incident.title}")
    console.print()

    agent = RCAAgent(build_sources(), on_event=make_event_printer(verbose))
    try:
        report = agent.investigate(incident)
    except RCAError as exc:
        console.print(f"[red]RCA failed:[/red] {exc}")
        raise typer.Exit(2) from exc
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        raise typer.Exit(1) from exc

    print_console(report)

    md_path = out or (settings.reports_dir / f"{incident.id}.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(to_markdown(report), encoding="utf-8")
    console.print()
    console.print(f"[green]Markdown report:[/green] {md_path}")

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]JSON report:[/green]     {json_out}")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        console.print("\n[dim]interrupted[/dim]")
        sys.exit(130)


if __name__ == "__main__":
    main()
