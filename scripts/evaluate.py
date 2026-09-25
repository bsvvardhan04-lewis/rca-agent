"""Run every scenario and score the reports against ground truth.

    python scripts/seed_demo.py --all
    python scripts/evaluate.py
    python scripts/evaluate.py --strategy guided --model qwen2.5:7b

Results are written to `reports/eval-<timestamp>.json` so two runs can be
compared after a prompt change. Without that, "the prompt feels better now" is
the only evidence available, which is not evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from rca.agent import RCAAgent  # noqa: E402
from rca.config import Settings  # noqa: E402
from rca.evaluation import ScoreCard, score_report, summarise  # noqa: E402
from rca.llm import build_client  # noqa: E402
from rca.models import IncidentReport, parse_ts  # noqa: E402
from rca.report import to_markdown  # noqa: E402
from rca.sources import build_sources  # noqa: E402

console = Console()
SCENARIO_ROOT = ROOT / "data" / "scenarios"


def load_incident(folder: Path) -> IncidentReport:
    files = list((folder / "incidents").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No incident file under {folder}")
    raw = json.loads(files[0].read_text(encoding="utf-8"))
    return IncidentReport(
        id=raw["id"],
        title=raw["title"],
        description=raw["description"],
        reported_at=parse_ts(raw["reported_at"]),
        reporter=raw.get("reporter", "unknown"),
        service_hint=raw.get("service_hint"),
        severity_hint=raw.get("severity_hint"),
    )


def run_one(folder: Path, settings: Settings, verbose: bool) -> tuple[ScoreCard, str]:
    incident = load_incident(folder)
    truth = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
    scoped = replace(settings, data_dir=folder)

    def on_event(kind: str, message: str) -> None:
        if verbose or kind in {"phase", "warn"}:
            console.print(f"    [dim]{message}[/dim]")

    console.print(f"\n[bold]{incident.id}[/bold] - {incident.title}")
    try:
        agent = RCAAgent(build_sources(scoped), settings=scoped, on_event=on_event)
        report = agent.investigate(incident)
    except Exception as exc:  # noqa: BLE001 - one bad scenario must not stop the sweep
        console.print(f"    [red]{type(exc).__name__}: {exc}[/red]")
        card = ScoreCard(incident_id=incident.id, error=f"{type(exc).__name__}: {exc}")
        return card, ""

    card = score_report(report, truth)
    mark = "[green]PASS[/green]" if card.passed else "[red]FAIL[/red]"
    console.print(f"    {mark}  score {card.score:.0%}  ({card.duration_seconds}s)")
    for check in card.failures():
        console.print(f"      [yellow]x {check.name}:[/yellow] {check.detail}")
    return card, to_markdown(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", help="Scenario IDs. Default: all of them.")
    parser.add_argument("--strategy", default="", help="agentic | guided | auto")
    parser.add_argument("--model", default="", help="Override RCA_MODEL for this run.")
    parser.add_argument("--base-url", default="", help="Override RCA_BASE_URL.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show every tool call.")
    parser.add_argument("--save-reports", action="store_true", help="Write each Markdown report.")
    args = parser.parse_args()

    settings = Settings()
    if args.strategy:
        settings = replace(settings, strategy=args.strategy)
    if args.model:
        settings = replace(settings, model=args.model)
    if args.base_url:
        settings = replace(settings, llm_base_url=args.base_url)

    reachable, detail = build_client(settings).health()
    console.print(f"[bold]model:[/bold] {settings.model}  [bold]strategy:[/bold] {settings.strategy}")
    console.print(f"[bold]server:[/bold] {detail}")
    if not reachable:
        console.print("\n[red]No model server reachable - nothing to evaluate.[/red]")
        raise SystemExit(1)

    if not SCENARIO_ROOT.exists():
        console.print("\n[red]No scenarios.[/red] Run: python scripts/seed_demo.py --all")
        raise SystemExit(1)

    folders = sorted(p for p in SCENARIO_ROOT.iterdir() if p.is_dir())
    if args.scenarios:
        wanted = {s.upper() for s in args.scenarios}
        folders = [f for f in folders if f.name.upper() in wanted]
    if not folders:
        console.print("[red]No matching scenarios.[/red]")
        raise SystemExit(1)

    cards: list[ScoreCard] = []
    for folder in folders:
        card, markdown = run_one(folder, settings, args.verbose)
        cards.append(card)
        if args.save_reports and markdown:
            out = settings.reports_dir / f"eval-{card.incident_id}.md"
            out.write_text(markdown, encoding="utf-8")

    summary = summarise(cards)

    table = Table(title="\nResults", title_justify="left")
    table.add_column("incident")
    table.add_column("result")
    table.add_column("score", justify="right")
    table.add_column("time", justify="right")
    table.add_column("tools", justify="right")
    table.add_column("failed checks")
    for card in cards:
        table.add_row(
            card.incident_id,
            "[green]PASS[/green]" if card.passed else "[red]FAIL[/red]",
            f"{card.score:.0%}",
            f"{card.duration_seconds:.0f}s",
            str(card.tool_calls),
            ", ".join(c.name for c in card.failures()) or "-",
        )
    console.print(table)

    console.print(
        f"\n[bold]{summary['passed']}/{summary['runs']} passed[/bold]  "
        f"mean score {summary['mean_score']:.0%}  "
        f"mean time {summary['mean_duration']}s  "
        f"{summary['total_tokens']} tokens"
    )
    console.print("per-check: " + "  ".join(f"{k} {v}" for k, v in summary["per_check"].items()))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = settings.reports_dir / f"eval-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "model": settings.model,
                "strategy": settings.strategy,
                "timestamp": stamp,
                "summary": summary,
                "results": [c.to_dict() for c in cards],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"\n[green]saved:[/green] {out}")


if __name__ == "__main__":
    main()
