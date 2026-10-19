"""The baseline: one model call, no tools, scored by the same scorer.

    python scripts/baseline_single_call.py

"Why an agent and not a single LLM call?" deserves an answer from a run, not an
assertion. This gives the same model the incident report and nothing else - no
logs, no changes, no alerts - and asks for the same structured analysis the full
pipeline produces. Each result goes through `score_report`, so the two are
compared on identical checks.

The prompt is deliberately plain. A baseline whose prompt was tuned to fail would
prove nothing; this is what a developer would write on a first attempt.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rich.console import Console  # noqa: E402

from evaluate import SCENARIO_ROOT, load_incident  # noqa: E402
from rca.config import Settings  # noqa: E402
from rca.evaluation import ScoreCard, score_report, summarise  # noqa: E402
from rca.llm import Usage, build_client, complete_structured  # noqa: E402
from rca.models import RCAAnalysis, RCAReport, TokenUsage, TriagedIncident, iso  # noqa: E402

console = Console()

SYSTEM = (
    "You are an experienced on-call engineer. Given an incident report, write a "
    "root cause analysis: a headline, a timeline, ranked hypotheses with a "
    "category, confidence, reasoning and verification steps, the blast radius, "
    "immediate actions and evidence gaps."
)


def run_one(folder: Path, settings: Settings) -> ScoreCard:
    incident = load_incident(folder)
    truth = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
    client = build_client(settings)
    usage = Usage()

    console.print(f"\n[bold]{incident.id}[/bold] - {incident.title}")
    started = time.monotonic()
    try:
        analysis = complete_structured(
            client,
            system=SYSTEM,
            user=incident.as_prompt(),
            model_cls=RCAAnalysis,
            usage=usage,
            max_tokens=settings.max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - one bad scenario must not stop the sweep
        console.print(f"    [red]{type(exc).__name__}: {exc}[/red]")
        return ScoreCard(incident_id=incident.id, error=f"{type(exc).__name__}: {exc}")

    report = RCAReport(
        incident=incident,
        triage=TriagedIncident(
            summary=incident.title,
            affected_services=[],
            symptoms=[],
            error_signatures=[],
            search_keywords=[],
            window_start=iso(incident.reported_at),
            window_end=iso(incident.reported_at),
            severity="SEV3",
            triage_notes="Baseline: no triage step.",
        ),
        analysis=analysis,
        evidence=[],
        investigation_log=[],
        model=settings.model,
        usage=TokenUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens),
        duration_seconds=round(time.monotonic() - started, 1),
    )
    card = score_report(report, truth)
    mark = "[green]PASS[/green]" if card.passed else "[red]FAIL[/red]"
    console.print(f"    {mark}  score {card.score:.0%}  ({card.duration_seconds}s)")
    for check in card.failures():
        console.print(f"      [yellow]x {check.name}:[/yellow] {check.detail}")
    return card


def main() -> None:
    settings = replace(Settings(), strategy="auto")
    reachable, detail = build_client(settings).health()
    console.print(f"[bold]baseline: single call, no tools[/bold]  model: {settings.model}")
    console.print(f"[bold]server:[/bold] {detail}")
    if not reachable:
        raise SystemExit("No model server reachable.")

    folders = sorted(p for p in SCENARIO_ROOT.iterdir() if p.is_dir())
    cards = [run_one(folder, settings) for folder in folders]
    summary = summarise(cards)
    console.print(
        f"\n[bold]{summary['passed']}/{summary['runs']} passed[/bold]  "
        f"mean score {summary['mean_score']:.0%}  mean time {summary['mean_duration']}s"
    )
    console.print("per-check: " + "  ".join(f"{k} {v}" for k, v in summary["per_check"].items()))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = settings.reports_dir / f"eval-baseline-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "model": settings.model,
                "strategy": "baseline-single-call",
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
