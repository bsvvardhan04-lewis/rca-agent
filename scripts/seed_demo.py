"""Write incident scenarios to disk.

    python scripts/seed_demo.py                 # INC-1042 into data/ (the demo)
    python scripts/seed_demo.py --all           # every scenario into data/scenarios/
    python scripts/seed_demo.py INC-1043        # one scenario into data/

Scenario definitions and their ground truth live in `scripts/scenarios.py`.
Timestamps are anchored to the moment you run this, so the demo always looks
live; re-run it when the data goes stale.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scenarios import SCENARIOS, iso

UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def write_scenario(payload: dict, root: Path, quiet: bool = False) -> Path:
    root = Path(root)
    dirs = {name: root / name for name in ("logs", "alerts", "changes", "incidents")}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    total = 0
    for service, entries in payload["logs"].items():
        if not entries:
            continue
        entries.sort(key=lambda e: e["timestamp"])
        with (dirs["logs"] / f"{service}.jsonl").open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")
        total += len(entries)

    (dirs["alerts"] / "alerts.json").write_text(
        json.dumps(payload["alerts"], indent=2), encoding="utf-8"
    )
    (dirs["changes"] / "changes.json").write_text(
        json.dumps(payload["changes"], indent=2), encoding="utf-8"
    )

    incident = payload["incident"]
    (dirs["incidents"] / f"{incident['id']}.json").write_text(
        json.dumps(incident, indent=2), encoding="utf-8"
    )
    if "ground_truth" in payload:
        (root / "ground_truth.json").write_text(
            json.dumps(payload["ground_truth"], indent=2), encoding="utf-8"
        )

    if not quiet:
        services = sum(1 for e in payload["logs"].values() if e)
        print(
            f"  {incident['id']:<10} {total:>5} log lines / {services} services, "
            f"{len(payload['alerts'])} alerts, {len(payload['changes'])} changes  -> {root}"
        )
    return root


def resolve_anchor(raw: str) -> datetime:
    if not raw:
        now = datetime.now(UTC).replace(second=0, microsecond=0)
        return now - timedelta(minutes=20)
    anchor = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return anchor if anchor.tzinfo else anchor.replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        default="INC-1042",
        help=f"Which scenario to write to data/. One of: {', '.join(SCENARIOS)}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write every scenario to data/scenarios/<id>/ for the eval harness.",
    )
    parser.add_argument(
        "--anchor", default="", help="ISO-8601 UTC report time. Defaults to 20 minutes ago."
    )
    args = parser.parse_args()
    anchor = resolve_anchor(args.anchor)

    print(f"Seeding at {iso(anchor)}\n")

    if args.all:
        for scenario in SCENARIOS.values():
            write_scenario(
                scenario.render(anchor), DATA / "scenarios" / scenario.id
            )
        print(f"\n{len(SCENARIOS)} scenarios written. Next:  python scripts/evaluate.py")
        return

    if args.scenario not in SCENARIOS:
        parser.error(f"Unknown scenario {args.scenario!r}. Choose from: {', '.join(SCENARIOS)}")

    write_scenario(SCENARIOS[args.scenario].render(anchor), DATA)
    print(f"\nNext:  python -m rca.cli investigate {args.scenario}")


if __name__ == "__main__":
    main()
