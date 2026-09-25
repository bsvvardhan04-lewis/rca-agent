from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import seed_demo  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

from rca.config import Settings  # noqa: E402
from rca.evidence import EvidenceLedger  # noqa: E402
from rca.models import TimeWindow  # noqa: E402
from rca.sources.base import SourceBundle  # noqa: E402
from rca.sources.file_sources import (  # noqa: E402
    FileAlertSource,
    FileChangeSource,
    FileLogSource,
)
from rca.tools import ToolContext  # noqa: E402

UTC = timezone.utc

# A fixed anchor keeps every assertion about timings deterministic.
ANCHOR = datetime(2026, 9, 23, 14, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The INC-1042 scenario, at a fixed anchor so timing assertions hold."""
    root = tmp_path_factory.mktemp("rca-data")
    seed_demo.write_scenario(SCENARIOS["INC-1042"].render(ANCHOR), root=root, quiet=True)
    return root


@pytest.fixture(scope="session")
def scenario_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Every scenario, each in its own directory, for eval-shaped tests."""
    base = tmp_path_factory.mktemp("rca-scenarios")
    out: dict[str, Path] = {}
    for scenario in SCENARIOS.values():
        folder = base / scenario.id
        seed_demo.write_scenario(scenario.render(ANCHOR), root=folder, quiet=True)
        out[scenario.id] = folder
    return out


@pytest.fixture()
def settings(data_dir: Path, tmp_path: Path) -> Settings:
    s = Settings()
    s.data_dir = data_dir
    s.reports_dir = tmp_path / "reports"
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture()
def sources(settings: Settings) -> SourceBundle:
    return SourceBundle(
        logs=FileLogSource(settings.logs_dir),
        changes=FileChangeSource(settings.changes_dir),
        alerts=FileAlertSource(settings.alerts_dir),
    )


@pytest.fixture()
def window() -> TimeWindow:
    # What a sane triage step would produce: 3h back, ending just after the report.
    return TimeWindow(start=ANCHOR - timedelta(hours=3), end=ANCHOR + timedelta(minutes=5))


@pytest.fixture()
def ctx(sources: SourceBundle, window: TimeWindow, settings: Settings) -> ToolContext:
    return ToolContext(
        sources=sources, ledger=EvidenceLedger(), window=window, settings=settings
    )
