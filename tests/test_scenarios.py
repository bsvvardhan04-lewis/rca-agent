"""Structural tests for the scenarios themselves.

A scenario that quietly stops being hard makes the eval meaningless while still
reporting green. So these assert the properties each scenario is *supposed* to
have - above all, that the three no-deploy cases really do have no deploy near
the onset, so "blame the most recent change" genuinely fails on them.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from rca.models import parse_ts
from rca.sources.file_sources import FileAlertSource, FileChangeSource, FileLogSource
from scenarios import SCENARIOS

ALL_IDS = sorted(SCENARIOS)
NO_DEPLOY_IDS = [
    sid for sid, s in SCENARIOS.items() if s.ground_truth.expect_no_recent_change
]


def sources(folder: Path):
    return (
        FileLogSource(folder / "logs"),
        FileChangeSource(folder / "changes"),
        FileAlertSource(folder / "alerts"),
    )


def onset_of(folder: Path):
    """First bucket with a meaningful number of errors."""
    logs, _, _ = sources(folder)
    entries = [e for e in logs._all() if e.level == "ERROR"]
    return min((e.timestamp for e in entries), default=None)


def test_every_scenario_is_registered_with_ground_truth():
    assert len(SCENARIOS) >= 4
    for scenario in SCENARIOS.values():
        truth = scenario.ground_truth
        assert truth.category, f"{scenario.id} has no category"
        assert truth.notes, f"{scenario.id} has no explanation of the answer"
        if not truth.expect_no_recent_change:
            assert truth.root_cause_id, f"{scenario.id} must name a root cause"


@pytest.mark.parametrize("sid", ALL_IDS)
def test_each_scenario_writes_usable_data(sid, scenario_dirs):
    folder = scenario_dirs[sid]
    logs, changes, alerts = sources(folder)

    assert logs.services(), f"{sid} produced no logs"
    assert onset_of(folder) is not None, f"{sid} has no ERROR lines to find"
    assert alerts._all(), f"{sid} has no alerts"
    assert changes._all(), f"{sid} has no changes at all - even a decoy is needed"

    incident = json.loads(
        (folder / "incidents" / f"{sid}.json").read_text(encoding="utf-8")
    )
    assert incident["description"].strip()
    assert (folder / "ground_truth.json").exists()


@pytest.mark.parametrize("sid", ALL_IDS)
def test_decoys_named_in_ground_truth_actually_exist(sid, scenario_dirs):
    _, changes, _ = sources(scenario_dirs[sid])
    for decoy in SCENARIOS[sid].ground_truth.must_rule_out:
        assert changes.get_change(decoy) is not None, (
            f"{sid} ground truth says to rule out {decoy}, but no such change exists"
        )


@pytest.mark.parametrize("sid", ALL_IDS)
def test_the_signal_the_answer_depends_on_is_present_in_the_logs(sid, scenario_dirs):
    """must_mention terms have to be discoverable, or the check is unfair."""
    logs, _, _ = sources(scenario_dirs[sid])
    blob = " ".join(e.message.lower() for e in logs._all())
    for term in SCENARIOS[sid].ground_truth.must_mention:
        assert term.lower() in blob, f"{sid}: '{term}' appears nowhere in the logs"


@pytest.mark.parametrize("sid", NO_DEPLOY_IDS)
def test_no_deploy_scenarios_have_no_deploy_near_the_onset(sid, scenario_dirs):
    """The whole point: 'blame the newest deploy' must be WRONG here.

    If a change landed shortly before the errors started, the shortcut would
    accidentally score correct and the scenario would stop testing anything.
    """
    folder = scenario_dirs[sid]
    _, changes, _ = sources(folder)
    onset = onset_of(folder)
    assert onset is not None

    recent = [
        c
        for c in changes._all()
        if c.deployed_at and onset - timedelta(minutes=45) <= c.deployed_at <= onset
    ]
    assert not recent, (
        f"{sid} has {[c.id for c in recent]} deployed within 45 min of onset - "
        "the 'blame the newest deploy' shortcut would score correct by accident"
    )


def test_the_deploy_caused_scenario_does_have_a_deploy_before_onset():
    """The positive control: INC-1042 must reward correct deploy correlation."""
    from tests.conftest import ANCHOR

    payload = SCENARIOS["INC-1042"].render(ANCHOR)
    cause = next(c for c in payload["changes"] if c["id"] == "PR-4821")
    deployed = parse_ts(cause["deployed_at"])

    errors = [
        parse_ts(e["timestamp"])
        for e in payload["logs"]["payments-api"]
        if e["level"] == "ERROR"
    ]
    assert deployed < min(errors), "the cause must deploy before the errors start"
    assert min(errors) - deployed < timedelta(minutes=30), "and shortly before"


def test_scenarios_cover_distinct_root_cause_categories():
    """Four scenarios that are all code_change would test one thing four times."""
    categories = {s.ground_truth.category for s in SCENARIOS.values()}
    assert len(categories) >= 4, f"only {len(categories)} distinct categories: {categories}"


def test_rendering_is_deterministic():
    from tests.conftest import ANCHOR

    first = SCENARIOS["INC-1042"].render(ANCHOR)
    second = SCENARIOS["INC-1042"].render(ANCHOR)
    assert first["logs"]["payments-api"] == second["logs"]["payments-api"], (
        "a scenario that changes between runs makes eval deltas meaningless"
    )
