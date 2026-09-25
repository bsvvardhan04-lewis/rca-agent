"""The adapters have to be right, or every hypothesis above them is built on sand."""

from __future__ import annotations

from datetime import timedelta

from tests.conftest import ANCHOR


def test_all_services_are_discoverable(sources):
    assert set(sources.logs.services()) == {
        "payments-api",
        "checkout-web",
        "ledger-service",
        "notifications",
    }


def test_window_filter_excludes_lines_outside_it(sources):
    early = sources.logs.count(
        start=ANCHOR - timedelta(hours=6), end=ANCHOR - timedelta(hours=5)
    )
    assert early == 0, "nothing was seeded that long before the incident"

    during = sources.logs.count(start=ANCHOR - timedelta(minutes=20), end=ANCHOR)
    assert during > 0


def test_phrase_query_is_treated_as_one_unit(sources):
    window = {"start": ANCHOR - timedelta(hours=3), "end": ANCHOR}
    phrase = sources.logs.count(query='"pool exhausted"', **window)
    # The same two words, unquoted, also match lines where they are far apart.
    loose = sources.logs.count(query="pool exhausted", **window)
    assert phrase == 46
    assert loose >= phrase


def test_histogram_localises_the_onset(sources):
    buckets = sources.logs.histogram(
        start=ANCHOR - timedelta(hours=3),
        end=ANCHOR,
        bucket_minutes=5,
        levels=["ERROR"],
    )
    non_empty = [(ts, count) for ts, count in buckets if count]
    assert non_empty, "expected an error spike"

    first_error_bucket = non_empty[0][0]
    # Errors start ~15 minutes before the report, not at the top of the window.
    assert ANCHOR - timedelta(minutes=25) <= first_error_bucket <= ANCHOR

    quiet_before = [c for ts, c in buckets if ts < ANCHOR - timedelta(minutes=30)]
    assert sum(quiet_before) == 0, "the hours before the deploy should be clean"


def test_changes_are_matched_on_deploy_time_not_merge_time(sources):
    # PR-4817 merged at T-130 and deployed at T-110. A window that contains the
    # merge but not the deploy must not return it.
    merged_only = sources.changes.list_changes(
        start=ANCHOR - timedelta(minutes=135), end=ANCHOR - timedelta(minutes=120)
    )
    assert [c.id for c in merged_only] == []

    deployed = sources.changes.list_changes(
        start=ANCHOR - timedelta(minutes=115), end=ANCHOR - timedelta(minutes=105)
    )
    assert [c.id for c in deployed] == ["PR-4817"]


def test_undeployed_change_falls_back_to_merge_time(sources):
    # PR-4822 never deployed, so it is located by its merge time.
    changes = sources.changes.list_changes(
        start=ANCHOR - timedelta(minutes=50), end=ANCHOR - timedelta(minutes=45)
    )
    assert [c.id for c in changes] == ["PR-4822"]
    assert changes[0].deployed_at is None


def test_get_change_detail_is_id_tolerant(sources):
    assert sources.changes.get_change("PR-4821") is not None
    assert sources.changes.get_change("pr-4821") is not None
    assert sources.changes.get_change("4821") is not None
    assert sources.changes.get_change("PR-9999") is None


def test_alerts_overlapping_the_window_are_returned(sources):
    # A-8990 fired at T-240 and resolved at T-150: it overlaps a window that
    # contains neither endpoint.
    alerts = sources.alerts.list_alerts(
        start=ANCHOR - timedelta(minutes=200), end=ANCHOR - timedelta(minutes=180)
    )
    assert "A-8990" in {a.id for a in alerts}


def test_alert_fire_order_points_at_payments_first(sources):
    start = ANCHOR - timedelta(minutes=30)
    alerts = sources.alerts.list_alerts(start=start, end=ANCHOR)
    ordered = sorted(alerts, key=lambda a: a.fired_at)

    # A long-running alert that started before the window still overlaps it.
    # That is correct, and it is exactly the trap: it is background noise, not
    # a cause. It is distinguishable because it fired before the window opened.
    pre_existing = [a for a in ordered if a.fired_at < start]
    assert {a.service for a in pre_existing} == {"notifications"}

    # Among alerts that actually fired during the incident, payments-api goes
    # first and the downstream service goes last.
    fired_during = [a for a in ordered if a.fired_at >= start]
    assert [a.service for a in fired_during] == [
        "payments-api",
        "payments-api",
        "payments-api",
        "ledger-service",
    ]
