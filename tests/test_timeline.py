"""The ledger-derived timeline.

The fixtures use the exact one-line summaries the live run produced - the same
strings the evidence ledger writes - so the parsing is tested against what it
will really be handed, not against a shape I imagined.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rca.models import EvidenceItem
from rca.timeline import build_timeline

UTC = timezone.utc


def at(hh, mm, ss=0):
    return datetime(2026, 9, 25, hh, mm, ss, tzinfo=UTC)


CHANGE = EvidenceItem(
    id="C1", kind="change", source="git", timestamp=at(7, 51),
    summary="PR-4821 'Add merchant risk scoring to payment authorization' by priya.nair "
            "merged=2026-09-25T07:18:00Z deployed=2026-09-25T07:51:00Z services=payments-api (+287/-41, 3 files)",
)
ALERT = EvidenceItem(
    id="A1", kind="alert", source="mon", timestamp=at(8, 4),
    summary="A-9019 [warning] LedgerSettlementRetryRate on ledger-service "
            "2026-09-25T08:04:00Z -> ONGOING value=22.0 threshold=5.0",
)


def log(i, hh, mm, ss, msg, level="ERROR", service="payments-api"):
    return EvidenceItem(
        id=f"L{i}", kind="log", source=f"logs/{service}", timestamp=at(hh, mm, ss),
        summary=f"2026-09-25T{hh:02}:{mm:02}:{ss:02}Z {level:<5} [{service}] trace=6c0a404d6e2f883a {msg}",
    )


def test_the_alert_time_is_the_alerts_own_time_not_a_log_lines():
    """The live 3B timeline said the alert fired at 08:12:24. That is a log
    line; the alert fired at 08:04:00."""
    timeline = build_timeline([ALERT, log(3, 8, 12, 24, "POST /v1/payments/authorize 500 in 5013ms")])
    alert_line = next(line for line in timeline if "[A1]" in line)
    assert alert_line.startswith("2026-09-25T08:04:00Z")
    assert "LedgerSettlementRetryRate" in alert_line and "ledger-service" in alert_line


def test_every_line_names_the_evidence_it_comes_from():
    timeline = build_timeline([CHANGE, ALERT, log(3, 8, 12, 24, "boom")])
    assert [line.rsplit("[", 1)[1] for line in timeline] == ["C1]", "A1]", "L3]"]


def test_entries_are_in_time_order_regardless_of_input_order():
    timeline = build_timeline([ALERT, log(3, 8, 12, 24, "boom"), CHANGE])
    stamps = [line.split(" - ")[0] for line in timeline]
    assert stamps == sorted(stamps)


def test_a_deployed_change_says_deployed_and_an_undeployed_one_does_not():
    undeployed = EvidenceItem(
        id="C2", kind="change", source="git", timestamp=at(7, 18),
        summary="PR-4830 'Bump retry budget' by a.das merged=2026-09-25T07:18:00Z services=notifications (+4/-2, 1 file)",
    )
    lines = build_timeline([CHANGE, undeployed])
    assert any("PR-4821 deployed:" in line for line in lines)
    assert any("PR-4830 merged (no deploy time recorded):" in line for line in lines)


def test_repeated_log_shapes_collapse_with_a_count():
    """Fourteen identical timeouts are one fact, not fourteen lines."""
    logs = [log(i, 8, 12, i, f"TimeoutError: pool exhausted (size=10, in_use=10, waited {5000 + i}ms)")
            for i in range(1, 15)]
    lines = build_timeline(logs)
    assert len(lines) == 1
    assert "(x14)" in lines[0]
    assert lines[0].endswith("[L1]"), "the first occurrence is the one cited"


def test_different_messages_stay_separate():
    lines = build_timeline([log(1, 8, 0, 0, "slow query took 812ms", level="WARN"),
                            log(2, 8, 1, 0, "connection pool exhausted")])
    assert len(lines) == 2


def test_the_cap_keeps_every_change_and_alert_and_drops_the_latest_logs():
    """The first errors are the onset - the part of a timeline anyone reads."""
    logs = [log(i, 8, i, 0, f"distinct failure number-{chr(97 + i)}") for i in range(1, 25)]
    lines = build_timeline([CHANGE, ALERT, *logs], max_entries=8)

    assert any("[C1]" in line for line in lines) and any("[A1]" in line for line in lines)
    assert any("[L1]" in line for line in lines), "the earliest log survives"
    assert not any("[L24]" in line for line in lines), "the latest is what gets dropped"
    assert len(lines) <= 8


def test_evidence_with_no_timestamp_is_skipped_not_fatal():
    undated = EvidenceItem(id="L9", kind="log", source="x", summary="something happened")
    assert build_timeline([undated]) == []
    assert len(build_timeline([undated, CHANGE])) == 1


def test_an_unparseable_summary_still_yields_a_line():
    odd = EvidenceItem(id="L7", kind="log", source="x", timestamp=at(9, 0), summary="free text with no shape")
    lines = build_timeline([odd])
    assert lines == ["2026-09-25T09:00:00Z - free text with no shape [L7]"]


def test_an_empty_ledger_gives_an_empty_timeline():
    assert build_timeline([]) == []


def test_long_messages_are_shortened():
    line = build_timeline([log(1, 8, 0, 0, "x" * 400)])[0]
    assert len(line) < 220
    assert line.endswith("...") or "..." in line


def test_routine_traffic_is_dropped_once_real_errors_exist():
    """The live ledger held `GET /healthz 200` lines the log search returned
    alongside the errors. They are not incident events."""
    lines = build_timeline([
        log(1, 8, 12, 4, "TimeoutError: connection pool exhausted"),
        log(2, 8, 12, 13, "GET /healthz 200 in 71.5ms", level="INFO"),
        log(3, 8, 12, 31, "POST /v1/payments/authorize 200 in 57.2ms", level="INFO"),
    ])
    assert len(lines) == 1 and "[L1]" in lines[0]


def test_info_lines_are_kept_when_they_are_all_there_is():
    """An incident with no WARN/ERROR in the ledger must not render as empty."""
    lines = build_timeline([log(2, 8, 12, 13, "GET /healthz 200 in 71.5ms", level="INFO")])
    assert len(lines) == 1
