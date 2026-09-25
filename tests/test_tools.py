"""The tool layer is the agent's whole view of the world. Test what it shows."""

from __future__ import annotations

import json
from datetime import timedelta

from rca.tools import build_tools
from tests.conftest import ANCHOR


def tools_by_name(ctx):
    return {t.name: t for t in build_tools(ctx)}


def test_every_tool_generates_a_valid_schema(ctx):
    for tool in build_tools(ctx):
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
        assert tool.description, f"{tool.name} has no description"
        for name, prop in schema.get("properties", {}).items():
            assert prop.get("description"), f"{tool.name}.{name} is undocumented"
        # Must be JSON-serialisable to go over the wire.
        json.dumps(schema)


def test_search_logs_assigns_stable_evidence_ids(ctx):
    search = tools_by_name(ctx)["search_logs"]
    first = search.call({"query": '"pool exhausted"', "limit": 3})
    assert "[L1]" in first and "[L3]" in first

    # The same lines must keep the same IDs - a citation cannot mean two things.
    again = search.call({"query": '"pool exhausted"', "limit": 3})
    assert "[L1]" in again
    assert len(ctx.ledger) == 3


def test_search_logs_reports_the_full_match_count_not_just_what_it_shows(ctx):
    out = tools_by_name(ctx)["search_logs"].call({"query": '"pool exhausted"', "limit": 3})
    assert "46 line(s) matched" in out
    assert "43 older matching line(s) not shown" in out


def test_search_logs_limit_is_capped(ctx):
    out = tools_by_name(ctx)["search_logs"].call({"query": "", "limit": 10_000})
    shown = out.count("\n  [L")
    assert shown <= ctx.settings.max_log_lines


def test_empty_result_explains_what_to_try_next(ctx):
    out = tools_by_name(ctx)["search_logs"].call({"query": "kubernetes oomkilled"})
    assert "No log lines matched" in out
    assert "wider window" in out


def test_log_volume_shows_a_flat_then_spiking_shape(ctx):
    out = tools_by_name(ctx)["log_volume"].call({"levels": "ERROR", "bucket_minutes": 15})
    rows = [line for line in out.splitlines() if line.startswith("  2026-")]
    counts = [int(line.split()[1]) for line in rows]
    assert counts[0] == 0, "the start of a 3h window should be quiet"
    assert max(counts) > 0
    # The spike belongs at the end, near the report time.
    assert counts.index(max(counts)) >= len(counts) - 2


def test_blank_start_and_end_fall_back_to_the_incident_window(ctx):
    scoped = tools_by_name(ctx)["search_logs"].call({"query": "", "limit": 1})
    assert "matched" in scoped
    # An explicit window outside the seeded data returns nothing.
    far_past = tools_by_name(ctx)["search_logs"].call(
        {"query": "", "start": "2020-01-01T00:00:00Z", "end": "2020-01-02T00:00:00Z"}
    )
    assert "No log lines matched" in far_past


def test_unparseable_timestamps_do_not_crash_the_tool(ctx):
    out = tools_by_name(ctx)["search_logs"].call(
        {"query": "", "start": "yesterday-ish", "end": "soon", "limit": 1}
    )
    assert "matched" in out, "a bad timestamp should fall back, not raise"


def test_list_recent_changes_can_widen_with_lookback_hours(ctx):
    changes = tools_by_name(ctx)["list_recent_changes"]
    narrow = changes.call({"start": "", "end": "", "lookback_hours": 0})
    assert "PR-4821" in narrow

    # Yesterday's ledger change is outside the 3h window but inside 48h.
    wide = changes.call({"lookback_hours": 48})
    assert "PR-4809" in wide
    assert "PR-4809" not in narrow


def test_get_change_detail_exposes_the_mechanism(ctx):
    out = tools_by_name(ctx)["get_change_detail"].call({"change_id": "PR-4821"})
    assert "services/payments/db/transaction.py" in out
    assert "inside tx_authorize" in out
    assert "deployed:" in out


def test_get_change_detail_on_unknown_id_is_a_message_not_an_exception(ctx):
    out = tools_by_name(ctx)["get_change_detail"].call({"change_id": "PR-0000"})
    assert "No change found" in out


def test_list_alerts_is_ordered_oldest_first(ctx):
    out = tools_by_name(ctx)["list_alerts"].call({})
    lines = [line for line in out.splitlines() if line.strip().startswith("[A")]
    timestamps = [line.split(" on ")[1].split()[1] for line in lines]
    assert timestamps == sorted(timestamps), "fire order is the causal signal"


def test_tool_calls_are_recorded_for_the_audit_trail(ctx):
    tools = tools_by_name(ctx)
    tools["list_services"].call({})
    tools["list_alerts"].call({})
    assert len(ctx.call_log) == 2
    assert ctx.call_log[0].startswith("list_services")


def test_evidence_ids_are_namespaced_by_kind(ctx):
    tools = tools_by_name(ctx)
    tools["search_logs"].call({"query": "ERROR", "limit": 2})
    tools["list_recent_changes"].call({})
    tools["list_alerts"].call({})
    kinds = {item.id[0] for item in ctx.ledger.items()}
    assert kinds == {"L", "C", "A"}
    assert ctx.ledger.get("L1") is not None
    assert ctx.ledger.get("Z9") is None


def test_unattributed_changes_are_flagged_not_silently_mixed_in(ctx):
    """A change that survived a service filter only because we could not place
    it must say so - otherwise it reads as a confirmed hit on that service."""
    out = tools_by_name(ctx)["list_recent_changes"].call({"services": "payments-api"})
    assert "PR-4821" in out
    # PR-4822 is docs-only, declares no services, and cannot be ruled out.
    assert "PR-4822" in out
    assert "UNATTRIBUTED" in out
    assert "could not be attributed" in out


def test_no_unattributed_banner_when_no_service_filter_was_asked_for(ctx):
    out = tools_by_name(ctx)["list_recent_changes"].call({})
    assert "UNATTRIBUTED" not in out
