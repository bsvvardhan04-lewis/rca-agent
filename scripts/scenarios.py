"""Incident scenarios with recorded ground truth.

One scenario proves nothing. `INC-1042` can be solved by a rule as dumb as
"blame the most recent deploy" - and that rule is worse than useless in
production, because most incidents are not caused by the newest PR.

So the set is chosen to punish that shortcut:

    INC-1042  a recent deploy IS the cause          (the obvious case)
    INC-1043  NO deploy is the cause                (external provider)
    INC-1044  no deploy, gradual onset              (resource exhaustion)
    INC-1045  a config flip, not a code change      (looks like nothing shipped)

Each carries a `ground_truth` block the eval harness scores against, including
what must be *ruled out*. Getting the cause right while failing to dismiss the
decoy is a worse report than it looks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

UTC = timezone.utc
SEED = 20260923

ROUTES = [
    ("POST", "/v1/payments/authorize"),
    ("POST", "/v1/payments/capture"),
    ("GET", "/v1/payments/{id}"),
    ("POST", "/v1/refunds"),
    ("GET", "/healthz"),
]


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log(ts: datetime, level: str, service: str, message: str, **attrs) -> dict:
    entry = {"timestamp": iso(ts), "level": level, "service": service, "message": message}
    trace = attrs.pop("trace_id", None)
    if trace:
        entry["trace_id"] = trace
    if attrs:
        entry["attributes"] = attrs
    return entry


def trace_id(rng: random.Random) -> str:
    return f"{rng.getrandbits(64):016x}"


def normal_traffic(
    rng: random.Random, start: datetime, end: datetime, service: str, per_minute: int
) -> list[dict]:
    """Background chatter, so the interesting lines are not the only lines."""
    out: list[dict] = []
    cursor = start
    while cursor < end:
        for _ in range(rng.randint(max(1, per_minute - 2), per_minute + 2)):
            ts = cursor + timedelta(seconds=rng.uniform(0, 60))
            method, route = rng.choice(ROUTES)
            latency = max(9.0, round(rng.gauss(88, 26), 1))
            out.append(
                log(
                    ts,
                    "INFO",
                    service,
                    f"{method} {route} 200 in {latency}ms",
                    trace_id=trace_id(rng),
                    status=200,
                    latency_ms=latency,
                )
            )
        cursor += timedelta(minutes=1)
    return out


@dataclass
class GroundTruth:
    """What a correct RCA looks like, for scoring."""

    category: str
    root_cause_id: str = ""
    """The change that caused it, or "" when no change did."""

    expect_no_recent_change: bool = False
    """True when the correct answer is 'nothing shipped - look elsewhere'."""

    must_mention: list[str] = field(default_factory=list)
    """Substrings that a correct explanation of the mechanism contains."""

    must_rule_out: list[str] = field(default_factory=list)
    """Decoy IDs a good report explicitly dismisses."""

    notes: str = ""


@dataclass
class Scenario:
    id: str
    title: str
    description: str
    reporter: str
    service_hint: str | None
    severity_hint: str
    ground_truth: GroundTruth
    build: Callable[[datetime, random.Random], dict[str, Any]]

    def render(self, anchor: datetime) -> dict[str, Any]:
        rng = random.Random(SEED)
        payload = self.build(anchor, rng)
        payload["incident"] = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "reported_at": iso(anchor),
            "reporter": self.reporter,
            "service_hint": self.service_hint,
            "severity_hint": self.severity_hint,
        }
        payload["ground_truth"] = {
            "category": self.ground_truth.category,
            "root_cause_id": self.ground_truth.root_cause_id,
            "expect_no_recent_change": self.ground_truth.expect_no_recent_change,
            "must_mention": self.ground_truth.must_mention,
            "must_rule_out": self.ground_truth.must_rule_out,
            "notes": self.ground_truth.notes,
        }
        return payload


# ---------------------------------------------------------------------------
# INC-1042 - a recent deploy really is the cause
# ---------------------------------------------------------------------------


def _build_1042(anchor: datetime, rng: random.Random) -> dict[str, Any]:
    def at(m: float) -> datetime:
        return anchor + timedelta(minutes=m)

    payments = normal_traffic(rng, at(-180), at(-21), "payments-api", 6)
    checkout = normal_traffic(rng, at(-180), at(-10), "checkout-web", 4)
    ledger = normal_traffic(rng, at(-180), at(5), "ledger-service", 3)
    notifications: list[dict] = []

    cursor = at(-180)
    while cursor < at(5):
        ledger.append(
            log(
                cursor + timedelta(seconds=rng.uniform(0, 240)),
                "WARN",
                "ledger-service",
                "slow query: SELECT * FROM ledger_entries WHERE account_id = $1 took 812ms",
                duration_ms=round(rng.uniform(700, 1100), 1),
            )
        )
        cursor += timedelta(minutes=4)

    for i in range(14):
        notifications.append(
            log(
                at(-120 + i * 8),
                "WARN",
                "notifications",
                f"email queue depth {4200 + i * 310} exceeds soft limit 4000",
                depth=4200 + i * 310,
            )
        )

    checkout.append(
        log(at(-110), "INFO", "checkout-web", "deploy complete: build 2291 (PR #4817)", build=2291)
    )
    payments.append(
        log(at(-22), "INFO", "payments-api", "deploy complete: build 3317 (PR #4821)", build=3317)
    )
    payments.append(
        log(
            at(-22) + timedelta(seconds=4),
            "INFO",
            "payments-api",
            "db pool initialised: min=2 max=10 acquire_timeout=5000ms",
            pool_max=10,
        )
    )

    for i in range(7):
        payments.append(
            log(
                at(-21 + i * 0.5),
                "WARN",
                "payments-api",
                f"db pool checkout waited {420 + i * 190}ms (in_use={min(8 + i // 3, 10)}/10)",
                trace_id=trace_id(rng),
                pool_in_use=min(8 + i // 3, 10),
            )
        )
    for i in range(9):
        payments.append(
            log(
                at(-18 + i * 0.4),
                "WARN",
                "payments-api",
                f"risk-service call took {1800 + i * 140}ms inside transaction tx_authorize",
                trace_id=trace_id(rng),
                upstream="risk-service",
                inside_transaction=True,
            )
        )

    for i in range(46):
        t = at(-15 + i * 0.32)
        tid = trace_id(rng)
        payments.append(
            log(
                t,
                "ERROR",
                "payments-api",
                "TimeoutError: connection pool exhausted (size=10, in_use=10, "
                "waited 5001ms) acquiring connection for tx_authorize",
                trace_id=tid,
                error_type="TimeoutError",
                pool_max=10,
                pool_in_use=10,
            )
        )
        payments.append(
            log(
                t + timedelta(milliseconds=12),
                "ERROR",
                "payments-api",
                "POST /v1/payments/authorize 500 in 5013ms",
                trace_id=tid,
                status=500,
            )
        )

    for i in range(18):
        ledger.append(
            log(
                at(-10 + i * 0.6),
                "WARN",
                "ledger-service",
                f"retrying settlement for payment_intent=pi_{rng.getrandbits(32):08x}: "
                f"upstream 500 from payments-api (attempt {(i % 5) + 1}/5)",
                trace_id=trace_id(rng),
                upstream="payments-api",
            )
        )
    for i in range(12):
        checkout.append(
            log(
                at(-8 + i * 0.5),
                "ERROR",
                "checkout-web",
                "payment authorization failed: upstream returned 500, showing generic error",
                trace_id=trace_id(rng),
                upstream="payments-api",
            )
        )
    payments += normal_traffic(rng, at(-15), at(2), "payments-api", 2)

    return {
        "logs": {
            "payments-api": payments,
            "checkout-web": checkout,
            "ledger-service": ledger,
            "notifications": notifications,
        },
        "alerts": [
            _alert("A-9001", "NotificationsEmailQueueBacklog", "notifications", "warning",
                   at(-120), None, 8540.0, 4000.0),
            _alert("A-9014", "PaymentsApiLatencyP99High", "payments-api", "warning",
                   at(-18), None, 4870.0, 800.0),
            _alert("A-9016", "PaymentsApiDbPoolSaturation", "payments-api", "critical",
                   at(-14), None, 100.0, 85.0),
            _alert("A-9017", "PaymentsApi5xxRate", "payments-api", "critical",
                   at(-12), None, 31.4, 1.0),
            _alert("A-9019", "LedgerSettlementRetryRate", "ledger-service", "warning",
                   at(-9), None, 22.0, 5.0),
            _alert("A-8990", "CdnOriginCacheMissRateHigh", "checkout-web", "warning",
                   at(-240), at(-150), 41.0, 30.0),
        ],
        "changes": [
            _change("PR-4817", "Festive banner copy + asset swap on checkout", "arjun.rao",
                    at(-130), at(-110), ["checkout-web"],
                    ["web/src/components/Banner.tsx", "web/src/copy/en.json"], 64, 12,
                    "Seasonal banner. Copy and one asset. No API or data-layer changes."),
            _change("PR-4821", "Add merchant risk scoring to payment authorization",
                    "priya.nair", at(-55), at(-22), ["payments-api"],
                    ["services/payments/auth.py", "services/payments/clients/risk_service.py",
                     "services/payments/db/transaction.py"], 287, 41,
                    "Calls risk-service to score the merchant before authorising. The score "
                    "is written to the payment row, so the call happens inside tx_authorize "
                    "to keep the write atomic. Adds a 3s client timeout. Pool unchanged."),
            _change("PR-4822", "Update README badges and CODEOWNERS", "sam.fernandes",
                    at(-48), None, [], ["README.md", ".github/CODEOWNERS"], 9, 9, "Docs only."),
            _change("PR-4809", "Increase ledger settlement retry budget from 3 to 5",
                    "dev.kapoor", at(-1500), at(-1440), ["ledger-service"],
                    ["services/ledger/settlement.py"], 18, 6, "Yesterday's change."),
        ],
    }


# ---------------------------------------------------------------------------
# INC-1043 - nothing shipped; an external provider broke
# ---------------------------------------------------------------------------


def _build_1043(anchor: datetime, rng: random.Random) -> dict[str, Any]:
    def at(m: float) -> datetime:
        return anchor + timedelta(minutes=m)

    payments = normal_traffic(rng, at(-180), at(-24), "payments-api", 6)
    checkout = normal_traffic(rng, at(-180), at(-12), "checkout-web", 4)
    ledger = normal_traffic(rng, at(-180), at(5), "ledger-service", 3)

    # A deploy DID happen - but hours earlier, and to an unrelated service.
    # An agent that pattern-matches "most recent deploy" will blame this.
    ledger.append(
        log(at(-95), "INFO", "ledger-service", "deploy complete: build 881 (PR #4901)", build=881)
    )

    for i in range(6):
        payments.append(
            log(
                at(-24 + i * 0.6),
                "WARN",
                "payments-api",
                f"acquirer-gateway responded 503 in {820 + i * 60}ms, retrying (attempt {i % 3 + 1}/3)",
                trace_id=trace_id(rng),
                upstream="acquirer-gateway",
                upstream_status=503,
            )
        )

    for i in range(38):
        t = at(-20 + i * 0.45)
        tid = trace_id(rng)
        payments.append(
            log(
                t,
                "ERROR",
                "payments-api",
                "UpstreamError: acquirer-gateway returned 503 Service Unavailable "
                "after 3 retries (provider=northstar-acquiring)",
                trace_id=tid,
                error_type="UpstreamError",
                upstream="acquirer-gateway",
                upstream_status=503,
                provider="northstar-acquiring",
            )
        )
        payments.append(
            log(
                t + timedelta(milliseconds=9),
                "ERROR",
                "payments-api",
                "POST /v1/payments/authorize 502 in 2740ms",
                trace_id=tid,
                status=502,
            )
        )

    for i in range(10):
        checkout.append(
            log(
                at(-12 + i * 0.7),
                "ERROR",
                "checkout-web",
                "payment authorization failed: upstream returned 502",
                trace_id=trace_id(rng),
                upstream="payments-api",
            )
        )

    return {
        "logs": {
            "payments-api": payments,
            "checkout-web": checkout,
            "ledger-service": ledger,
            "notifications": [],
        },
        "alerts": [
            _alert("A-9101", "AcquirerGatewayErrorRate", "payments-api", "critical",
                   at(-22), None, 96.0, 2.0,
                   {"upstream": "acquirer-gateway", "provider": "northstar-acquiring"}),
            _alert("A-9103", "PaymentsApi5xxRate", "payments-api", "critical",
                   at(-18), None, 44.0, 1.0),
            _alert("A-9104", "CheckoutConversionDrop", "checkout-web", "warning",
                   at(-11), None, 38.0, 10.0),
        ],
        "changes": [
            _change("PR-4901", "Add settlement batch size metric", "dev.kapoor",
                    at(-140), at(-95), ["ledger-service"],
                    ["services/ledger/metrics.py"], 34, 2,
                    "Emits a gauge for settlement batch size. Observability only - no "
                    "change to the settlement or payment path."),
            _change("PR-4899", "Bump lint config", "sam.fernandes", at(-300), None, [],
                    [".ruff.toml"], 4, 4, "Tooling only."),
        ],
    }


# ---------------------------------------------------------------------------
# INC-1044 - gradual resource exhaustion, no change at all
# ---------------------------------------------------------------------------


def _build_1044(anchor: datetime, rng: random.Random) -> dict[str, Any]:
    def at(m: float) -> datetime:
        return anchor + timedelta(minutes=m)

    reports = normal_traffic(rng, at(-240), at(-30), "reporting-api", 3)

    # Disk has been creeping for days. The warnings go back the whole window
    # and get worse - there is no sharp onset to find.
    for i in range(40):
        pct = 71 + i * 0.7
        reports.append(
            log(
                at(-240 + i * 6),
                "WARN" if pct < 95 else "ERROR",
                "reporting-api",
                f"disk usage on /var/lib/reports at {pct:.1f}% (threshold 90%)",
                disk_pct=round(pct, 1),
                mount="/var/lib/reports",
            )
        )

    for i in range(22):
        reports.append(
            log(
                at(-28 + i * 1.2),
                "ERROR",
                "reporting-api",
                "OSError: [Errno 28] No space left on device: "
                "'/var/lib/reports/export-{}.parquet'".format(rng.getrandbits(24)),
                error_type="OSError",
                errno=28,
            )
        )

    return {
        "logs": {"reporting-api": reports, "payments-api": [], "checkout-web": [],
                 "ledger-service": [], "notifications": []},
        "alerts": [
            _alert("A-9201", "ReportingDiskUsageHigh", "reporting-api", "warning",
                   at(-1900), None, 91.0, 90.0, {"mount": "/var/lib/reports"}),
            _alert("A-9202", "ReportingDiskUsageCritical", "reporting-api", "critical",
                   at(-34), None, 99.4, 98.0, {"mount": "/var/lib/reports"}),
            _alert("A-9203", "ReportExportFailureRate", "reporting-api", "critical",
                   at(-26), None, 88.0, 5.0),
        ],
        "changes": [
            # Deployed 150 minutes before the first hard error, and - crucially -
            # *after* the disk warnings had already been climbing for 90 minutes.
            # That ordering is what lets a careful agent rule it out. Putting it
            # closer to the onset would make blaming it defensible, which would
            # make the ground truth unfair rather than the scenario hard.
            _change("PR-5010", "Tidy report export filename format", "arjun.rao",
                    at(-180), at(-150), ["reporting-api"],
                    ["services/reporting/export.py"], 12, 8,
                    "Renames exported files to include a date prefix. No change to "
                    "retention, volume or cleanup."),
        ],
    }


# ---------------------------------------------------------------------------
# INC-1045 - a feature flag flip; no PR deployed
# ---------------------------------------------------------------------------


def _build_1045(anchor: datetime, rng: random.Random) -> dict[str, Any]:
    def at(m: float) -> datetime:
        return anchor + timedelta(minutes=m)

    search = normal_traffic(rng, at(-180), at(-26), "search-api", 5)
    checkout = normal_traffic(rng, at(-180), at(-15), "checkout-web", 4)

    search.append(
        log(
            at(-27),
            "INFO",
            "search-api",
            "feature flag changed: use_vector_reranker false -> true "
            "(actor=ops-console, rollout=100%)",
            flag="use_vector_reranker",
            old_value=False,
            new_value=True,
            rollout_pct=100,
        )
    )

    for i in range(8):
        search.append(
            log(
                at(-26 + i * 0.5),
                "WARN",
                "search-api",
                f"reranker inference took {900 + i * 220}ms (budget 300ms)",
                trace_id=trace_id(rng),
                component="vector-reranker",
            )
        )
    for i in range(30):
        t = at(-22 + i * 0.5)
        search.append(
            log(
                t,
                "ERROR",
                "search-api",
                "DeadlineExceeded: reranker did not return within 3000ms budget "
                "(flag use_vector_reranker enabled)",
                trace_id=trace_id(rng),
                error_type="DeadlineExceeded",
                component="vector-reranker",
            )
        )
    for i in range(9):
        checkout.append(
            log(
                at(-15 + i * 0.8),
                "ERROR",
                "checkout-web",
                "product search timed out, showing empty results",
                trace_id=trace_id(rng),
                upstream="search-api",
            )
        )

    return {
        "logs": {"search-api": search, "checkout-web": checkout, "payments-api": [],
                 "ledger-service": [], "notifications": []},
        "alerts": [
            _alert("A-9301", "SearchApiLatencyP99High", "search-api", "critical",
                   at(-25), None, 3100.0, 400.0),
            _alert("A-9302", "SearchApiTimeoutRate", "search-api", "critical",
                   at(-21), None, 41.0, 2.0),
        ],
        "changes": [
            _change("PR-5120", "Add vector reranker behind a feature flag", "priya.nair",
                    at(-4300), at(-4200), ["search-api"],
                    ["services/search/rerank.py", "config/flags.yaml"], 402, 15,
                    "Adds a vector reranker. Shipped DISABLED behind use_vector_reranker. "
                    "Merged three days ago; no behaviour change until the flag is enabled."),
            _change("PR-5133", "Update search API docs", "sam.fernandes",
                    at(-60), None, [], ["docs/search.md"], 22, 3, "Docs only."),
        ],
    }


# ---------------------------------------------------------------------------


def _alert(aid, name, service, severity, fired, resolved, value, threshold, labels=None):
    return {
        "id": aid,
        "name": name,
        "service": service,
        "severity": severity,
        "fired_at": iso(fired),
        "resolved_at": iso(resolved) if resolved else None,
        "value": value,
        "threshold": threshold,
        "labels": labels or {},
    }


def _change(cid, title, author, merged, deployed, services, files, adds, dels, description):
    return {
        "id": cid,
        "title": title,
        "author": author,
        "merged_at": iso(merged),
        "deployed_at": iso(deployed) if deployed else None,
        "services": services,
        "files_changed": files,
        "additions": adds,
        "deletions": dels,
        "url": f"https://github.example.com/acme/platform/pull/{cid.removeprefix('PR-')}",
        "description": description,
    }


SCENARIOS: dict[str, Scenario] = {
    "INC-1042": Scenario(
        id="INC-1042",
        title="Checkout failing for a chunk of customers - payments returning 500",
        description=(
            "Support is getting a wave of tickets since roughly 20 minutes ago. Customers "
            "click Pay and get 'Something went wrong, please try again'. It is not everyone "
            "- some payments go through. Our own test card failed twice then succeeded on "
            "the third attempt. Checkout page itself loads fine. Nobody on the team knows "
            "of a planned change. Please find out what is going on."
        ),
        reporter="nisha.support",
        service_hint="checkout-web",
        severity_hint="SEV2",
        ground_truth=GroundTruth(
            category="code_change",
            root_cause_id="PR-4821",
            must_mention=["pool", "transaction"],
            must_rule_out=["PR-4817"],
            notes="A synchronous risk-service call inside tx_authorize holds a pooled "
            "connection across a network call, exhausting a pool of 10.",
        ),
        build=_build_1042,
    ),
    "INC-1043": Scenario(
        id="INC-1043",
        title="Payments failing - card declines spiking",
        description=(
            "Payments started failing about 20 minutes ago. The error customers see is "
            "generic. We have not deployed anything today as far as I know. Finance is "
            "asking whether this is us or the bank. Need an answer quickly."
        ),
        reporter="rahul.ops",
        service_hint="payments-api",
        severity_hint="SEV1",
        ground_truth=GroundTruth(
            category="external_provider",
            root_cause_id="",
            expect_no_recent_change=True,
            must_mention=["acquirer", "503"],
            must_rule_out=["PR-4901"],
            notes="The upstream acquirer is returning 503. No deploy is responsible. "
            "PR-4901 is observability-only, on a different service, 95 minutes earlier - "
            "a report that blames it has pattern-matched 'most recent deploy'.",
        ),
        build=_build_1043,
    ),
    "INC-1044": Scenario(
        id="INC-1044",
        title="Scheduled report exports failing since this morning",
        description=(
            "Client report exports are failing. A few went through earlier but now almost "
            "all of them error out. This has been getting worse rather than breaking all "
            "at once. No one has touched the reporting service recently."
        ),
        reporter="meera.delivery",
        service_hint="reporting-api",
        severity_hint="SEV2",
        ground_truth=GroundTruth(
            category="resource_exhaustion",
            root_cause_id="",
            expect_no_recent_change=True,
            must_mention=["disk", "space"],
            must_rule_out=["PR-5010"],
            notes="Disk filled gradually over days. There is no sharp onset, and the "
            "only recent PR is a cosmetic filename change that cannot consume disk.",
        ),
        build=_build_1044,
    ),
    "INC-1045": Scenario(
        id="INC-1045",
        title="Product search returning nothing on the storefront",
        description=(
            "Search on the storefront has stopped returning results for a lot of queries - "
            "it just spins and then shows an empty page. Started around half an hour ago. "
            "The last search deploy was days ago so it cannot be that."
        ),
        reporter="nisha.support",
        service_hint="checkout-web",
        severity_hint="SEV2",
        ground_truth=GroundTruth(
            category="configuration",
            root_cause_id="",
            expect_no_recent_change=True,
            must_mention=["flag", "rerank"],
            must_rule_out=["PR-5133"],
            notes="A feature flag was flipped in the ops console - no deploy. The code "
            "shipped three days ago but was inert until the flag was enabled. The "
            "reporter's own claim that 'it cannot be the deploy' is correct but for the "
            "wrong reason.",
        ),
        build=_build_1045,
    ),
}
