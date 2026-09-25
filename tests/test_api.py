"""The webhook layer: does a vendor-shaped payload become the right incident?

The agent itself is stubbed here - what is under test is the adapter code and
the job lifecycle, not the model.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api import server  # noqa: E402
from tests.test_agent import analysis  # noqa: E402


class StubAgent:
    """Records the incident it was handed and returns a canned report."""

    last_incident = None

    def __init__(self, *_args, **kwargs):
        self.on_event = kwargs.get("on_event") or (lambda k, m: None)

    def investigate(self, incident):
        StubAgent.last_incident = incident
        self.on_event("phase", "Triage")
        self.on_event("phase", "Investigation")
        from rca.models import RCAReport, TriagedIncident
        from rca.models import iso, utcnow
        from datetime import timedelta

        now = utcnow()
        return RCAReport(
            incident=incident,
            triage=TriagedIncident(
                summary="stub",
                affected_services=["payments-api"],
                symptoms=["stub"],
                error_signatures=["500"],
                search_keywords=["stub"],
                window_start=iso(now - timedelta(hours=3)),
                window_end=iso(now),
                severity="SEV2",
                triage_notes="stub",
            ),
            analysis=analysis(["L1"]),
            evidence=[],
            model="stub",
        )


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "RCAAgent", StubAgent)
    server.JOBS.clear()
    StubAgent.last_incident = None
    # TestClient runs BackgroundTasks synchronously once the response is sent.
    with TestClient(server.app) as c:
        yield c


class UpClient:
    model = "qwen2.5:7b"

    def health(self):
        return True, "http://localhost:11434/v1 serving qwen2.5:7b"


class DownClient:
    model = "qwen2.5:7b"

    def health(self):
        return False, "http://localhost:11434/v1 unreachable (APIConnectionError)"


def test_health_is_ok_when_the_model_server_answers(client, monkeypatch):
    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: UpClient())
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_server_reachable"] is True
    assert "payments-api" in body["services"]
    assert body["strategy"] in {"agentic", "guided", "auto"}


def test_health_is_degraded_when_the_model_server_is_down(client, monkeypatch):
    """A readiness probe must not claim ok while the model is unreachable -
    the sources work, but no investigation can actually run."""
    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: DownClient())
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_server_reachable"] is False
    assert "unreachable" in body["model_server_detail"]


def test_posting_an_incident_returns_a_pollable_job(client):
    resp = client.post(
        "/incidents",
        json={"title": "Checkout broken", "description": "500s on pay", "id": "INC-9"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.json()["poll"] == f"/jobs/{job_id}"

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["incident_id"] == "INC-9"
    assert job["progress"], "phase events should be recorded as progress"


def test_markdown_is_available_once_the_job_finishes(client):
    job_id = client.post(
        "/incidents", json={"title": "t", "description": "d"}
    ).json()["job_id"]
    md = client.get(f"/jobs/{job_id}/markdown").json()["markdown"]
    assert md.startswith("# RCA:")


def test_markdown_before_completion_is_a_conflict_not_a_crash(client):
    server.JOBS["pending"] = server.Job(id="pending", incident_id="X", status="running")
    assert client.get("/jobs/pending/markdown").status_code == 409


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_a_failing_investigation_is_recorded_not_swallowed(client, monkeypatch):
    class Boom(StubAgent):
        def investigate(self, incident):
            raise RuntimeError("upstream exploded")

    monkeypatch.setattr(server, "RCAAgent", Boom)
    job_id = client.post("/incidents", json={"title": "t", "description": "d"}).json()[
        "job_id"
    ]
    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "upstream exploded" in job["error"]


def test_jira_payload_with_plain_text_description(client):
    client.post(
        "/webhooks/jira",
        json={
            "issue": {
                "key": "OPS-512",
                "fields": {
                    "summary": "Payments 500s",
                    "description": "Customers cannot pay.",
                    "reporter": {"displayName": "Nisha"},
                    "components": [{"name": "checkout-web"}],
                    "created": "2026-09-23T14:00:00.000+0000",
                },
            }
        },
    )
    incident = StubAgent.last_incident
    assert incident.id == "OPS-512"
    assert incident.reporter == "Nisha"
    assert incident.service_hint == "checkout-web"
    assert incident.description == "Customers cannot pay."


def test_jira_atlassian_document_format_is_flattened(client):
    client.post(
        "/webhooks/jira",
        json={
            "issue": {
                "key": "OPS-513",
                "fields": {
                    "summary": "Payments 500s",
                    "description": {
                        "type": "doc",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {"type": "text", "text": "Pay button"},
                                    {"type": "text", "text": "returns an error."},
                                ],
                            }
                        ],
                    },
                },
            }
        },
    )
    assert StubAgent.last_incident.description == "Pay button returns an error."


def test_jira_payload_without_a_summary_is_rejected(client):
    resp = client.post("/webhooks/jira", json={"issue": {"fields": {}}})
    assert resp.status_code == 422


def test_sentry_payload_becomes_an_incident(client):
    client.post(
        "/webhooks/sentry",
        json={
            "data": {
                "issue": {
                    "title": "TimeoutError: connection pool exhausted",
                    "shortId": "PAY-4F",
                    "culprit": "payments.auth in tx_authorize",
                    "count": 46,
                    "project": {"slug": "payments-api"},
                    "firstSeen": "2026-09-23T13:45:00Z",
                }
            }
        },
    )
    incident = StubAgent.last_incident
    assert incident.id == "SENTRY-PAY-4F"
    assert incident.service_hint == "payments-api"
    assert "tx_authorize" in incident.description
    assert incident.reported_at.hour == 13


def test_a_bad_timestamp_in_a_webhook_does_not_reject_the_incident(client):
    client.post(
        "/webhooks/sentry",
        json={"data": {"issue": {"title": "Something broke", "firstSeen": "whenever"}}},
    )
    assert StubAgent.last_incident.title == "Something broke"
