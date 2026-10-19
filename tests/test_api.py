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
        from rca.models import EvidenceItem, RCAReport, TriagedIncident
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
            evidence=[
                EvidenceItem(
                    id="L1", kind="log", summary="ERROR pool exhausted", source="logs/pay"
                ),
                EvidenceItem(
                    id="C1", kind="change", summary="PR-4821 reuses connections", source="git"
                ),
            ],
            investigation_log=["log_volume(ERROR)", "search_logs('pool exhausted')"],
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


# --------------------------------------------------------------------------
# Follow-up chat
# --------------------------------------------------------------------------


class ChatLLM:
    """Stands in for the model server on the /chat route."""

    def __init__(self, reply="The deploy [C1] preceded the first error [L1]."):
        self.model = "stub"
        self.reply = reply
        self.calls = 0

    def health(self):
        return True, "stub"

    def chat(self, **kwargs):
        from rca.llm.base import LLMResponse, Usage

        self.calls += 1
        self.seen = kwargs
        return LLMResponse(text=self.reply, usage=Usage(prompt_tokens=10, completion_tokens=5))


@pytest.fixture()
def llm(client, monkeypatch):
    stub = ChatLLM()
    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: stub)
    return stub


def finished_job(client) -> str:
    return client.post(
        "/incidents", json={"title": "Checkout broken", "description": "500s on pay"}
    ).json()["job_id"]


def test_asking_a_question_returns_a_cited_answer(client, llm):
    job_id = finished_job(client)
    body = client.post(f"/jobs/{job_id}/chat", json={"question": "Why that PR?"}).json()

    assert body["answer"].startswith("The deploy")
    assert body["citations"] == ["L1", "C1"]
    assert body["unverified"] == []
    assert [e["id"] for e in body["evidence"]] == ["L1", "C1"]
    assert body["evidence"][0]["summary"] == "ERROR pool exhausted"


def test_an_invented_citation_is_flagged_to_the_caller(client, monkeypatch):
    """The API must not hand back a hallucinated ID as though it resolved -
    the UI renders citations as clickable evidence."""
    monkeypatch.setattr(
        server, "build_client", lambda *_a, **_k: ChatLLM("It was [C9], obviously.")
    )
    job_id = finished_job(client)
    body = client.post(f"/jobs/{job_id}/chat", json={"question": "cause?"}).json()

    assert body["unverified"] == ["C9"]
    assert body["citations"] == []
    assert "[unverified]" in body["answer"]


def test_the_transcript_accumulates_across_questions(client, llm):
    job_id = finished_job(client)
    client.post(f"/jobs/{job_id}/chat", json={"question": "first"})
    client.post(f"/jobs/{job_id}/chat", json={"question": "second"})

    state = client.get(f"/jobs/{job_id}/chat").json()
    roles = [t["role"] for t in state["turns"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert state["turns"][2]["content"] == "second"
    assert state["turns"][1]["citations"] == ["L1", "C1"]


def test_history_is_replayed_to_the_model(client, llm):
    """Without this, 'and the other one?' has nothing to refer back to."""
    job_id = finished_job(client)
    client.post(f"/jobs/{job_id}/chat", json={"question": "was it PR-4821?"})
    client.post(f"/jobs/{job_id}/chat", json={"question": "and the other one?"})

    contents = [m["content"] for m in llm.seen["messages"]]
    assert "was it PR-4821?" in contents
    assert contents[-1] == "and the other one?"


def test_chat_state_offers_openers_and_the_full_ledger(client, llm):
    state = client.get(f"/jobs/{finished_job(client)}/chat").json()
    assert state["headline"]
    assert [e["id"] for e in state["evidence"]] == ["L1", "C1"]
    assert state["suggestions"], "a blank chat box with no openers gets no use"
    assert state["turns"] == []


def test_the_transcript_can_be_cleared(client, llm):
    job_id = finished_job(client)
    client.post(f"/jobs/{job_id}/chat", json={"question": "why?"})
    assert client.delete(f"/jobs/{job_id}/chat").status_code == 204
    assert client.get(f"/jobs/{job_id}/chat").json()["turns"] == []


def test_asking_about_an_unknown_job_is_a_404(client, llm):
    assert client.post("/jobs/nope/chat", json={"question": "hi"}).status_code == 404
    assert client.get("/jobs/nope/chat").status_code == 404


def test_asking_before_the_investigation_finishes_is_a_conflict(client, llm):
    """409, not 500 and not an answer invented from nothing."""
    job = server.Job(id="pending", incident_id="INC-X")
    server.JOBS["pending"] = job

    resp = client.post("/jobs/pending/chat", json={"question": "what broke?"})
    assert resp.status_code == 409
    assert "queued" in resp.json()["detail"]
    assert llm.calls == 0, "no generation should run without a report to ground it"


def test_an_empty_question_is_rejected(client, llm):
    job_id = finished_job(client)
    assert client.post(f"/jobs/{job_id}/chat", json={"question": ""}).status_code == 422
    assert client.post(f"/jobs/{job_id}/chat", json={"question": "   "}).status_code == 422
    assert llm.calls == 0


def test_a_dead_model_server_is_a_503_not_a_500(client, monkeypatch):
    """The sources and the report are fine; only generation is unavailable.
    The UI shows that differently from a crash."""
    from rca.llm.base import LLMError

    class Dead(ChatLLM):
        def chat(self, **_kwargs):
            raise LLMError("connection refused")

    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: Dead())
    job_id = finished_job(client)
    resp = client.post(f"/jobs/{job_id}/chat", json={"question": "why?"})
    assert resp.status_code == 503
    assert "connection refused" in resp.json()["detail"]


def test_a_failed_question_leaves_no_half_turn_in_the_transcript(client, monkeypatch):
    from rca.llm.base import LLMError

    class Dead(ChatLLM):
        def chat(self, **_kwargs):
            raise LLMError("boom")

    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: Dead())
    job_id = finished_job(client)
    client.post(f"/jobs/{job_id}/chat", json={"question": "why?"})
    assert client.get(f"/jobs/{job_id}/chat").json()["turns"] == []


def test_the_console_page_is_served(client):
    resp = client.get("/chat")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "RCA Agent" in resp.text


# --------------------------------------------------------------------------
# POST /chat - the evaluator-facing endpoint (message in, reply out, no job ID)
# --------------------------------------------------------------------------


@pytest.fixture()
def standing(client, tmp_path, monkeypatch):
    """Point the endpoint at a saved report, the way RCA_CHAT_REPORT does."""
    from tests.test_chat import build_report

    path = tmp_path / "report.json"
    path.write_text(build_report().model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(server.settings, "chat_report", path)
    server._STANDING.clear()
    server.SESSIONS.clear()
    return path


def test_simple_chat_needs_only_a_message(client, standing, monkeypatch):
    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: ChatLLM("Deploy [C1], errors [L1]."))
    body = client.post("/chat", json={"message": "Why that PR?"}).json()

    assert body["reply"] == "Deploy [C1], errors [L1]."
    assert body["citations"] == ["L1", "C1"]
    assert body["unverified"] == []
    assert body["incident_id"] == "INC-1042"
    assert body["session_id"], "a session is minted when the caller sends none"


def test_simple_chat_threads_a_session(client, standing, monkeypatch):
    stub = ChatLLM("ok [C1]")
    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: stub)

    first = client.post("/chat", json={"message": "was it PR-4821?"}).json()
    client.post("/chat", json={"message": "and the other one?", "session_id": first["session_id"]})

    contents = [m["content"] for m in stub.seen["messages"]]
    assert "was it PR-4821?" in contents
    assert contents[-1] == "and the other one?"


def test_simple_chat_sessions_do_not_leak_into_each_other(client, standing, monkeypatch):
    stub = ChatLLM("ok")
    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: stub)

    client.post("/chat", json={"message": "secret from session A", "session_id": "A"})
    client.post("/chat", json={"message": "hello from B", "session_id": "B"})

    contents = [m["content"] for m in stub.seen["messages"]]
    assert "secret from session A" not in contents


def test_simple_chat_flags_an_invented_citation(client, standing, monkeypatch):
    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: ChatLLM("Cause.\nSources: [I1]"))
    body = client.post("/chat", json={"message": "cause?"}).json()
    assert body["unverified"] == ["I1"]
    assert "[I1]" not in body["reply"]


def test_simple_chat_without_a_report_is_a_clear_503(client, tmp_path, monkeypatch):
    """The failure an operator hits on a fresh checkout. It must say what to
    run, not return a bare 500."""
    monkeypatch.setattr(server.settings, "chat_report", tmp_path / "missing.json")
    server._STANDING.clear()
    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 503
    assert "--json-out" in resp.json()["detail"]


def test_simple_chat_rejects_an_empty_message(client, standing):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "   "}).status_code == 422


def test_simple_chat_with_a_dead_model_server_is_a_503(client, standing, monkeypatch):
    from rca.llm.base import LLMError

    class Dead(ChatLLM):
        def chat(self, **_kwargs):
            raise LLMError("connection refused")

    monkeypatch.setattr(server, "build_client", lambda *_a, **_k: Dead())
    resp = client.post("/chat", json={"message": "why?"})
    assert resp.status_code == 503
    assert SESSIONS_EMPTY_FOR(resp), "a failed turn must not be recorded"


def SESSIONS_EMPTY_FOR(_resp) -> bool:
    return all(len(turns) == 0 for turns in server.SESSIONS.values())


def test_simple_chat_reports_an_unsupported_time(client, standing, monkeypatch):
    """The warning must reach a caller that only reads `reply` *and* one that
    reads the structured field."""
    monkeypatch.setattr(
        server, "build_client",
        lambda *_a, **_k: ChatLLM("It was at 2026-09-25T07:51:00Z.\nSources: [L1]"),
    )
    body = client.post("/chat", json={"message": "when?"}).json()
    assert body["unsupported_times"] == ["2026-09-25T07:51:00Z"]
    assert "[Check:" in body["reply"]


def test_demo_seed_is_off_by_default(client, standing):
    """Tests and real deployments must not grow a phantom job unasked."""
    assert client.get("/jobs").json() == []


def test_demo_seed_opens_the_console_on_a_finished_report(client, standing, monkeypatch):
    monkeypatch.setenv("RCA_SEED_DEMO", "1")
    job = server.seed_demo_job()
    assert job is not None and job.status == "done"

    listed = client.get("/jobs").json()
    assert [j["incident_id"] for j in listed] == ["INC-1042"]
    state = client.get(f"/jobs/{job.id}/chat").json()
    assert state["evidence"], "the seeded job must be askable straight away"


def test_demo_seed_without_a_report_does_not_crash_startup(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server.settings, "chat_report", tmp_path / "nope.json")
    server._STANDING.clear()
    assert server.seed_demo_job() is None
