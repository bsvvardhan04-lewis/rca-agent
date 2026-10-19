"""HTTP trigger for the RCA agent.

The problem this solves: an RCA should start the moment a ticket is filed, not
when someone remembers to run a CLI. Point a Jira automation rule, a Sentry
alert rule, or a PagerDuty webhook at this service and the investigation is
already done by the time an engineer opens the ticket.

    uvicorn api.server:app --reload --port 8000

A full investigation takes tens of seconds, which is longer than most webhook
senders will wait. So the default is fire-and-forget: POST returns a job ID
immediately, and the caller (or a human) polls for the result.

The job store is in-process and non-durable. That is fine for a single worker
and a demo; for real use, back it with Redis or a table and run the agent in a
worker process rather than a BackgroundTask.
"""

from __future__ import annotations

import logging
import os
import threading
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rca.agent import RCAAgent
from rca.chat import ReportChat, suggested_questions
from rca.config import settings
from rca.llm import LLMError, build_client
from rca.models import ChatTurn, IncidentReport, RCAReport, utcnow
from rca.report import to_markdown
from rca.sources import SourceBundle, build_sources

STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger("rca.api")

app = FastAPI(
    title="RCA Agent",
    description="Automated root cause analysis for incidents.",
    version="0.1.0",
)


# --------------------------------------------------------------------------
# Job store
# --------------------------------------------------------------------------


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    incident_id: str
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    progress: list[str] = Field(default_factory=list)
    report: RCAReport | None = None
    error: str | None = None
    chat: list[ChatTurn] = Field(default_factory=list)


JOBS: dict[str, Job] = {}

# One generation at a time per job. Two browser tabs asking the same job
# simultaneously would otherwise interleave their turns into the transcript and
# each would see a history the other half-wrote.
CHAT_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)


def run_investigation(job_id: str, incident: IncidentReport) -> None:
    job = JOBS[job_id]
    job.status = "running"

    def on_event(kind: str, message: str) -> None:
        if kind in {"phase", "warn"}:
            job.progress.append(message)

    try:
        agent = RCAAgent(build_sources(), on_event=on_event)
        job.report = agent.investigate(incident)
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 - the job records its own failure
        logger.exception("RCA job %s failed", job_id)
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.progress.append(traceback.format_exc(limit=3))
    finally:
        job.finished_at = utcnow()


def enqueue(incident: IncidentReport, background: BackgroundTasks) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], incident_id=incident.id)
    JOBS[job.id] = job
    background.add_task(run_investigation, job.id, incident)
    return job


def seed_demo_job() -> Job | None:
    """Put the standing report in the job list as a finished job.

    Off unless RCA_SEED_DEMO=1. A fresh server has no jobs, so the chat console
    would open empty and a visitor would have to wait out a full investigation
    (minutes on a CPU) before they could ask anything. With this on, the console
    opens on a real, already-finished report.
    """
    try:
        report = _standing_report()
    except HTTPException:
        logger.warning("RCA_SEED_DEMO is set but there is no standing report to seed")
        return None
    job = Job(
        id="demo",
        incident_id=report.incident.id,
        status="done",
        report=report,
        finished_at=utcnow(),
        progress=["Loaded saved report - no live investigation was run for this job."],
    )
    JOBS[job.id] = job
    return job


@app.on_event("startup")
def _maybe_seed() -> None:
    if os.environ.get("RCA_SEED_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}:
        seed_demo_job()


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class IncidentRequest(BaseModel):
    title: str
    description: str
    id: str | None = None
    reporter: str = "api"
    service_hint: str | None = None
    severity_hint: Literal["SEV1", "SEV2", "SEV3", "SEV4"] | None = None
    reported_at: datetime | None = None

    def to_incident(self) -> IncidentReport:
        return IncidentReport(
            id=self.id or f"API-{uuid.uuid4().hex[:8].upper()}",
            title=self.title,
            description=self.description,
            reporter=self.reporter,
            service_hint=self.service_hint,
            severity_hint=self.severity_hint,
            reported_at=self.reported_at or utcnow(),
        )


class JobAccepted(BaseModel):
    job_id: str
    incident_id: str
    status: str
    poll: str


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = build_sources()
    reachable, detail = build_client(settings).health()
    return {
        "status": "ok" if reachable else "degraded",
        "model": settings.model,
        "model_server": settings.llm_base_url,
        "model_server_reachable": reachable,
        "model_server_detail": detail,
        "strategy": settings.strategy,
        "sources": bundle.describe(),
        "services": bundle.logs.services(),
        "jobs": len(JOBS),
    }


@app.post("/incidents", response_model=JobAccepted, status_code=202)
def create_incident(payload: IncidentRequest, background: BackgroundTasks) -> JobAccepted:
    """File an incident and start an investigation in the background."""
    incident = payload.to_incident()
    job = enqueue(incident, background)
    return JobAccepted(
        job_id=job.id,
        incident_id=incident.id,
        status=job.status,
        poll=f"/jobs/{job.id}",
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}")
    return job


@app.get("/jobs/{job_id}/markdown")
def get_job_markdown(job_id: str) -> dict[str, str]:
    """The report as Markdown, ready to paste back onto the ticket."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}")
    if job.status != "done" or job.report is None:
        raise HTTPException(status_code=409, detail=f"Job is {job.status}, not done")
    return {"markdown": to_markdown(job.report)}


@app.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return [
        {
            "job_id": j.id,
            "incident_id": j.incident_id,
            "status": j.status,
            "created_at": j.created_at,
            "headline": j.report.analysis.headline if j.report else None,
        }
        for j in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)
    ]


# --------------------------------------------------------------------------
# Follow-up chat
#
# The report is a push: it lands before the developer starts looking. The chat
# is the pull that always follows - "why not the other PR?", "how sure are
# you?". It reads the evidence the investigation already collected; it does not
# collect more. See rca/chat.py for why that limit is deliberate.
# --------------------------------------------------------------------------


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ChatReply(BaseModel):
    answer: str
    citations: list[str]
    evidence: list[dict[str, str]]
    unverified: list[str]
    unsupported_times: list[str] = Field(default_factory=list)
    turns: int


class ChatState(BaseModel):
    job_id: str
    incident_id: str
    headline: str
    evidence: list[dict[str, str]]
    suggestions: list[str]
    turns: list[ChatTurn]


def _answerable(job_id: str) -> Job:
    """A job you can ask questions about: it exists and it finished."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}")
    if job.status != "done" or job.report is None:
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status}; there is no report to ask about yet.",
        )
    return job


def _evidence_rows(report: RCAReport) -> list[dict[str, str]]:
    return [
        {"id": item.id, "kind": item.kind, "summary": item.summary}
        for item in report.evidence
    ]


@app.get("/jobs/{job_id}/chat", response_model=ChatState)
def get_chat(job_id: str) -> ChatState:
    """The transcript so far, plus what the reader needs to ask a good question."""
    job = _answerable(job_id)
    assert job.report is not None
    return ChatState(
        job_id=job.id,
        incident_id=job.incident_id,
        headline=job.report.analysis.headline,
        evidence=_evidence_rows(job.report),
        suggestions=suggested_questions(job.report),
        turns=job.chat,
    )


@app.post("/jobs/{job_id}/chat", response_model=ChatReply)
def ask(job_id: str, payload: Question) -> ChatReply:
    """Ask a follow-up question about a finished report.

    Synchronous, unlike `/incidents`. An investigation is a twenty-step loop;
    an answer is one generation. Handing back a job ID for something the caller
    is sitting and waiting for would be worse, not better.
    """
    job = _answerable(job_id)
    assert job.report is not None

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="The question is empty.")

    with CHAT_LOCKS[job_id]:
        chat = ReportChat(job.report, client=build_client(settings), settings=settings)
        history = list(job.chat)
        try:
            result = chat.answer(question, history)
        except LLMError as exc:
            logger.warning("Chat on job %s failed: %s", job_id, exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    f"The model server at {settings.llm_base_url} could not answer: {exc}"
                ),
            ) from exc

        if result.unresolved:
            logger.warning(
                "Chat on job %s cited %d ID(s) that do not exist: %s",
                job_id,
                len(result.unresolved),
                ", ".join(result.unresolved),
            )

        job.chat.append(ChatTurn(role="user", content=question))
        job.chat.append(
            ChatTurn(role="assistant", content=result.text, citations=result.citations)
        )

    return ChatReply(
        answer=result.text,
        citations=result.citations,
        evidence=chat.resolve(result.citations),
        unverified=result.unresolved,
        unsupported_times=result.unsupported_times,
        turns=len(job.chat),
    )


@app.delete("/jobs/{job_id}/chat", status_code=204)
def reset_chat(job_id: str) -> None:
    """Clear the transcript. The report itself is untouched."""
    job = _answerable(job_id)
    with CHAT_LOCKS[job_id]:
        job.chat.clear()


class SimpleMessage(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class SimpleReply(BaseModel):
    reply: str
    citations: list[str]
    unverified: list[str]
    unsupported_times: list[str] = Field(default_factory=list)
    session_id: str
    incident_id: str


SESSIONS: dict[str, list[ChatTurn]] = {}
_STANDING: dict[str, RCAReport] = {}


def _standing_report() -> RCAReport:
    """The report `POST /chat` talks about, loaded once from `settings.chat_report`."""
    path = settings.chat_report
    cached = _STANDING.get(str(path))
    if cached is not None:
        return cached
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No report to chat about at {path}. Create one with "
                "`python -m rca.cli investigate INC-1042 --json-out <that path>`, "
                "or set RCA_CHAT_REPORT."
            ),
        )
    try:
        report = RCAReport.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"{path} is not a valid report: {exc}") from exc
    _STANDING[str(path)] = report
    return report


@app.post("/chat", response_model=SimpleReply)
def chat_simple(payload: SimpleMessage) -> SimpleReply:
    """Send one message, get one answer. No job ID, no polling.

    The shape a generic chatbot evaluator expects: a URL, a message in, a reply
    out. The conversation is about the standing report (`RCA_CHAT_REPORT`);
    pass the returned `session_id` back to keep the thread, omit it to start one.
    """
    report = _standing_report()
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="The message is empty.")

    session_id = payload.session_id or uuid.uuid4().hex[:12]
    with CHAT_LOCKS[f"session:{session_id}"]:
        history = SESSIONS.setdefault(session_id, [])
        chat = ReportChat(report, client=build_client(settings), settings=settings)
        try:
            result = chat.answer(message, history)
        except LLMError as exc:
            logger.warning("Chat session %s failed: %s", session_id, exc)
            raise HTTPException(
                status_code=503,
                detail=f"The model server at {settings.llm_base_url} could not answer: {exc}",
            ) from exc
        history.append(ChatTurn(role="user", content=message))
        history.append(
            ChatTurn(role="assistant", content=result.text, citations=result.citations)
        )

    return SimpleReply(
        reply=result.text,
        citations=result.citations,
        unverified=result.unresolved,
        unsupported_times=result.unsupported_times,
        session_id=session_id,
        incident_id=report.incident.id,
    )


@app.get("/chat", include_in_schema=False)
def chat_ui() -> FileResponse:
    """The chat console: one static page, no build step, no CDN."""
    page = STATIC_DIR / "chat.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail=f"UI not installed at {page}")
    return FileResponse(page, media_type="text/html")


# --------------------------------------------------------------------------
# Vendor webhooks
#
# Each adapter's only job is to find the title and the body in a vendor-shaped
# payload. Everything downstream is identical.
# --------------------------------------------------------------------------


@app.post("/webhooks/jira", response_model=JobAccepted, status_code=202)
def jira_webhook(payload: dict[str, Any], background: BackgroundTasks) -> JobAccepted:
    """Accepts a Jira `issue_created` automation payload."""
    issue = payload.get("issue") or {}
    fields = issue.get("fields") or {}
    summary = fields.get("summary")
    if not summary:
        raise HTTPException(status_code=422, detail="No issue.fields.summary in payload")

    description = fields.get("description") or ""
    if isinstance(description, dict):  # Atlassian Document Format
        description = _flatten_adf(description)

    incident = IncidentReport(
        id=issue.get("key") or f"JIRA-{uuid.uuid4().hex[:8].upper()}",
        title=summary,
        description=description or summary,
        reporter=(fields.get("reporter") or {}).get("displayName", "jira"),
        service_hint=(fields.get("components") or [{}])[0].get("name"),
        reported_at=_parse_or_now(fields.get("created")),
    )
    job = enqueue(incident, background)
    return JobAccepted(
        job_id=job.id, incident_id=incident.id, status=job.status, poll=f"/jobs/{job.id}"
    )


@app.post("/webhooks/sentry", response_model=JobAccepted, status_code=202)
def sentry_webhook(payload: dict[str, Any], background: BackgroundTasks) -> JobAccepted:
    """Accepts a Sentry issue alert webhook."""
    data = (payload.get("data") or {}).get("issue") or payload.get("data") or {}
    title = data.get("title") or payload.get("message")
    if not title:
        raise HTTPException(status_code=422, detail="No issue title in payload")

    culprit = data.get("culprit", "")
    count = data.get("count", "")
    incident = IncidentReport(
        id=f"SENTRY-{data.get('shortId') or uuid.uuid4().hex[:8].upper()}",
        title=title,
        description="\n".join(
            filter(None, [title, f"Culprit: {culprit}" if culprit else "", f"Events: {count}"])
        ),
        reporter="sentry",
        service_hint=(data.get("project") or {}).get("slug")
        if isinstance(data.get("project"), dict)
        else data.get("project"),
        reported_at=_parse_or_now(data.get("firstSeen")),
    )
    job = enqueue(incident, background)
    return JobAccepted(
        job_id=job.id, incident_id=incident.id, status=job.status, poll=f"/jobs/{job.id}"
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _parse_or_now(value: Any) -> datetime:
    if not isinstance(value, str):
        return utcnow()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return utcnow()
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _flatten_adf(node: Any) -> str:
    """Pull the plain text out of an Atlassian Document Format blob."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(filter(None, (_flatten_adf(n) for n in node)))
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        return _flatten_adf(node.get("content", []))
    return ""
