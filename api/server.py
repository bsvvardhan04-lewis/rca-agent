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
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rca.agent import RCAAgent
from rca.config import settings
from rca.llm import build_client
from rca.models import IncidentReport, RCAReport, utcnow
from rca.report import to_markdown
from rca.sources import SourceBundle, build_sources

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


JOBS: dict[str, Job] = {}


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
