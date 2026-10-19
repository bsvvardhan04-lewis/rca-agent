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

import hmac
import logging
import os
import threading
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field

from rca.agent import RCAAgent
from rca.chat import ChatAnswer, ReportChat, suggested_questions
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
    if _flag("RCA_SEED_DEMO"):
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
def health(deep: bool = False) -> dict[str, Any]:
    """Readiness. `?deep=true` also runs a 4-token generation.

    Listing models only proves the server answers. An interrupted Ollama upgrade
    once left a server that listed every model and returned HTTP 500 on every
    generation - shallow health said "ok" the whole time. The deep probe costs a
    real (small) generation, so it is opt-in rather than what a poller hits.
    """
    bundle = build_sources()
    client = build_client(settings)
    reachable, detail = client.health()

    generation_ok: bool | None = None
    generation_detail = ""
    if deep and reachable:
        try:
            client.chat(
                system="Reply with the single word OK.",
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=4,
            )
            generation_ok = True
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            generation_ok = False
            generation_detail = f"{type(exc).__name__}: {str(exc)[:300]}"

    healthy = reachable and generation_ok is not False
    return {
        "status": "ok" if healthy else "degraded",
        "generation_ok": generation_ok,
        "generation_detail": generation_detail,
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


# --------------------------------------------------------------------------
# Evaluator-facing chat
#
# A generic chatbot evaluator cannot file an incident, wait for a job and fish a
# job ID out of the result. It has one URL and a message. These routes give it
# exactly that, in two shapes:
#
#   POST /chat                  {"message": ...}  ->  {"reply": ...}
#   POST /v1/chat/completions   the OpenAI chat format, which most evaluation
#                               harnesses already speak
#
# Both answer about the one standing report (RCA_CHAT_REPORT). Both are
# stateless for the caller: send the history you want considered with each
# request and nothing is kept on the server. (`POST /chat` additionally offers a
# server-side session_id, convenient for testing by hand.)
#
# Serving them to anyone but yourself: run `public_app`, not `app`. It carries
# only these routes - not the job list, not incident filing, not the webhooks -
# and it refuses to answer until a token is configured.
# --------------------------------------------------------------------------

evaluator = APIRouter()

MAX_HISTORY = 40
MAX_CONTENT_CHARS = 4000
EVALUATOR_MODEL_NAME = "rca-agent"


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _check_token(authorization: str | None, x_api_key: str | None, *, public: bool) -> None:
    expected = settings.chat_token
    if not expected:
        if public and not _flag("RCA_PUBLIC_OPEN"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "This endpoint is closed until RCA_CHAT_TOKEN is set. "
                    "(RCA_PUBLIC_OPEN=1 serves it with no authentication.)"
                ),
            )
        return

    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    elif x_api_key:
        supplied = x_api_key.strip()
    # Constant-time compare: the token guards a route that spends the host's CPU.
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid token. Send `Authorization: Bearer <token>`.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_token(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """Local use: open unless RCA_CHAT_TOKEN is set."""
    _check_token(authorization, x_api_key, public=False)


def require_token_public(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """Public use: fails closed - no token configured means no answers."""
    _check_token(authorization, x_api_key, public=True)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_CONTENT_CHARS)


class SimpleMessage(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    # When given, the call is fully stateless: this is the whole conversation so
    # far and the server neither reads nor writes a session.
    history: list[ChatMessage] | None = Field(default=None, max_length=MAX_HISTORY)


class SimpleReply(BaseModel):
    reply: str
    citations: list[str]
    unverified: list[str]
    unsupported_times: list[str] = Field(default_factory=list)
    session_id: str | None = None
    incident_id: str


SESSIONS: dict[str, list[ChatTurn]] = {}
_STANDING: dict[str, RCAReport] = {}


def _standing_report() -> RCAReport:
    """The report the evaluator routes talk about, loaded once from `settings.chat_report`."""
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


def _ask(report: RCAReport, message: str, history: list[ChatTurn], *, who: str) -> ChatAnswer:
    """One grounded answer. A dead model server is a 503, not a crash."""
    chat = ReportChat(report, client=build_client(settings), settings=settings)
    try:
        return chat.answer(message, history)
    except LLMError as exc:
        logger.warning("Chat (%s) failed: %s", who, exc)
        raise HTTPException(
            status_code=503,
            detail=f"The model server at {settings.llm_base_url} could not answer: {exc}",
        ) from exc


@evaluator.post("/chat", response_model=SimpleReply)
def chat_simple(payload: SimpleMessage) -> SimpleReply:
    """Send one message, get one answer. No job ID, no polling.

    Stateless when `history` is supplied. Otherwise the server keeps the thread
    under `session_id` (returned; pass it back to continue, omit it to start).
    """
    report = _standing_report()
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="The message is empty.")

    if payload.history is not None:
        turns = [ChatTurn(role=m.role, content=m.content) for m in payload.history]
        result = _ask(report, message, turns, who="stateless")
        session_id = None
    else:
        session_id = payload.session_id or uuid.uuid4().hex[:12]
        with CHAT_LOCKS[f"session:{session_id}"]:
            thread = SESSIONS.setdefault(session_id, [])
            result = _ask(report, message, thread, who=f"session {session_id}")
            thread.append(ChatTurn(role="user", content=message))
            thread.append(
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


class OAIMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class OAIRequest(BaseModel):
    model: str | None = None
    messages: list[OAIMessage] = Field(min_length=1, max_length=MAX_HISTORY * 2)
    stream: bool = False
    # temperature, max_tokens, top_p and the rest are accepted and ignored: the
    # service fixes its own sampling so an answer does not depend on the caller.


def _oai_text(content: str | list[dict[str, Any]] | None) -> str:
    """Message content as text. Newer clients send a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") in (None, "text")
    )


@evaluator.post("/v1/chat/completions")
def chat_completions(payload: OAIRequest) -> dict[str, Any]:
    """OpenAI-compatible, non-streaming, stateless.

    The caller's `system` / `developer` / `tool` messages are dropped on
    purpose: the grounding and scope rules are this service's, and a request
    must not be able to replace them. Only user and assistant turns count, and
    the last must be a user turn.
    """
    if payload.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported; send stream=false.")

    turns: list[ChatTurn] = []
    for message in payload.messages:
        if message.role not in ("user", "assistant"):
            continue
        text = _oai_text(message.content).strip()
        if not text:
            continue
        if len(text) > MAX_CONTENT_CHARS:
            raise HTTPException(
                status_code=400,
                detail=f"A message is over the {MAX_CONTENT_CHARS}-character limit.",
            )
        turns.append(ChatTurn(role=message.role, content=text))  # type: ignore[arg-type]

    if not turns or turns[-1].role != "user":
        raise HTTPException(status_code=400, detail="The last message must be a non-empty user message.")

    report = _standing_report()
    result = _ask(report, turns[-1].content, turns[:-1], who="openai")
    usage = result.usage
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model or EVALUATOR_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total,
        },
        # Not part of the OpenAI schema; clients that do not know it ignore it.
        "rca": {
            "incident_id": report.incident.id,
            "underlying_model": settings.model,
            "citations": result.citations,
            "unverified": result.unresolved,
            "unsupported_times": result.unsupported_times,
        },
    }


@evaluator.get("/v1/models")
def list_models() -> dict[str, Any]:
    """Many OpenAI-format clients list models first as a connectivity check."""
    return {
        "object": "list",
        "data": [
            {"id": EVALUATOR_MODEL_NAME, "object": "model", "created": 0, "owned_by": "team-7"}
        ],
    }


async def _openai_style_errors(request: Request, exc: StarletteHTTPException) -> Response:
    """Under /v1/, errors use OpenAI's `{"error": {...}}` envelope.

    OpenAI-format clients look for that key to show a useful message; without it
    they report only a bare status code.
    """
    if request.url.path.startswith("/v1/"):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
                    "code": exc.status_code,
                }
            },
            headers=getattr(exc, "headers", None),
        )
    return await http_exception_handler(request, exc)


app.include_router(evaluator, dependencies=[Depends(require_token)])
app.add_exception_handler(StarletteHTTPException, _openai_style_errors)

# What to put on the internet. Run with: uvicorn api.server:public_app --port 8001
public_app = FastAPI(
    title="RCA Agent - evaluator endpoints",
    description="The chat routes only. No incident filing, job list or webhooks.",
    version=app.version,
)
public_app.include_router(evaluator, dependencies=[Depends(require_token_public)])
public_app.add_exception_handler(StarletteHTTPException, _openai_style_errors)


@public_app.get("/health")
def public_health() -> dict[str, Any]:
    """Liveness for whoever is checking the tunnel. Reveals nothing internal."""
    reachable, _detail = build_client(settings).health()
    return {"status": "ok" if reachable else "degraded"}


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
