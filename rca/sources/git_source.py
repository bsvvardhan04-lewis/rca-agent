"""A ChangeSource backed by a real git repository.

This is the first source that touches something other than seeded files, so it
is where the honest caveats live:

**Git does not know about deploys.** It knows when a commit was authored and
when it landed on a branch. The investigator is told to correlate an incident
against *deploy* time, because that is what actually changes production - and a
bare git repo cannot supply it. So `deployed_at` is left `None`, and
`list_changes` falls back to merge time.

That fallback is a real loss of precision: code that merged on Friday and
deployed on Monday will look like a Friday change. If you have a deploy feed
(a CD webhook, a `deploy-*` tag convention, an Argo/Spinnaker API), wire it in
via `deploy_times` and the correlation becomes correct again.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

from rca.models import CodeChange, parse_ts
from rca.sources.base import matches

# GitHub and GitLab both write the request number into squash-merge subjects:
#   "Add merchant risk scoring (#4821)"
#   "Merge pull request #4821 from acme/risk-scoring"
PR_IN_SUBJECT = re.compile(r"\(#(\d+)\)\s*$")
PR_IN_MERGE_COMMIT = re.compile(r"^Merge pull request #(\d+)\b")

# Conventional monorepo layouts: the segment after these is the service name.
SERVICE_PREFIXES = ("services", "apps", "packages", "cmd", "microservices")

# Unit separator - will not appear in a commit subject, unlike commas or pipes.
FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"

GIT_FORMAT = FIELD_SEP.join(["%H", "%h", "%cI", "%an", "%s", "%b"]) + RECORD_SEP


class GitError(RuntimeError):
    """Raised when the repository is unusable - missing, not a repo, no git binary."""


def _run_git(repo: Path, args: list[str], timeout: float = 30.0) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out after {timeout}s") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else f"exit {result.returncode}"
        raise GitError(f"git {' '.join(args[:2])} failed: {detail}")
    return result.stdout


def infer_services(paths: list[str], extra: dict[str, str] | None = None) -> list[str]:
    """Guess which services a set of changed paths belongs to.

    `extra` maps a path prefix to a service name and wins over the heuristic,
    for repos that do not follow a `services/<name>/` convention.
    """
    found: list[str] = []
    for path in paths:
        normalised = path.replace("\\", "/")

        matched = None
        for prefix, service in (extra or {}).items():
            if normalised.startswith(prefix.replace("\\", "/")):
                matched = service
                break
        if matched is None:
            segments = [s for s in normalised.split("/") if s]
            if len(segments) >= 2 and segments[0] in SERVICE_PREFIXES:
                matched = segments[1]

        if matched and matched not in found:
            found.append(matched)
    return found


def extract_change_id(subject: str, body: str, short_sha: str) -> str:
    """Prefer a PR number over a SHA - it is what a human will search for."""
    match = PR_IN_SUBJECT.search(subject) or PR_IN_MERGE_COMMIT.match(subject)
    if match:
        return f"PR-{match.group(1)}"
    for line in body.splitlines():
        match = PR_IN_MERGE_COMMIT.match(line.strip())
        if match:
            return f"PR-{match.group(1)}"
    return short_sha


class GitChangeSource:
    """Reads merged commits out of a local clone.

    Args:
        repo: Path to a git working tree.
        service_map: Optional `{path_prefix: service_name}` overrides.
        deploy_times: Optional `{change_id: datetime}` from a real deploy feed.
            Supplying this is what makes deploy-time correlation accurate.
        url_template: Optional `str.format`-style template with `{id}` for
            linking back, e.g. "https://github.com/acme/platform/pull/{id}".
        max_commits: Ceiling on how many commits one query will walk.
    """

    name = "git"

    def __init__(
        self,
        repo: str | Path,
        *,
        service_map: dict[str, str] | None = None,
        deploy_times: dict[str, datetime] | None = None,
        url_template: str = "",
        max_commits: int = 200,
    ):
        self.repo = Path(repo).expanduser().resolve()
        self.service_map = service_map or {}
        self.deploy_times = deploy_times or {}
        self.url_template = url_template
        self.max_commits = max_commits
        self._verify()

    def _verify(self) -> None:
        if not self.repo.exists():
            raise GitError(f"No such directory: {self.repo}")
        try:
            _run_git(self.repo, ["rev-parse", "--git-dir"])
        except GitError as exc:
            raise GitError(f"{self.repo} is not a git repository ({exc})") from exc

    # -- parsing -----------------------------------------------------------

    def _numstat(self, sha: str) -> tuple[list[str], int, int]:
        raw = _run_git(
            self.repo, ["show", "--numstat", "--format=", "--no-renames", sha]
        )
        files: list[str] = []
        additions = deletions = 0
        for line in raw.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, removed, path = parts
            files.append(path)
            # Binary files report "-" rather than a count.
            additions += int(added) if added.isdigit() else 0
            deletions += int(removed) if removed.isdigit() else 0
        return files, additions, deletions

    def _commits_between(self, start: datetime, end: datetime) -> list[CodeChange]:
        raw = _run_git(
            self.repo,
            [
                "log",
                f"--since={start.isoformat()}",
                f"--until={end.isoformat()}",
                f"--max-count={self.max_commits}",
                f"--format={GIT_FORMAT}",
                "--date-order",
            ],
        )

        changes: list[CodeChange] = []
        for record in raw.split(RECORD_SEP):
            record = record.strip("\n")
            if not record.strip():
                continue
            fields = record.split(FIELD_SEP)
            if len(fields) < 5:
                continue
            sha, short_sha, committed, author, subject = fields[:5]
            body = fields[5] if len(fields) > 5 else ""

            files, additions, deletions = self._numstat(sha)
            change_id = extract_change_id(subject, body, short_sha)

            changes.append(
                CodeChange(
                    id=change_id,
                    title=PR_IN_SUBJECT.sub("", subject).strip() or subject,
                    author=author,
                    merged_at=parse_ts(committed),
                    deployed_at=self.deploy_times.get(change_id),
                    services=infer_services(files, self.service_map),
                    files_changed=files,
                    additions=additions,
                    deletions=deletions,
                    url=self.url_template.format(id=change_id.removeprefix("PR-"))
                    if self.url_template
                    else None,
                    description=body.strip(),
                )
            )
        return changes

    # -- ChangeSource protocol ---------------------------------------------

    def list_changes(
        self,
        *,
        start: datetime,
        end: datetime,
        services: list[str] | None = None,
        query: str = "",
    ) -> list[CodeChange]:
        changes = self._commits_between(start, end)

        wanted = {s.lower() for s in services} if services else None
        out: list[CodeChange] = []
        for change in changes:
            # A change with no known services is *unattributed*, not
            # irrelevant - a root config or shared library can break any
            # service. Filtering it out silently during an incident is the
            # worse failure, so it survives the filter and the tool layer
            # flags it so the agent can weigh it for itself.
            if wanted and change.services and not ({s.lower() for s in change.services} & wanted):
                continue
            if query:
                blob = " ".join(
                    [change.title, change.description, change.author, *change.files_changed]
                )
                if not matches(blob, query):
                    continue
            out.append(change)
        out.sort(key=lambda c: c.deployed_at or c.merged_at, reverse=True)
        return out

    def get_change(self, change_id: str) -> CodeChange | None:
        needle = change_id.strip()

        # A PR number needs a log search; a SHA can be resolved directly.
        if needle.upper().startswith("PR-") or needle.isdigit():
            number = needle.upper().removeprefix("PR-")
            raw = _run_git(
                self.repo,
                [
                    "log",
                    "--max-count=1",
                    f"--grep=#{number}\\b",
                    "--extended-regexp",
                    "--format=%H",
                ],
            ).strip()
            if not raw:
                return None
            sha = raw.splitlines()[0]
        else:
            try:
                sha = _run_git(self.repo, ["rev-parse", "--verify", f"{needle}^{{commit}}"]).strip()
            except GitError:
                return None

        raw = _run_git(self.repo, ["show", "--no-patch", f"--format={GIT_FORMAT}", sha])
        fields = raw.split(RECORD_SEP)[0].split(FIELD_SEP)
        if len(fields) < 5:
            return None
        full_sha, short_sha, committed, author, subject = fields[:5]
        body = fields[5] if len(fields) > 5 else ""

        files, additions, deletions = self._numstat(full_sha)
        resolved_id = extract_change_id(subject, body, short_sha)
        return CodeChange(
            id=resolved_id,
            title=PR_IN_SUBJECT.sub("", subject).strip() or subject,
            author=author,
            merged_at=parse_ts(committed),
            deployed_at=self.deploy_times.get(resolved_id),
            services=infer_services(files, self.service_map),
            files_changed=files,
            additions=additions,
            deletions=deletions,
            url=self.url_template.format(id=resolved_id.removeprefix("PR-"))
            if self.url_template
            else None,
            description=body.strip(),
        )
