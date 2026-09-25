"""Source wiring: pick real adapters where configured, files everywhere else."""

from __future__ import annotations

from rca.config import Settings, settings as default_settings
from rca.sources.base import SourceBundle
from rca.sources.file_sources import FileAlertSource, FileChangeSource, FileLogSource
from rca.sources.git_source import GitChangeSource, GitError

__all__ = [
    "SourceBundle",
    "FileLogSource",
    "FileChangeSource",
    "FileAlertSource",
    "GitChangeSource",
    "GitError",
    "build_sources",
]


def build_sources(settings: Settings | None = None, *, strict: bool = False) -> SourceBundle:
    """Assemble the evidence sources this deployment should use.

    `RCA_GIT_REPO` swaps the demo change file for a real clone. If that repo is
    unusable we fall back to the file source and say so, because an RCA that
    runs with two of three sources beats one that refuses to start - unless
    `strict` is set, in which case the misconfiguration is raised.
    """
    s = settings or default_settings

    changes: FileChangeSource | GitChangeSource = FileChangeSource(s.changes_dir)
    if s.git_repo:
        try:
            changes = GitChangeSource(
                s.git_repo,
                url_template=(
                    f"https://github.com/{s.github_repo}/pull/{{id}}" if s.github_repo else ""
                ),
            )
        except GitError:
            if strict:
                raise

    return SourceBundle(
        logs=FileLogSource(s.logs_dir),
        changes=changes,
        alerts=FileAlertSource(s.alerts_dir),
    )
