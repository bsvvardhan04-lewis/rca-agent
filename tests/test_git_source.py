"""Git source tests, run against a real throwaway repository.

Mocking `subprocess` here would only prove that the mock matches the mock. So
these build an actual repo with actual commits at controlled timestamps.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rca.config import Settings
from rca.sources import build_sources
from rca.sources.file_sources import FileChangeSource
from rca.sources.git_source import (
    GitChangeSource,
    GitError,
    extract_change_id,
    infer_services,
)

UTC = timezone.utc
BASE = datetime(2026, 9, 23, 10, 0, 0, tzinfo=UTC)

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not installed"
)


def commit(repo: Path, when: datetime, subject: str, files: dict[str, str], body: str = ""):
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    stamp = when.isoformat()
    env = {
        "GIT_AUTHOR_DATE": stamp,
        "GIT_COMMITTER_DATE": stamp,
        "GIT_AUTHOR_NAME": "Priya Nair",
        "GIT_AUTHOR_EMAIL": "priya@example.com",
        "GIT_COMMITTER_NAME": "Priya Nair",
        "GIT_COMMITTER_EMAIL": "priya@example.com",
    }
    import os

    message = subject if not body else f"{subject}\n\n{body}"
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", message, "--no-gpg-sign"],
        check=True,
        capture_output=True,
        env={**os.environ, **env},
    )


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("repo")
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )

    commit(path, BASE, "Initial commit", {"README.md": "# platform\n"})
    commit(
        path,
        BASE + timedelta(hours=1),
        "Festive banner copy swap (#4817)",
        {"apps/checkout-web/Banner.tsx": "banner\n"},
    )
    commit(
        path,
        BASE + timedelta(hours=2),
        "Add merchant risk scoring to payment authorization (#4821)",
        {
            "services/payments/auth.py": "def authorize():\n    pass\n" * 10,
            "services/payments/db/transaction.py": "tx\n" * 5,
        },
        body="Calls risk-service inside tx_authorize to keep the write atomic.",
    )
    commit(
        path,
        BASE + timedelta(hours=3),
        "Merge pull request #4822 from acme/docs",
        {"CODEOWNERS": "* @team\n"},
    )
    return path


@pytest.fixture()
def source(repo: Path) -> GitChangeSource:
    return GitChangeSource(repo, url_template="https://github.com/acme/platform/pull/{id}")


# -- unit bits ------------------------------------------------------------


@pytest.mark.parametrize(
    "subject, body, expected",
    [
        ("Add risk scoring (#4821)", "", "PR-4821"),
        ("Merge pull request #4822 from acme/docs", "", "PR-4822"),
        ("Plain commit with no number", "", "abc1234"),
        ("Squashed", "Merge pull request #99 from x/y", "PR-99"),
        ("Mentions #123 mid-sentence but is not a PR", "", "abc1234"),
    ],
)
def test_change_id_prefers_a_pr_number(subject, body, expected):
    assert extract_change_id(subject, body, "abc1234") == expected


def test_service_inference_from_monorepo_layout():
    assert infer_services(["services/payments/auth.py", "services/payments/db/tx.py"]) == [
        "payments"
    ]
    assert infer_services(["apps/checkout-web/Banner.tsx"]) == ["checkout-web"]
    assert infer_services(["README.md"]) == []


def test_service_map_overrides_the_heuristic():
    assert infer_services(["README.md"], {"README": "docs"}) == ["docs"]
    assert infer_services(["src/pay/x.go"], {"src/pay": "payments-api"}) == ["payments-api"]


def test_windows_style_paths_are_normalised():
    assert infer_services(["services\\payments\\auth.py"]) == ["payments"]


# -- against the real repo ------------------------------------------------


def test_commits_in_window_are_returned_newest_first(source):
    changes = source.list_changes(start=BASE, end=BASE + timedelta(hours=4))
    assert [c.id for c in changes] == ["PR-4822", "PR-4821", "PR-4817", changes[-1].id]
    assert changes[-1].title == "Initial commit"


def test_the_window_actually_filters(source):
    changes = source.list_changes(
        start=BASE + timedelta(hours=1, minutes=30), end=BASE + timedelta(hours=2, minutes=30)
    )
    assert [c.id for c in changes] == ["PR-4821"]


def test_pr_number_is_stripped_from_the_title(source):
    change = source.get_change("PR-4821")
    assert change.title == "Add merchant risk scoring to payment authorization"
    assert "#4821" not in change.title


def test_file_list_and_line_counts_come_from_the_diff(source):
    change = source.get_change("PR-4821")
    assert sorted(change.files_changed) == [
        "services/payments/auth.py",
        "services/payments/db/transaction.py",
    ]
    assert change.additions == 25
    assert change.deletions == 0


def test_services_are_inferred_and_the_body_is_kept(source):
    change = source.get_change("PR-4821")
    assert change.services == ["payments"]
    assert "inside tx_authorize" in change.description


def test_url_template_is_applied(source):
    assert source.get_change("PR-4821").url == "https://github.com/acme/platform/pull/4821"


def test_a_change_can_be_fetched_by_bare_number_or_sha(source, repo):
    assert source.get_change("4821").id == "PR-4821"
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert source.get_change(sha).id == "PR-4822"


def test_unknown_ids_return_none_rather_than_raising(source):
    assert source.get_change("PR-99999") is None
    assert source.get_change("deadbeef") is None


def test_service_filter_excludes_other_services(source):
    payments = source.list_changes(
        start=BASE, end=BASE + timedelta(hours=4), services=["payments"]
    )
    ids = [c.id for c in payments]
    assert "PR-4821" in ids
    # The checkout-web change was attributed elsewhere, so it is filtered out.
    assert "PR-4817" not in ids


def test_unattributable_changes_survive_a_service_filter(source):
    """A root-level change could break any service, so it is never hidden."""
    payments = source.list_changes(
        start=BASE, end=BASE + timedelta(hours=4), services=["payments"]
    )
    survivors = {c.id for c in payments if not c.services}
    # PR-4822 touches CODEOWNERS and the initial commit touches README: neither
    # maps to a service, so neither can be ruled out on service grounds.
    assert "PR-4822" in survivors


def test_query_searches_the_body_and_file_paths(source):
    window = {"start": BASE, "end": BASE + timedelta(hours=4)}
    assert [c.id for c in source.list_changes(query="tx_authorize", **window)] == ["PR-4821"]
    assert [c.id for c in source.list_changes(query="transaction.py", **window)] == ["PR-4821"]


def test_deploy_times_when_supplied_drive_the_ordering(repo):
    # Without a deploy feed, PR-4822 (newest merge) sorts first. With one that
    # says PR-4821 shipped later, the order flips - which is the whole point.
    deployed = GitChangeSource(
        repo, deploy_times={"PR-4821": BASE + timedelta(hours=5)}
    )
    changes = deployed.list_changes(start=BASE, end=BASE + timedelta(hours=4))
    assert changes[0].id == "PR-4821"
    assert changes[0].deployed_at == BASE + timedelta(hours=5)


def test_no_deploy_feed_means_no_deploy_time(source):
    assert source.get_change("PR-4821").deployed_at is None


# -- wiring ---------------------------------------------------------------


def test_a_bad_repo_path_is_rejected(tmp_path):
    with pytest.raises(GitError, match="No such directory"):
        GitChangeSource(tmp_path / "nope")

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(GitError, match="not a git repository"):
        GitChangeSource(plain)


def test_build_sources_uses_git_when_configured(repo, data_dir):
    s = Settings()
    s.data_dir = data_dir
    s.git_repo = str(repo)
    bundle = build_sources(s)
    assert isinstance(bundle.changes, GitChangeSource)
    assert "git" in bundle.describe()


def test_build_sources_falls_back_when_the_repo_is_unusable(tmp_path, data_dir):
    s = Settings()
    s.data_dir = data_dir
    s.git_repo = str(tmp_path / "does-not-exist")

    bundle = build_sources(s)
    assert isinstance(bundle.changes, FileChangeSource), "degrade, do not refuse to start"

    with pytest.raises(GitError):
        build_sources(s, strict=True)


def test_build_sources_defaults_to_files(data_dir):
    s = Settings()
    s.data_dir = data_dir
    assert isinstance(build_sources(s).changes, FileChangeSource)
