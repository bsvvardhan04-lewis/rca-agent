"""CLI smoke tests.

These exist because a syntax error in `rca/cli.py` once passed the whole suite:
nothing imported it. Every entry point needs at least one test that loads it.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rca import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def patched(monkeypatch, settings):
    """Point the CLI at the test data and a stubbed model server."""

    class StubClient:
        model = settings.model

        def health(self):
            return False, "http://localhost:11434/v1 unreachable (APIConnectionError)"

    monkeypatch.setattr(cli, "settings", settings)
    monkeypatch.setattr(cli, "build_client", lambda *_a, **_k: StubClient())
    return settings


def test_the_module_imports(patched):
    """A syntax error here used to pass every other test."""
    assert cli.app is not None


def test_help_lists_the_commands(runner):
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("sources", "incidents", "investigate"):
        assert command in result.stdout


def test_sources_reports_data_and_model_server(runner, patched):
    result = runner.invoke(cli.app, ["sources"])
    assert result.exit_code == 0
    assert "payments-api" in result.stdout
    assert patched.model in result.stdout
    # With no server running it must say so, and say how to start one.
    assert "unreachable" in result.stdout
    assert "ollama serve" in result.stdout
    assert "vllm serve" in result.stdout


def test_incidents_lists_the_seeded_ticket(runner, patched):
    result = runner.invoke(cli.app, ["incidents"])
    assert result.exit_code == 0
    assert "INC-1042" in result.stdout


def test_investigate_without_arguments_is_a_usage_error(runner, patched):
    result = runner.invoke(cli.app, ["investigate"])
    assert result.exit_code != 0


def test_investigate_with_an_unknown_incident_id_explains_itself(runner, patched):
    result = runner.invoke(cli.app, ["investigate", "INC-DOES-NOT-EXIST"])
    assert result.exit_code != 0
    # Typer writes usage errors to stderr, which the runner keeps separate.
    written = (result.output or "") + (result.stderr or "")
    assert "No incident file" in written
    assert "seed_demo" in written, "the message must say how to fix it"
