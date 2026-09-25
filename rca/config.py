"""Runtime configuration, loaded from the environment (and optionally a .env file).

The model runs on infrastructure you control. There is no API key and no
third-party call: incident data, log contents and source-code metadata stay
inside the network. For a consultancy handling client systems that is the
point, not a side effect.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency - the project still runs without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sensible defaults for a laptop running Ollama. Production points at vLLM
# serving something in the 32B-70B class - see README.
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"

VALID_STRATEGIES = {"agentic", "guided", "auto"}


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _int_env(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # -- model ----------------------------------------------------------
    model: str = field(default_factory=lambda: _env("RCA_MODEL", DEFAULT_MODEL))
    llm_base_url: str = field(default_factory=lambda: _env("RCA_BASE_URL", DEFAULT_BASE_URL))
    # Self-hosted servers ignore this; the OpenAI SDK requires a non-empty string.
    llm_api_key: str = field(default_factory=lambda: _env("RCA_API_KEY", "not-needed"))
    llm_timeout: float = field(default_factory=lambda: _float_env("RCA_TIMEOUT", 600.0))
    # vLLM and recent Ollama support guided decoding. Set false for older servers.
    llm_json_schema: bool = field(default_factory=lambda: _bool_env("RCA_JSON_SCHEMA", True))

    # -- investigation strategy -----------------------------------------
    # agentic: the model chooses which tool to call next. Needs a capable model.
    # guided:  code runs the proven query sequence; the model reads the results.
    # auto:    try agentic, fall back to guided if the model flounders.
    strategy: str = field(default_factory=lambda: _env("RCA_STRATEGY", "auto"))

    # -- data -----------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(_env("RCA_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
    )
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    # Optional real sources. Blank means "use the files in data/".
    # A local clone needs no token - git handles its own auth - so
    # github_repo is only ever used to build links back to PRs.
    git_repo: str = field(default_factory=lambda: _env("RCA_GIT_REPO"))
    github_repo: str = field(default_factory=lambda: _env("RCA_GITHUB_REPO"))

    # -- guard rails ----------------------------------------------------
    max_tool_iterations: int = field(
        default_factory=lambda: _int_env("RCA_MAX_TOOL_ITERATIONS", 20)
    )
    max_log_lines: int = field(default_factory=lambda: _int_env("RCA_MAX_LOG_LINES", 60))
    max_tokens: int = field(default_factory=lambda: _int_env("RCA_MAX_TOKENS", 4096))

    def __post_init__(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"RCA_STRATEGY={self.strategy!r} is not one of {sorted(VALID_STRATEGIES)}"
            )
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def alerts_dir(self) -> Path:
        return self.data_dir / "alerts"

    @property
    def changes_dir(self) -> Path:
        return self.data_dir / "changes"

    def describe_model(self) -> str:
        return f"{self.model} @ {self.llm_base_url} (strategy: {self.strategy})"


settings = Settings()
