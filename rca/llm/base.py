"""Provider-neutral LLM interface.

The agent talks to this, never to a vendor SDK. That matters for two reasons:

* **Deployment freedom.** The same code runs against vLLM on a GPU box, Ollama
  on a laptop, TGI, LM Studio or llama.cpp - they all speak the OpenAI chat
  format. Swapping model or host is a config change, not a rewrite.
* **Testability.** The whole pipeline can be driven by a scripted client with
  no server running, which is how the test suite covers the agent loop.

Nothing here imports a vendor package.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Any failure talking to the model server, or an unusable response."""


class SchemaValidationError(LLMError):
    """The model returned JSON that does not satisfy the requested schema."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolSpec:
    """A callable tool plus the JSON schema the model is shown.

    Vendor-neutral on purpose: every server we target accepts this shape as
    `{"type": "function", "function": {name, description, parameters}}`.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]

    def call(self, arguments: dict[str, Any] | str) -> str:
        """Invoke the tool, tolerating the ways models mangle arguments.

        Open-weight models fairly often send arguments as a JSON *string*
        rather than an object, or include keys the schema does not declare.
        Neither is worth failing a whole investigation over.
        """
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise LLMError(f"{self.name}: arguments were not valid JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise LLMError(f"{self.name}: expected an object of arguments, got {type(arguments)}")

        known = set(self.input_schema.get("properties", {}))
        cleaned = {k: v for k, v in arguments.items() if k in known}
        return self.fn(**cleaned)

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@runtime_checkable
class LLMClient(Protocol):
    """What the agent needs from a model server."""

    model: str

    def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """One turn. `json_schema` constrains the reply to that shape."""

    def health(self) -> tuple[bool, str]:
        """(reachable, human-readable detail) - used by `rca.cli sources`."""
