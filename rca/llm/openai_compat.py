"""A client for any server speaking the OpenAI chat-completions format.

That covers every self-hosted stack worth deploying:

    vLLM        python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B
    Ollama      ollama serve                    (base_url .../v1)
    TGI         text-generation-inference
    LM Studio   local server
    llama.cpp   llama-server

No API key is required by any of them; the OpenAI SDK insists on *some* string,
so we send a placeholder. Nothing leaves the network.

Structured output is where self-hosted servers differ most, so this class tries
the strict path first and degrades:

1. `response_format={"type": "json_schema", ...}` - vLLM, LM Studio, recent
   Ollama. Guided decoding makes the output schema-valid by construction.
2. `response_format={"type": "json_object"}` - older servers. Valid JSON, but
   the shape is only as good as the prompt.
3. Prompt-only, with the schema inlined. Last resort.

Whichever path runs, the result is validated against the schema and one repair
attempt is made before giving up, because open-weight models miss required
fields more often than hosted frontier models do.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rca.llm.base import (
    LLMError,
    LLMResponse,
    SchemaValidationError,
    ToolCall,
    ToolSpec,
    Usage,
)

# A fenced block, or the first balanced object in a chatty reply.
FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a reply that may be wrapped in prose or fences."""
    if not text or not text.strip():
        raise SchemaValidationError("The model returned an empty response.")

    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    fenced = FENCED_JSON.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1))

    # Fall back to scanning for the first balanced object.
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    preview = text[:200].replace("\n", " ")
    raise SchemaValidationError(f"No JSON object found in the response: {preview!r}")


def missing_fields(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Shallow required-field check - enough to trigger a repair round."""
    return [key for key in schema.get("required", []) if key not in payload]


class OpenAICompatClient:
    """Talks to a local or self-hosted OpenAI-compatible server."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
        timeout: float = 600.0,
        max_retries: int = 2,
        supports_json_schema: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "The `openai` package is required to talk to an OpenAI-compatible "
                "server. Install it with `pip install openai`. This does not mean "
                "you need an OpenAI account - the same wire format is what vLLM, "
                "Ollama, TGI and LM Studio all serve."
            ) from exc

        self.model = model
        self.base_url = base_url.rstrip("/")
        # Long timeout on purpose: a 70B doing a 24-step agentic loop on a busy
        # box is slow, and a premature timeout wastes the whole investigation.
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=max_retries,
        )
        self.supports_json_schema = supports_json_schema

    # -- protocol ----------------------------------------------------------

    def health(self, timeout: float = 4.0) -> tuple[bool, str]:
        """Probe the server. Fast and non-retrying on purpose.

        The generation timeout is minutes long because a 70B doing a 20-step
        loop is slow. A *health check* inheriting that is useless - it would
        hang a CLI or a readiness probe for the same minutes. So this gets its
        own short budget and no retries.
        """
        try:
            models = self._client.with_options(timeout=timeout, max_retries=0).models.list()
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return False, f"{self.base_url} unreachable ({type(exc).__name__})"

        names = [m.id for m in getattr(models, "data", [])]
        if not names:
            return True, f"{self.base_url} reachable, but it lists no models"
        if self.model in names:
            return True, f"{self.base_url} serving {self.model}"
        return (
            False,
            f"{self.base_url} is up but does not serve {self.model!r}. Available: "
            + ", ".join(names[:8]),
        )

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            # Deterministic by default: an RCA that changes its mind between
            # identical runs is not something an on-call engineer can trust.
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
            payload["tool_choice"] = "auto"
        if json_schema and self.supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        try:
            completion = self._client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001
            if json_schema and self.supports_json_schema and _looks_like_schema_refusal(exc):
                # The server does not implement guided decoding. Fall back once,
                # permanently, so we do not pay this round-trip every call.
                self.supports_json_schema = False
                return self.chat(
                    system=system,
                    messages=messages,
                    tools=tools,
                    json_schema=json_schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            raise LLMError(f"{self.base_url} request failed: {type(exc).__name__}: {exc}") from exc

        return self._to_response(completion, json_schema)

    # -- internals ---------------------------------------------------------

    def _to_response(self, completion: Any, json_schema: dict | None) -> LLMResponse:
        try:
            choice = completion.choices[0]
        except (AttributeError, IndexError) as exc:
            raise LLMError("The server returned no choices.") from exc

        message = choice.message
        usage = Usage(
            prompt_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
        )

        calls: list[ToolCall] = []
        for i, raw in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(raw, "function", None)
            if function is None:
                continue
            arguments = getattr(function, "arguments", "") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    # Keep the raw string; ToolSpec.call reports it properly.
                    arguments = {"__malformed__": arguments}
            calls.append(
                ToolCall(
                    id=getattr(raw, "id", None) or f"call_{i}",
                    name=getattr(function, "name", "") or "",
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        return LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=calls,
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            raw=completion,
        )


def _looks_like_schema_refusal(exc: Exception) -> bool:
    text = f"{exc}".lower()
    return any(
        marker in text
        for marker in ("response_format", "json_schema", "guided", "not supported", "unrecognized")
    )
