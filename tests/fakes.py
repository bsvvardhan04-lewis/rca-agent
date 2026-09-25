"""A scripted `LLMClient` so the pipeline runs with no model server.

This is not a mock of HTTP. It implements the same protocol the real client
does and, in the tool loop, hands back tool calls that the agent then executes
**for real** against the seeded data. So the tests cover the parts that
actually break: schema generation, tool dispatch, argument handling, evidence
IDs, citation resolution and strategy selection.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from rca.llm.base import LLMResponse, ToolCall, ToolSpec, Usage


class ScriptedLLM:
    """Replays a script of turns.

    Structured calls (those carrying a `json_schema`) are answered from
    `parsed`, keyed by the Pydantic model's title. Tool turns come from
    `tool_script`: a list of batches, each batch a list of (tool_name, args)
    executed in one assistant turn. When the script runs out, the client
    returns `final_findings` as plain text, which ends the loop.
    """

    def __init__(
        self,
        *,
        parsed: dict[str, BaseModel],
        tool_script: list[list[tuple[str, dict]]] | None = None,
        final_findings: str = "Investigation complete.",
        prompt_tokens: int = 900,
        completion_tokens: int = 220,
    ):
        self.model = "scripted-model"
        self._parsed = parsed
        self._tool_script = list(tool_script or [])
        self.final_findings = final_findings
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

        self.calls: list[str] = []
        self.systems: list[str] = []
        self.tools_offered: list[list[str]] = []

    def health(self) -> tuple[bool, str]:
        return True, "scripted client"

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
        self.systems.append(system)
        usage = Usage(
            prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens
        )

        if json_schema is not None:
            title = json_schema.get("title", "")
            self.calls.append(f"structured:{title}")
            model = self._parsed.get(title)
            if model is None:
                raise AssertionError(f"no scripted response for schema {title!r}")
            return LLMResponse(text=model.model_dump_json(), usage=usage)

        if tools:
            self.tools_offered.append([t.name for t in tools])

        if self._tool_script:
            batch = self._tool_script.pop(0)
            self.calls.append("tools:" + ",".join(name for name, _ in batch))
            return LLMResponse(
                tool_calls=[
                    ToolCall(id=f"call_{i}", name=name, arguments=args)
                    for i, (name, args) in enumerate(batch)
                ],
                usage=usage,
                finish_reason="tool_calls",
            )

        self.calls.append("text")
        return LLMResponse(text=self.final_findings, usage=usage)


class RefusingToolLLM(ScriptedLLM):
    """Answers with prose and never calls a tool - what a weak model does."""

    def chat(self, **kwargs) -> LLMResponse:
        if kwargs.get("json_schema") is not None:
            return super().chat(**kwargs)
        self.calls.append("text-no-tools")
        return LLMResponse(
            text=self.final_findings,
            usage=Usage(prompt_tokens=self.prompt_tokens, completion_tokens=50),
        )


class BadJSONLLM:
    """Fails schema validation once, then succeeds - exercises the repair round."""

    def __init__(self, *, good: BaseModel, bad_payload: str = "not json at all"):
        self.model = "bad-json-model"
        self._good = good
        self._bad = bad_payload
        self.attempts = 0

    def health(self) -> tuple[bool, str]:
        return True, "bad json client"

    def chat(self, *, json_schema=None, **_kwargs) -> LLMResponse:
        self.attempts += 1
        usage = Usage(prompt_tokens=100, completion_tokens=20)
        if self.attempts == 1:
            return LLMResponse(text=self._bad, usage=usage)
        return LLMResponse(text=self._good.model_dump_json(), usage=usage)


class MissingFieldLLM(BadJSONLLM):
    """Returns valid JSON that is missing a required field, then repairs."""

    def chat(self, *, json_schema=None, **_kwargs) -> LLMResponse:
        self.attempts += 1
        usage = Usage(prompt_tokens=100, completion_tokens=20)
        if self.attempts == 1:
            return LLMResponse(text=json.dumps({"summary": "only this one field"}), usage=usage)
        return LLMResponse(text=self._good.model_dump_json(), usage=usage)


class AlwaysBadLLM(BadJSONLLM):
    """Never produces the schema - two failures must raise, not loop."""

    def chat(self, *, json_schema=None, **_kwargs) -> LLMResponse:
        self.attempts += 1
        return LLMResponse(text="I cannot do that.", usage=Usage(prompt_tokens=10))
