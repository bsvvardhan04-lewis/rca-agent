"""Structured output with validation and one repair round.

Hosted frontier models usually return schema-valid JSON on the first try.
Open-weight models miss a required field, invent an extra one, or wrap the
object in commentary often enough that a single attempt is not good enough to
build a pipeline on.

So: ask, parse, validate against the Pydantic model, and if it fails, show the
model its own output plus the specific complaint and ask once more. Two failures
is a real error - retrying forever just burns GPU time on a model that cannot
produce the shape.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from rca.llm.base import LLMClient, SchemaValidationError, Usage
from rca.llm.openai_compat import extract_json

T = TypeVar("T", bound=BaseModel)

REPAIR_TEMPLATE = """Your previous reply did not satisfy the required schema.

Problem:
{problem}

Your previous reply:
{previous}

Return the corrected JSON object only. No commentary, no code fences.
Every required field must be present. Do not add fields that are not in the schema."""


def describe_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors()[:8]:
        location = ".".join(str(p) for p in error["loc"]) or "(root)"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)


def complete_structured(
    client: LLMClient,
    *,
    system: str,
    user: str,
    model_cls: type[T],
    usage: Usage | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> T:
    """Get one validated instance of `model_cls` out of the model."""
    schema = model_cls.model_json_schema()

    # Inline the schema in the prompt as well as the request. Servers without
    # guided decoding need it, and servers with it are not harmed by it.
    prompt = (
        f"{user}\n\n"
        f"Reply with a single JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    previous_text = ""
    problem = ""

    for attempt in (1, 2):
        if attempt == 2:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": previous_text},
                {
                    "role": "user",
                    "content": REPAIR_TEMPLATE.format(
                        problem=problem, previous=previous_text[:2000]
                    ),
                },
            ]

        response = client.chat(
            system=system,
            messages=messages,
            json_schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if usage is not None:
            usage.add(response.usage)

        previous_text = response.text
        try:
            payload = extract_json(response.text)
        except SchemaValidationError as exc:
            problem = str(exc)
            continue

        try:
            return model_cls.model_validate(payload)
        except ValidationError as exc:
            problem = describe_validation_error(exc)
            continue

    raise SchemaValidationError(
        f"{client.model} could not produce a valid {model_cls.__name__} in two attempts. "
        f"Last problem:\n{problem}"
    )
