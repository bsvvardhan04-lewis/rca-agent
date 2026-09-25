"""Build a JSON schema for a tool from its Python signature and docstring.

This replaces a vendor decorator, so the tool layer depends on nothing but the
standard library. The contract is the same one every SDK uses: types come from
annotations, descriptions come from a Google-style `Args:` block, and a
parameter is required exactly when it has no default.

Undocumented parameters are a hard error. A model given a parameter with no
description will guess, and a wrong guess here means a wrong query against
production logs.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any

from rca.llm.base import ToolSpec

JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

ARGS_HEADING = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
ARG_LINE = re.compile(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")
SECTION_HEADING = re.compile(
    r"^\s*(Returns|Raises|Yields|Examples?|Notes?)\s*:\s*$", re.IGNORECASE
)


def split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    params: dict[str, str] = {}

    in_args = False
    current: str | None = None

    for line in lines:
        if ARGS_HEADING.match(line):
            in_args = True
            current = None
            continue
        if SECTION_HEADING.match(line):
            in_args = False
            current = None
            continue

        if not in_args:
            summary.append(line)
            continue

        match = ARG_LINE.match(line)
        if match:
            current = match.group(1)
            params[current] = match.group(2).strip()
        elif current and line.strip():
            # A wrapped continuation of the previous parameter's description.
            params[current] = f"{params[current]} {line.strip()}"

    return "\n".join(summary).strip(), params


def build_schema(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    """Return (description, json_schema) for a tool function."""
    summary, param_docs = split_docstring(fn.__doc__ or "")
    if not summary:
        raise ValueError(f"{fn.__name__} needs a docstring - the model reads it.")

    signature = inspect.signature(fn)
    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:  # noqa: BLE001 - fall back to raw annotations
        hints = getattr(fn, "__annotations__", {})

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = hints.get(name, str)
        json_type = JSON_TYPES.get(annotation)
        if json_type is None:
            raise TypeError(
                f"{fn.__name__}.{name}: unsupported annotation {annotation!r}. "
                f"Tool parameters must be one of {sorted(t.__name__ for t in JSON_TYPES)}."
            )

        description = param_docs.get(name)
        if not description:
            raise ValueError(
                f"{fn.__name__}.{name} has no description in the docstring Args block. "
                "An undocumented parameter is a parameter the model will guess at."
            )

        prop: dict[str, Any] = {"type": json_type, "description": description}
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)
        if json_type == "array":
            prop["items"] = {"type": "string"}

        properties[name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required

    return summary, schema


def tool(fn: Callable[..., str]) -> ToolSpec:
    """Decorator turning a documented function into a ToolSpec."""
    description, schema = build_schema(fn)
    return ToolSpec(
        name=fn.__name__,
        description=description,
        input_schema=schema,
        fn=fn,
    )
