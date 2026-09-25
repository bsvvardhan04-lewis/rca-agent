"""The provider layer: schema generation, JSON recovery, and schema repair.

These are the parts that absorb open-weight-model sloppiness, so they are worth
testing hard. A hosted frontier model rarely wraps JSON in prose or drops a
required field; a 7B served by Ollama does both regularly.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from rca.llm.base import LLMError, ToolSpec
from rca.llm.openai_compat import extract_json
from rca.llm.schema import build_schema, split_docstring, tool
from rca.llm.structured import complete_structured
from rca.llm.base import SchemaValidationError
from tests.fakes import AlwaysBadLLM, BadJSONLLM, MissingFieldLLM


# -- docstring parsing ----------------------------------------------------


def test_summary_and_args_are_separated():
    doc = """Do a thing.

    More detail about the thing.

    Args:
        alpha: The first one.
        beta: The second one.

    Returns:
        Something.
    """
    summary, params = split_docstring(doc)
    assert "Do a thing." in summary
    assert "More detail" in summary
    assert "Returns" not in summary
    assert params == {"alpha": "The first one.", "beta": "The second one."}


def test_wrapped_argument_descriptions_are_joined():
    doc = """Thing.

    Args:
        alpha: A description that runs on
            to a second line.
    """
    _, params = split_docstring(doc)
    assert params["alpha"] == "A description that runs on to a second line."


# -- schema generation ----------------------------------------------------


def test_types_defaults_and_required_are_derived_from_the_signature():
    def sample(name: str, count: int = 5, ratio: float = 1.0, flag: bool = False) -> str:
        """Do a thing.

        Args:
            name: Who to do it to.
            count: How many times.
            ratio: A fraction.
            flag: Whether to shout.
        """
        return ""

    description, schema = build_schema(sample)
    assert description == "Do a thing."
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name"], "only the parameter without a default"
    assert schema["properties"]["count"] == {
        "type": "integer",
        "description": "How many times.",
        "default": 5,
    }
    assert schema["properties"]["ratio"]["type"] == "number"
    assert schema["properties"]["flag"]["type"] == "boolean"


def test_an_undocumented_parameter_is_rejected():
    def sample(alpha: str = "", beta: str = "") -> str:
        """Thing.

        Args:
            alpha: Documented.
        """
        return ""

    with pytest.raises(ValueError, match="beta has no description"):
        build_schema(sample)


def test_a_tool_without_a_docstring_is_rejected():
    def sample(alpha: str = "") -> str:
        return ""

    with pytest.raises(ValueError, match="needs a docstring"):
        build_schema(sample)


def test_an_unsupported_annotation_is_rejected():
    def sample(when: complex = 0j) -> str:
        """Thing.

        Args:
            when: A complex number.
        """
        return ""

    with pytest.raises(TypeError, match="unsupported annotation"):
        build_schema(sample)


def test_the_decorator_emits_the_openai_function_shape():
    @tool
    def sample(alpha: str = "") -> str:
        """Do a thing.

        Args:
            alpha: The input.
        """
        return f"got {alpha}"

    assert isinstance(sample, ToolSpec)
    payload = sample.to_openai()
    assert payload["type"] == "function"
    assert payload["function"]["name"] == "sample"
    assert payload["function"]["parameters"]["properties"]["alpha"]["description"] == "The input."
    json.dumps(payload)  # must survive the wire


# -- tolerating model sloppiness -----------------------------------------


@tool
def echo(alpha: str = "", count: int = 1) -> str:
    """Echo a value.

    Args:
        alpha: What to echo.
        count: How many times.
    """
    return f"{alpha}x{count}"


def test_arguments_sent_as_a_json_string_are_accepted():
    """Open-weight models often send `arguments` as a string, not an object."""
    assert echo.call('{"alpha": "hi", "count": 2}') == "hix2"


def test_unknown_argument_keys_are_dropped_rather_than_raising():
    assert echo.call({"alpha": "hi", "hallucinated": True}) == "hix1"


def test_empty_arguments_fall_back_to_defaults():
    assert echo.call("") == "x1"
    assert echo.call({}) == "x1"


def test_argument_strings_that_are_not_json_are_a_clear_error():
    with pytest.raises(LLMError, match="not valid JSON"):
        echo.call("alpha=hi")


# -- JSON recovery --------------------------------------------------------


def test_plain_json_is_parsed():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_is_unwrapped():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_json_buried_in_commentary_is_recovered():
    text = 'Sure! Here is the analysis you asked for:\n\n{"a": 1, "b": "two"}\n\nHope that helps.'
    assert extract_json(text) == {"a": 1, "b": "two"}


def test_nested_objects_are_not_truncated_at_the_first_brace():
    text = 'Result: {"outer": {"inner": [1, 2]}, "done": true} - end'
    assert extract_json(text) == {"outer": {"inner": [1, 2]}, "done": True}


def test_braces_inside_strings_do_not_confuse_the_scanner():
    text = 'Here: {"message": "a } brace in a string", "ok": true}'
    assert extract_json(text) == {"message": "a } brace in a string", "ok": True}


def test_an_empty_or_prose_only_reply_is_an_error():
    with pytest.raises(SchemaValidationError, match="empty response"):
        extract_json("   ")
    with pytest.raises(SchemaValidationError, match="No JSON object"):
        extract_json("I am afraid I cannot help with that.")


# -- structured output with repair ---------------------------------------


class Tiny(BaseModel):
    summary: str
    count: int


def test_a_malformed_first_reply_is_repaired_on_the_second_attempt():
    good = Tiny(summary="ok", count=3)
    client = BadJSONLLM(good=good)
    result = complete_structured(client, system="s", user="u", model_cls=Tiny)
    assert result == good
    assert client.attempts == 2, "one repair round, not a retry storm"


def test_a_missing_required_field_triggers_a_repair():
    good = Tiny(summary="ok", count=3)
    client = MissingFieldLLM(good=good)
    assert complete_structured(client, system="s", user="u", model_cls=Tiny) == good
    assert client.attempts == 2


def test_two_failures_raise_rather_than_looping_forever():
    client = AlwaysBadLLM(good=Tiny(summary="x", count=1))
    with pytest.raises(SchemaValidationError, match="two attempts"):
        complete_structured(client, system="s", user="u", model_cls=Tiny)
    assert client.attempts == 2, "must not keep burning GPU time"


def test_usage_is_accumulated_across_repair_rounds():
    from rca.llm.base import Usage

    usage = Usage()
    complete_structured(
        BadJSONLLM(good=Tiny(summary="ok", count=1)),
        system="s",
        user="u",
        model_cls=Tiny,
        usage=usage,
    )
    assert usage.prompt_tokens == 200, "both attempts counted"
