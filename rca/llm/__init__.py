"""LLM provider wiring.

`build_client` reads config and hands back something satisfying `LLMClient`.
Everything self-hosted speaks the OpenAI chat format, so one implementation
covers vLLM, Ollama, TGI, LM Studio and llama.cpp.
"""

from __future__ import annotations

from rca.config import Settings, settings as default_settings
from rca.llm.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    SchemaValidationError,
    ToolCall,
    ToolSpec,
    Usage,
)
from rca.llm.openai_compat import OpenAICompatClient
from rca.llm.schema import build_schema, tool
from rca.llm.structured import complete_structured

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "SchemaValidationError",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "OpenAICompatClient",
    "build_client",
    "build_schema",
    "complete_structured",
    "tool",
]


def build_client(settings: Settings | None = None) -> LLMClient:
    s = settings or default_settings
    return OpenAICompatClient(
        model=s.model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        timeout=s.llm_timeout,
        supports_json_schema=s.llm_json_schema,
    )
