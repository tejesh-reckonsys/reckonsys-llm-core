import json
import logging
import os
from typing import Any, cast

import pydantic
from anthropic import Anthropic, AnthropicBedrock
from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ToolParam,
    ToolUseBlock,
    StopReason as AnthropicStopReason,
)

from reckonsys_llm_core.types import (
    LLMParams,
    LLMResponse,
    LLMStructuredParams,
    LLMStructuredResponse,
    StopReason,
    TokenUsage,
)

logger = logging.getLogger(__name__)

STOP_REASON_MAP: dict[AnthropicStopReason, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
}

DEFAULT_MAX_TOKENS = 8000


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively ensure all object types have additionalProperties: false."""
    schema = dict(schema)
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        if "properties" in schema:
            schema["properties"] = {
                k: _strict_schema(v) for k, v in schema["properties"].items()
            }
    if "items" in schema:
        schema["items"] = _strict_schema(schema["items"])
    if "$defs" in schema:
        schema["$defs"] = {k: _strict_schema(v) for k, v in schema["$defs"].items()}
    return schema


def _map_usage(message: Message) -> TokenUsage:
    usage = message.usage
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


class ClaudeLLMStrategy:
    def __init__(
        self,
        client: Anthropic | AnthropicBedrock,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        strict: bool = False,
    ):
        self.client = client
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.strict = strict

    def __str__(self):
        return f"Claude({self.model})"

    def _build_kwargs(self, params: LLMParams) -> dict[str, Any]:
        messages = [
            MessageParam(role=m.role, content=m.content) for m in params.messages
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": params.max_tokens or self.default_max_tokens,
        }

        if params.system:
            kwargs["system"] = [
                TextBlockParam(
                    text=params.system,
                    type="text",
                    cache_control={"type": "ephemeral"},
                )
            ]

        if params.temperature is not None:
            kwargs["temperature"] = params.temperature

        if params.top_p is not None:
            kwargs["top_p"] = params.top_p

        if params.stop:
            kwargs["stop_sequences"] = params.stop

        if params.thinking and params.thinking.enabled:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": params.thinking.budget_tokens,
            }

        return kwargs

    def send_query(self, params: LLMParams) -> LLMResponse:
        res = cast(Message, self.client.messages.create(**self._build_kwargs(params)))

        # Extract text and thinking content
        text = ""
        thinking_text = None
        for block in res.content:
            if isinstance(block, TextBlock):
                text = block.text
            elif isinstance(block, ThinkingBlock):
                thinking_text = block.thinking

        return LLMResponse(
            content=text,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=(
                STOP_REASON_MAP.get(res.stop_reason) if res.stop_reason else None
            ),
            thinking=thinking_text,
        )

    def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        if self.strict and len(params.response_models) == 1:
            return self._send_structured_query_json_output(params)
        return self._send_structured_query_tools(params)

    def _send_structured_query_json_output(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        """Use output_config JSON schema approach (single model, strict mode).

        Supported on: claude-opus-4-6, claude-sonnet-4-6, claude-sonnet-4-5,
        claude-opus-4-5, claude-haiku-4-5.
        """
        response_model = params.response_models[0]
        kwargs = self._build_kwargs(params)
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": _strict_schema(response_model.model_json_schema()),
            }
        }

        res = cast(Message, self.client.messages.create(**kwargs))

        content = None
        thinking_text = None
        stop_reason = STOP_REASON_MAP.get(res.stop_reason) if res.stop_reason else None

        for block in res.content:
            if isinstance(block, ThinkingBlock):
                thinking_text = block.thinking
            elif isinstance(block, TextBlock):
                try:
                    content = response_model.model_validate(json.loads(block.text))
                except (pydantic.ValidationError, json.JSONDecodeError) as e:
                    logger.warning("Structured response validation failed: %s", e)
                    stop_reason = StopReason.ERROR

        return LLMStructuredResponse(
            content=content,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=stop_reason,
            thinking=thinking_text,
        )

    def _send_structured_query_tools(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        """Use tool-use approach (multiple models, or strict=False)."""
        response_types: dict[str, type] = {
            m.__name__: m for m in params.response_models
        }

        if self.strict:
            tools: list[Any] = [
                {
                    "name": name,
                    "description": f"Respond using the {name} schema.",
                    "input_schema": model.model_json_schema(),
                    "strict": True,
                }
                for name, model in response_types.items()
            ]
        else:
            tools = [
                ToolParam(
                    name=name,
                    description=f"Respond using the {name} schema.",
                    input_schema=model.model_json_schema(),
                )
                for name, model in response_types.items()
            ]

        kwargs = self._build_kwargs(params)

        if params.thinking and params.thinking.enabled:
            kwargs["temperature"] = 1  # Required for thinking + tools
        else:
            kwargs["tool_choice"] = {"type": "any"}

        res = cast(Message, self.client.messages.create(tools=tools, **kwargs))

        content = None
        thinking_text = None
        stop_reason = STOP_REASON_MAP.get(res.stop_reason) if res.stop_reason else None

        for block in res.content:
            if isinstance(block, ThinkingBlock):
                thinking_text = block.thinking
            elif isinstance(block, ToolUseBlock) and block.name in response_types:
                try:
                    content = response_types[block.name].model_validate(block.input)
                except pydantic.ValidationError as e:
                    logger.warning("Structured response validation failed: %s", e)
                    stop_reason = StopReason.ERROR

        return LLMStructuredResponse(
            content=content,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=stop_reason,
            thinking=thinking_text,
        )


def create_claude_client(**kwargs) -> Anthropic:
    return Anthropic(max_retries=1, **kwargs)


def create_bedrock_client(
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-west-2",
) -> AnthropicBedrock:
    return AnthropicBedrock(
        aws_access_key=access_key or os.getenv("AWS_IAM_ACCESS_KEY"),
        aws_secret_key=secret_key or os.getenv("AWS_IAM_SECRET_KEY"),
        aws_region=region,
        max_retries=1,
    )
