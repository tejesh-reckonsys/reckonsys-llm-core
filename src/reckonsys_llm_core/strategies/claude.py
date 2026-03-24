import json
import logging
import os
from typing import Any, AsyncGenerator, Generator, cast

from pydantic import BaseModel

from anthropic import Anthropic, AnthropicBedrock, AsyncAnthropic, AsyncAnthropicBedrock
from anthropic.types import (
    ContentBlockParam,
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ToolParam,
    ToolUseBlock,
    StopReason as AnthropicStopReason,
)
from anthropic.types.messages.batch_create_params import Request as BatchCreateRequest

from reckonsys_llm_core._utils import parse_json_response, validate_dict_response
from reckonsys_llm_core.types import (
    Batch,
    BatchRequest,
    BatchRequestCounts,
    BatchResult,
    BatchStatus,
    ChatContent,
    ChatMessage,
    ImageContent,
    LLMParams,
    LLMResponse,
    LLMStructuredParams,
    LLMStructuredResponse,
    StopReason,
    StreamDone,
    StreamEvent,
    StreamToken,
    TextContent,
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


# --- Module-level helpers ---


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


def _content_to_api(content: ChatContent) -> str | list[ContentBlockParam]:
    """Convert ChatContent to Anthropic API content format.

    Images can be base64 or URL. Both are supported by Claude.
    """
    if isinstance(content, str):
        return content

    blocks: list[ContentBlockParam] = []
    for item in content:
        if isinstance(item, TextContent):
            blocks.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            if item.is_url:
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": item.source},
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.media_type,
                            "data": item.source,
                        },
                    }
                )
    return blocks


def _build_message_params(messages: list[ChatMessage]) -> list[MessageParam]:
    return [
        MessageParam(role=m.role, content=_content_to_api(m.content)) for m in messages
    ]


def _build_kwargs(
    params: LLMParams,
    model: str,
    default_max_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _build_message_params(params.messages),
        "max_tokens": params.max_tokens or default_max_tokens,
    }
    if params.system:
        kwargs["system"] = [
            TextBlockParam(
                text=params.system, type="text", cache_control={"type": "ephemeral"}
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


def _extract_text_and_thinking(message: Message) -> tuple[str, str | None]:
    text = ""
    thinking_text = None
    for block in message.content:
        if isinstance(block, TextBlock):
            text = block.text
        elif isinstance(block, ThinkingBlock):
            thinking_text = block.thinking
    return text, thinking_text


def _build_tools(
    response_models: list[type[BaseModel]], strict: bool
) -> list[ToolParam]:
    # strict=True applies additionalProperties:false to the schema
    # (no "strict" key — that's OpenAI-specific and has no effect on Anthropic)
    return [
        ToolParam(
            name=m.__name__,
            description=f"Respond using the {m.__name__} schema.",
            input_schema=_strict_schema(m.model_json_schema())
            if strict
            else m.model_json_schema(),
        )
        for m in response_models
    ]


# --- Shared base for sync and async Claude strategies ---


class _ClaudeBase:
    """Holds all config and response-parsing logic shared between sync and async."""

    def __init__(
        self,
        client: Any,
        model: str,
        default_max_tokens: int,
        strict: bool,
    ) -> None:
        self.client = client
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.strict = strict

    def _kwargs(self, params: LLMParams) -> dict[str, Any]:
        return _build_kwargs(params, self.model, self.default_max_tokens)

    def _json_output_kwargs(self, params: LLMStructuredParams) -> dict[str, Any]:
        """Build kwargs for the output_config JSON schema approach (single model, strict).

        Supported on: claude-opus-4-6, claude-sonnet-4-6, claude-sonnet-4-5,
        claude-opus-4-5, claude-haiku-4-5.
        """
        kwargs = self._kwargs(params)
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": _strict_schema(params.response_models[0].model_json_schema()),
            }
        }
        return kwargs

    def _tools_kwargs(
        self, params: LLMStructuredParams
    ) -> tuple[list[ToolParam], dict[str, Any]]:
        """Build tools list and kwargs for the tool-use approach."""
        tools = _build_tools(params.response_models, self.strict)
        kwargs = self._kwargs(params)
        if params.thinking and params.thinking.enabled:
            kwargs["temperature"] = 1  # required for thinking + tools
        else:
            kwargs["tool_choice"] = {"type": "any"}
        return tools, kwargs

    def _parse_response(self, res: Message) -> LLMResponse:
        text, thinking = _extract_text_and_thinking(res)
        return LLMResponse(
            content=text,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=STOP_REASON_MAP.get(res.stop_reason)
            if res.stop_reason
            else None,
            thinking=thinking,
        )

    def _parse_json_output(
        self, res: Message, response_model: type[BaseModel]
    ) -> LLMStructuredResponse:
        text, thinking = _extract_text_and_thinking(res)
        stop_reason = STOP_REASON_MAP.get(res.stop_reason) if res.stop_reason else None
        content, error = parse_json_response(text, response_model)
        if error:
            stop_reason = StopReason.ERROR
        return LLMStructuredResponse(
            content=content,
            raw_content=text,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=stop_reason,
            thinking=thinking,
            error=error,
        )

    def _parse_tools_output(
        self, res: Message, response_types: dict[str, type[BaseModel]]
    ) -> LLMStructuredResponse:
        _, thinking = _extract_text_and_thinking(res)
        stop_reason = STOP_REASON_MAP.get(res.stop_reason) if res.stop_reason else None
        content = None
        raw_content = ""
        error = None

        for block in res.content:
            if isinstance(block, ToolUseBlock) and block.name in response_types:
                raw_content = json.dumps(block.input)
                content, error = validate_dict_response(
                    block.input, response_types[block.name]
                )
                if error:
                    stop_reason = StopReason.ERROR
                break

        return LLMStructuredResponse(
            content=content,
            raw_content=raw_content,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=stop_reason,
            thinking=thinking,
            error=error,
        )


# --- Sync strategy ---


class ClaudeLLMStrategy(_ClaudeBase):
    def __init__(
        self,
        client: Anthropic | AnthropicBedrock,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        strict: bool = False,
    ) -> None:
        super().__init__(client, model, default_max_tokens, strict)

    def __str__(self) -> str:
        return f"Claude({self.model})"

    def send_query(self, params: LLMParams) -> LLMResponse:
        res = cast(Message, self.client.messages.create(**self._kwargs(params)))
        return self._parse_response(res)

    def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        if self.strict and len(params.response_models) == 1:
            res = cast(
                Message, self.client.messages.create(**self._json_output_kwargs(params))
            )
            return self._parse_json_output(res, params.response_models[0])

        tools, kwargs = self._tools_kwargs(params)
        res = cast(Message, self.client.messages.create(tools=tools, **kwargs))
        return self._parse_tools_output(
            res, {m.__name__: m for m in params.response_models}
        )


# --- Async strategy ---


class AsyncClaudeLLMStrategy(_ClaudeBase):
    def __init__(
        self,
        client: AsyncAnthropic | AsyncAnthropicBedrock,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        strict: bool = False,
    ) -> None:
        super().__init__(client, model, default_max_tokens, strict)

    def __str__(self) -> str:
        return f"AsyncClaude({self.model})"

    async def send_query(self, params: LLMParams) -> LLMResponse:
        res = cast(Message, await self.client.messages.create(**self._kwargs(params)))
        return self._parse_response(res)

    async def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        if self.strict and len(params.response_models) == 1:
            res = cast(
                Message,
                await self.client.messages.create(**self._json_output_kwargs(params)),
            )
            return self._parse_json_output(res, params.response_models[0])

        tools, kwargs = self._tools_kwargs(params)
        res = cast(Message, await self.client.messages.create(tools=tools, **kwargs))
        return self._parse_tools_output(
            res, {m.__name__: m for m in params.response_models}
        )

    async def stream_query(
        self, params: LLMParams
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yields StreamToken per token, then a final StreamDone with full metadata."""
        full_content = ""
        thinking_text = None

        async with self.client.messages.stream(**self._kwargs(params)) as stream:
            async for text in stream.text_stream:
                full_content += text
                yield StreamToken(token=text)

            final = await stream.get_final_message()
            for block in final.content:
                if isinstance(block, ThinkingBlock):
                    thinking_text = block.thinking

        yield StreamDone(
            full_content=full_content,
            usage=_map_usage(final),
            model=self.model,
            stop_reason=STOP_REASON_MAP.get(final.stop_reason)
            if final.stop_reason
            else None,
            thinking=thinking_text,
        )


# --- Batch strategies ---

_BATCH_STATUS_MAP: dict[str, BatchStatus] = {
    "in_progress": BatchStatus.IN_PROGRESS,
    "canceling": BatchStatus.CANCELING,
    "ended": BatchStatus.ENDED,
}


class ClaudeBatchStrategy:
    """
    Synchronous Claude batch API — for scripts, CLI tools, and eval pipelines.

    The underlying batch processing is always async on Anthropic's side.
    This is just the Python calling convention (no async/await required).
    """

    def __init__(
        self,
        client: Anthropic | AnthropicBedrock,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.default_max_tokens = default_max_tokens

    def __str__(self) -> str:
        return f"ClaudeBatch({self.model})"

    def create_batch(self, requests: list[BatchRequest]) -> Batch:
        batch_requests: list[BatchCreateRequest] = [
            BatchCreateRequest(
                custom_id=req.custom_id,
                params=_build_kwargs(req.params, self.model, self.default_max_tokens),  # type: ignore[arg-type]
            )
            for req in requests
        ]
        res = self.client.messages.batches.create(requests=batch_requests)
        return _map_batch(res)

    def get_batch(self, batch_id: str) -> Batch:
        res = self.client.messages.batches.retrieve(batch_id)
        return _map_batch(res)

    def cancel_batch(self, batch_id: str) -> Batch:
        res = self.client.messages.batches.cancel(batch_id)
        return _map_batch(res)

    def get_results(self, batch_id: str) -> Generator[BatchResult, None, None]:
        """Stream results one by one — avoids loading all results into memory."""
        for result in self.client.messages.batches.results(batch_id):
            yield _map_batch_result(result)


def _map_batch(res: Any) -> Batch:
    c = res.request_counts
    return Batch(
        batch_id=res.id,
        status=_BATCH_STATUS_MAP.get(res.processing_status, BatchStatus.IN_PROGRESS),
        counts=BatchRequestCounts(
            processing=c.processing,
            succeeded=c.succeeded,
            errored=c.errored,
            canceled=c.canceled,
            expired=c.expired,
        ),
        created_at=res.created_at,
        expires_at=res.expires_at,
        ended_at=res.ended_at,
    )


def _map_batch_result(res: Any) -> BatchResult:
    if res.result.type == "succeeded":
        msg = res.result.message
        text, thinking = _extract_text_and_thinking(msg)
        return BatchResult(
            custom_id=res.custom_id,
            response=LLMResponse(
                content=text,
                usage=_map_usage(msg),
                model=msg.model,
                stop_reason=STOP_REASON_MAP.get(msg.stop_reason)
                if msg.stop_reason
                else None,
                thinking=thinking,
            ),
        )
    error_detail = getattr(res.result, "error", None)
    return BatchResult(
        custom_id=res.custom_id,
        response=None,
        error=str(error_detail) if error_detail else res.result.type,
    )


class AsyncClaudeBatchStrategy:
    """
    Claude's native batch API — up to 50% cheaper, higher rate limits.

    Processes up to 100k requests per batch asynchronously.
    Batches expire after 24 hours if not completed.

    The caller is responsible for:
    - Persisting the batch_id returned by create_batch()
    - Deciding when and how often to poll via get_batch()
    - Retrieving and acting on results via get_results()
    """

    def __init__(
        self,
        client: AsyncAnthropic | AsyncAnthropicBedrock,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.default_max_tokens = default_max_tokens

    def __str__(self) -> str:
        return f"AsyncClaudeBatch({self.model})"

    async def create_batch(self, requests: list[BatchRequest]) -> Batch:
        batch_requests: list[BatchCreateRequest] = [
            BatchCreateRequest(
                custom_id=req.custom_id,
                params=_build_kwargs(req.params, self.model, self.default_max_tokens),  # type: ignore[arg-type]
            )
            for req in requests
        ]
        res = await self.client.messages.batches.create(requests=batch_requests)
        return _map_batch(res)

    async def get_batch(self, batch_id: str) -> Batch:
        res = await self.client.messages.batches.retrieve(batch_id)
        return _map_batch(res)

    async def cancel_batch(self, batch_id: str) -> Batch:
        res = await self.client.messages.batches.cancel(batch_id)
        return _map_batch(res)

    async def get_results(self, batch_id: str) -> AsyncGenerator[BatchResult, None]:
        """Stream results one by one — avoids loading all results into memory."""
        async for result in await self.client.messages.batches.results(batch_id):
            yield _map_batch_result(result)


# --- Factory helpers ---


def create_claude_client(**kwargs: Any) -> Anthropic:
    return Anthropic(max_retries=1, **kwargs)


def create_async_claude_client(**kwargs: Any) -> AsyncAnthropic:
    return AsyncAnthropic(max_retries=1, **kwargs)


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


def create_async_bedrock_client(
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-west-2",
) -> AsyncAnthropicBedrock:
    return AsyncAnthropicBedrock(
        aws_access_key=access_key or os.getenv("AWS_IAM_ACCESS_KEY"),
        aws_secret_key=secret_key or os.getenv("AWS_IAM_SECRET_KEY"),
        aws_region=region,
        max_retries=1,
    )
