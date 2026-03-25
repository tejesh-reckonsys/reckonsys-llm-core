# OpenAI strategy — uses the Responses API (client.responses.create).
# See docs/providers/openai.md for full documentation and gotchas.
import json
import logging
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI, OpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from pydantic import BaseModel

from reckonsys_llm_core._utils import parse_json_response, validate_dict_response
from reckonsys_llm_core.types import (
    Batch,
    BatchRequest,
    BatchRequestCounts,
    BatchResult,
    BatchStatus,
    ChatMessage,
    DocumentContent,
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
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096

_BATCH_STATUS_MAP: dict[str, BatchStatus] = {
    "validating": BatchStatus.IN_PROGRESS,
    "in_progress": BatchStatus.IN_PROGRESS,
    "finalizing": BatchStatus.IN_PROGRESS,
    "completed": BatchStatus.ENDED,
    "failed": BatchStatus.ENDED,
    "expired": BatchStatus.ENDED,
    "cancelling": BatchStatus.CANCELING,
    "cancelled": BatchStatus.ENDED,
}


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


def _map_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    reasoning_tokens = 0
    if getattr(usage, "output_tokens_details", None):
        reasoning_tokens = (
            getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
        )
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        reasoning_tokens=reasoning_tokens,
    )


def _content_blocks_to_input(items: list[Any]) -> list[dict[str, Any]]:
    """Convert content block objects to Responses API input content parts.

    ToolUseContent and ToolResultContent are handled at the message level.
    DocumentContent is flattened to text (no native document type in the Responses API).
    """
    parts: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, TextContent):
            parts.append({"type": "input_text", "text": item.text})
        elif isinstance(item, ImageContent):
            url = (
                item.source
                if item.is_url
                else f"data:{item.media_type};base64,{item.source}"
            )
            parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
        elif isinstance(item, DocumentContent):
            prefix = f"[Document: {item.title}]\n" if item.title else ""
            parts.append({"type": "input_text", "text": prefix + item.text})
    return parts


def _build_input(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert ChatMessage list to Responses API input items.

    Tool calls (ToolUseContent) and tool results (ToolResultContent) are lifted
    out of their parent messages and emitted as top-level input items, which is
    what the Responses API expects.
    """
    items: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg.content, str):
            items.append({"type": "message", "role": msg.role, "content": msg.content})
            continue

        tool_use_blocks = [b for b in msg.content if isinstance(b, ToolUseContent)]
        tool_result_blocks = [
            b for b in msg.content if isinstance(b, ToolResultContent)
        ]
        other_blocks = [
            b
            for b in msg.content
            if not isinstance(b, ToolUseContent | ToolResultContent)
        ]

        if tool_use_blocks:
            # Text content stays as an assistant message.
            if other_blocks:
                parts = _content_blocks_to_input(other_blocks)
                if parts:
                    items.append(
                        {"type": "message", "role": "assistant", "content": parts}
                    )
            # Each tool call becomes its own function_call item.
            for b in tool_use_blocks:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": b.id,
                        "name": b.name,
                        "arguments": json.dumps(b.input),
                    }
                )

        elif tool_result_blocks:
            # Each tool result becomes its own function_call_output item.
            for b in tool_result_blocks:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": b.tool_use_id,
                        "output": b.content,
                    }
                )
            # Any non-tool content goes as a regular user message.
            if other_blocks:
                parts = _content_blocks_to_input(other_blocks)
                if parts:
                    items.append(
                        {"type": "message", "role": msg.role, "content": parts}
                    )

        else:
            parts = _content_blocks_to_input(other_blocks)
            content: str | list[dict[str, Any]] = parts if parts else ""
            items.append({"type": "message", "role": msg.role, "content": content})

    return items


def _build_kwargs(
    params: LLMParams,
    model: str,
    default_max_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": _build_input(params.messages),
        "max_output_tokens": params.max_tokens or default_max_tokens,
    }
    if params.system:
        kwargs["instructions"] = params.system
    if params.temperature is not None:
        kwargs["temperature"] = params.temperature
    if params.top_p is not None:
        kwargs["top_p"] = params.top_p
    if params.thinking and params.thinking.enabled:
        effort = params.thinking.reasoning_effort or "medium"
        kwargs["reasoning"] = {"effort": effort}
    return kwargs


def _build_tool_params(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert ToolDefinition list to Responses API tool dicts.

    Tools with raw_config are passed through as-is — this supports OpenAI
    built-in tools like ``{"type": "web_search"}``.
    Custom tools (without raw_config) are converted to function tool format.
    """
    result: list[dict[str, Any]] = []
    for t in tools:
        if t.raw_config is not None:
            result.append(t.raw_config)
        else:
            result.append(
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                    "strict": False,
                }
            )
    return result


def _apply_tool_choice(kwargs: dict[str, Any], tool_choice: ToolChoice | None) -> None:
    if tool_choice is None:
        return
    if tool_choice.type == "tool":
        kwargs["tool_choice"] = {"type": "function", "name": tool_choice.name}
    elif tool_choice.type == "any":
        kwargs["tool_choice"] = "required"
    else:
        kwargs["tool_choice"] = tool_choice.type  # "auto" or "none"


def _stop_reason_from_response(res: Response) -> StopReason | None:
    """Derive a StopReason from the response status and output items."""
    if res.status == "incomplete":
        reason = (
            getattr(res.incomplete_details, "reason", None)
            if res.incomplete_details
            else None
        )
        if reason == "max_output_tokens":
            return StopReason.MAX_TOKENS
        return StopReason.STOP_SEQUENCE
    if res.status in ("failed", "cancelled"):
        return StopReason.ERROR
    # "completed" — check if output contains function calls
    for item in res.output:
        if isinstance(item, ResponseFunctionToolCall):
            return StopReason.TOOL_USE
    return StopReason.END_TURN


def _parse_response(res: Response, model: str) -> LLMResponse:
    text = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = []

    for item in res.output:
        if isinstance(item, ResponseOutputMessage):
            for part in item.content:
                if part.type == "output_text":
                    text = part.text
        elif isinstance(item, ResponseFunctionToolCall):
            try:
                input_args = json.loads(item.arguments)
            except json.JSONDecodeError:
                input_args = {}
            tool_calls.append(
                ToolCall(id=item.call_id, name=item.name, input=input_args)
            )
        elif isinstance(item, ResponseReasoningItem):
            parts = [
                getattr(s, "text", "")
                for s in (item.summary or [])
                if getattr(s, "type", None) == "summary_text"
            ]
            if parts:
                thinking = "\n".join(parts)

    return LLMResponse(
        content=text,
        usage=_map_usage(res.usage),
        model=model,
        stop_reason=_stop_reason_from_response(res),
        thinking=thinking,
        tool_calls=tool_calls,
    )


# --- Shared base ---


class _OpenAIBase:
    """Config and response-parsing logic shared between sync and async strategies."""

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

    @property
    def provider_name(self) -> str:
        return "openai"

    def _kwargs(self, params: LLMParams) -> dict[str, Any]:
        return _build_kwargs(params, self.model, self.default_max_tokens)

    def _json_output_kwargs(self, params: LLMStructuredParams) -> dict[str, Any]:
        """Use the Responses API native JSON schema format (single model, strict).

        Supported on: gpt-4o and later, o-series models.
        """
        kwargs = self._kwargs(params)
        schema = _strict_schema(params.response_models[0].model_json_schema())
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": params.response_models[0].__name__,
                "schema": schema,
                "strict": True,
            }
        }
        return kwargs

    def _tools_kwargs(
        self, params: LLMStructuredParams
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Use function calling for structured output (supports multiple response models)."""
        tools = [
            {
                "type": "function",
                "name": m.__name__,
                "description": f"Respond using the {m.__name__} schema.",
                "parameters": (
                    _strict_schema(m.model_json_schema())
                    if self.strict
                    else m.model_json_schema()
                ),
                "strict": self.strict,
            }
            for m in params.response_models
        ]
        kwargs = self._kwargs(params)
        if len(params.response_models) == 1:
            kwargs["tool_choice"] = {
                "type": "function",
                "name": params.response_models[0].__name__,
            }
        else:
            kwargs["tool_choice"] = "required"
        return tools, kwargs

    def _parse_response(self, res: Response) -> LLMResponse:
        return _parse_response(res, self.model)

    def _parse_json_output(
        self, res: Response, response_model: type[BaseModel]
    ) -> LLMStructuredResponse:
        text = ""
        for item in res.output:
            if isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if part.type == "output_text":
                        text = part.text
        stop_reason = _stop_reason_from_response(res)
        content, error = parse_json_response(text, response_model)
        if error:
            stop_reason = StopReason.ERROR
        return LLMStructuredResponse(
            content=content,
            raw_content=text,
            usage=_map_usage(res.usage),
            model=self.model,
            stop_reason=stop_reason,
            error=error,
        )

    def _parse_tools_output(
        self, res: Response, response_types: dict[str, type[BaseModel]]
    ) -> LLMStructuredResponse:
        stop_reason = _stop_reason_from_response(res)
        content = None
        raw_content = ""
        error = None

        for item in res.output:
            if (
                isinstance(item, ResponseFunctionToolCall)
                and item.name in response_types
            ):
                raw_content = item.arguments
                try:
                    input_dict = json.loads(item.arguments)
                except json.JSONDecodeError:
                    input_dict = {}
                content, error = validate_dict_response(
                    input_dict, response_types[item.name]
                )
                if error:
                    stop_reason = StopReason.ERROR
                break

        return LLMStructuredResponse(
            content=content,
            raw_content=raw_content,
            usage=_map_usage(res.usage),
            model=self.model,
            stop_reason=stop_reason,
            error=error,
        )


# --- Sync strategy ---


class OpenAILLMStrategy(_OpenAIBase):
    def __init__(
        self,
        client: OpenAI,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        strict: bool = False,
    ) -> None:
        super().__init__(client, model, default_max_tokens, strict)

    def __str__(self) -> str:
        return f"OpenAI({self.model})"

    def send_query(self, params: LLMParams) -> LLMResponse:
        kwargs = self._kwargs(params)
        if params.tools:
            tool_params = _build_tool_params(params.tools)
            if tool_params:
                kwargs["tools"] = tool_params
                _apply_tool_choice(kwargs, params.tool_choice)
        res = self.client.responses.create(**kwargs)
        return self._parse_response(res)

    def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        if self.strict and len(params.response_models) == 1:
            res = self.client.responses.create(**self._json_output_kwargs(params))
            return self._parse_json_output(res, params.response_models[0])

        tools, kwargs = self._tools_kwargs(params)
        res = self.client.responses.create(tools=tools, **kwargs)
        return self._parse_tools_output(
            res, {m.__name__: m for m in params.response_models}
        )

    def stream_query(self, params: LLMParams) -> Generator[StreamEvent, None, None]:
        """Yields StreamToken per token, then a final StreamDone with full metadata."""
        full_content = ""
        final_response: Response | None = None

        stream = self.client.responses.create(stream=True, **self._kwargs(params))
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                text = getattr(event, "delta", "")
                if text:
                    full_content += text
                    yield StreamToken(token=text)
            elif event_type == "response.completed":
                final_response = event.response

        if final_response is not None:
            res = final_response
        else:
            # Fallback: reconstruct minimal metadata from accumulated text
            yield StreamDone(
                full_content=full_content,
                usage=TokenUsage(),
                model=self.model,
            )
            return

        thinking: str | None = None
        for item in res.output:
            if isinstance(item, ResponseReasoningItem):
                parts = [
                    getattr(s, "text", "")
                    for s in (item.summary or [])
                    if getattr(s, "type", None) == "summary_text"
                ]
                if parts:
                    thinking = "\n".join(parts)

        yield StreamDone(
            full_content=full_content,
            usage=_map_usage(res.usage),
            model=self.model,
            stop_reason=_stop_reason_from_response(res),
            thinking=thinking,
        )


# --- Async strategy ---


class AsyncOpenAILLMStrategy(_OpenAIBase):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        strict: bool = False,
    ) -> None:
        super().__init__(client, model, default_max_tokens, strict)

    def __str__(self) -> str:
        return f"AsyncOpenAI({self.model})"

    async def send_query(self, params: LLMParams) -> LLMResponse:
        kwargs = self._kwargs(params)
        if params.tools:
            tool_params = _build_tool_params(params.tools)
            if tool_params:
                kwargs["tools"] = tool_params
                _apply_tool_choice(kwargs, params.tool_choice)
        res = await self.client.responses.create(**kwargs)
        return self._parse_response(res)

    async def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        if self.strict and len(params.response_models) == 1:
            res = await self.client.responses.create(**self._json_output_kwargs(params))
            return self._parse_json_output(res, params.response_models[0])

        tools, kwargs = self._tools_kwargs(params)
        res = await self.client.responses.create(tools=tools, **kwargs)
        return self._parse_tools_output(
            res, {m.__name__: m for m in params.response_models}
        )

    async def stream_query(
        self, params: LLMParams
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yields StreamToken per token, then a final StreamDone with full metadata."""
        full_content = ""
        final_response: Response | None = None

        stream = await self.client.responses.create(stream=True, **self._kwargs(params))
        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                text = getattr(event, "delta", "")
                if text:
                    full_content += text
                    yield StreamToken(token=text)
            elif event_type == "response.completed":
                final_response = event.response

        if final_response is not None:
            res = final_response
        else:
            yield StreamDone(
                full_content=full_content,
                usage=TokenUsage(),
                model=self.model,
            )
            return

        thinking: str | None = None
        for item in res.output:
            if isinstance(item, ResponseReasoningItem):
                parts = [
                    getattr(s, "text", "")
                    for s in (item.summary or [])
                    if getattr(s, "type", None) == "summary_text"
                ]
                if parts:
                    thinking = "\n".join(parts)

        yield StreamDone(
            full_content=full_content,
            usage=_map_usage(res.usage),
            model=self.model,
            stop_reason=_stop_reason_from_response(res),
            thinking=thinking,
        )


# --- Batch strategies ---


def _map_batch(res: Any) -> Batch:
    counts = getattr(res, "request_counts", None)
    return Batch(
        batch_id=res.id,
        status=_BATCH_STATUS_MAP.get(res.status, BatchStatus.IN_PROGRESS),
        counts=BatchRequestCounts(
            processing=getattr(counts, "total", 0) or 0,
            succeeded=getattr(counts, "completed", 0) or 0,
            errored=getattr(counts, "failed", 0) or 0,
        ),
        created_at=datetime.fromtimestamp(res.created_at),
        expires_at=datetime.fromtimestamp(res.expires_at) if res.expires_at else None,
        ended_at=datetime.fromtimestamp(res.completed_at)
        if getattr(res, "completed_at", None)
        else None,
    )


def _map_batch_result(result: dict[str, Any]) -> BatchResult:
    custom_id = result["custom_id"]

    if result.get("error"):
        error_info = result["error"]
        return BatchResult(
            custom_id=custom_id,
            response=None,
            error=error_info.get("message", str(error_info))
            if isinstance(error_info, dict)
            else str(error_info),
        )

    resp = result.get("response", {})
    if not resp or resp.get("status_code", 200) != 200:
        err = resp.get("body", {}).get("error", {})
        return BatchResult(
            custom_id=custom_id,
            response=None,
            error=err.get("message", "non-200 status")
            if isinstance(err, dict)
            else str(err),
        )

    body = resp.get("body", {})
    output = body.get("output", [])

    text = ""
    tool_calls: list[ToolCall] = []
    has_tool_calls = False

    for item in output:
        item_type = item.get("type", "")
        if item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text = part.get("text", "")
        elif item_type == "function_call":
            has_tool_calls = True
            try:
                input_args = json.loads(item.get("arguments", "{}"))
            except json.JSONDecodeError:
                input_args = {}
            tool_calls.append(
                ToolCall(
                    id=item.get("call_id", ""),
                    name=item.get("name", ""),
                    input=input_args,
                )
            )

    usage = body.get("usage", {})
    output_details = usage.get("output_tokens_details") or {}
    reasoning_tokens = (
        output_details.get("reasoning_tokens", 0)
        if isinstance(output_details, dict)
        else 0
    )

    status = body.get("status", "completed")
    if status == "incomplete":
        incomplete = body.get("incomplete_details") or {}
        reason = incomplete.get("reason", "")
        stop_reason = (
            StopReason.MAX_TOKENS
            if reason == "max_output_tokens"
            else StopReason.STOP_SEQUENCE
        )
    elif has_tool_calls:
        stop_reason = StopReason.TOOL_USE
    elif status in ("failed", "cancelled"):
        stop_reason = StopReason.ERROR
    else:
        stop_reason = StopReason.END_TURN

    return BatchResult(
        custom_id=custom_id,
        response=LLMResponse(
            content=text,
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                reasoning_tokens=reasoning_tokens,
            ),
            model=body.get("model", ""),
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        ),
    )


class OpenAIBatchStrategy:
    """
    Synchronous OpenAI batch API — 50% cheaper, higher rate limits.

    Submits requests via a JSONL file upload against the ``/v1/responses``
    endpoint. The caller is responsible for:
    - Persisting the batch_id returned by create_batch()
    - Polling via get_batch() until status is ENDED
    - Retrieving results via get_results()
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.default_max_tokens = default_max_tokens

    @property
    def provider_name(self) -> str:
        return "openai"

    def __str__(self) -> str:
        return f"OpenAIBatch({self.model})"

    def create_batch(self, requests: list[BatchRequest]) -> Batch:
        lines = [
            json.dumps(
                {
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": _build_kwargs(
                        req.params, self.model, self.default_max_tokens
                    ),
                }
            )
            for req in requests
        ]
        jsonl_bytes = "\n".join(lines).encode()
        file_obj = self.client.files.create(
            file=("batch.jsonl", jsonl_bytes, "application/jsonl"),
            purpose="batch",
        )
        res = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        return _map_batch(res)

    def get_batch(self, batch_id: str) -> Batch:
        return _map_batch(self.client.batches.retrieve(batch_id))

    def cancel_batch(self, batch_id: str) -> Batch:
        return _map_batch(self.client.batches.cancel(batch_id))

    def get_results(self, batch_id: str) -> Generator[BatchResult, None, None]:
        """Stream results one by one — avoids loading all results into memory."""
        batch = self.client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            return
        content = self.client.files.content(batch.output_file_id)
        for line in content.text.splitlines():
            if line.strip():
                yield _map_batch_result(json.loads(line))


class AsyncOpenAIBatchStrategy:
    """
    Asynchronous OpenAI batch API — 50% cheaper, higher rate limits.

    Submits requests via a JSONL file upload against the ``/v1/responses``
    endpoint. The caller is responsible for:
    - Persisting the batch_id returned by create_batch()
    - Polling via get_batch() until status is ENDED
    - Retrieving results via get_results()
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.default_max_tokens = default_max_tokens

    @property
    def provider_name(self) -> str:
        return "openai"

    def __str__(self) -> str:
        return f"AsyncOpenAIBatch({self.model})"

    async def create_batch(self, requests: list[BatchRequest]) -> Batch:
        lines = [
            json.dumps(
                {
                    "custom_id": req.custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": _build_kwargs(
                        req.params, self.model, self.default_max_tokens
                    ),
                }
            )
            for req in requests
        ]
        jsonl_bytes = "\n".join(lines).encode()
        file_obj = await self.client.files.create(
            file=("batch.jsonl", jsonl_bytes, "application/jsonl"),
            purpose="batch",
        )
        res = await self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        return _map_batch(res)

    async def get_batch(self, batch_id: str) -> Batch:
        return _map_batch(await self.client.batches.retrieve(batch_id))

    async def cancel_batch(self, batch_id: str) -> Batch:
        return _map_batch(await self.client.batches.cancel(batch_id))

    async def get_results(self, batch_id: str) -> AsyncGenerator[BatchResult, None]:
        """Stream results one by one — avoids loading all results into memory."""
        batch = await self.client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            return
        content = await self.client.files.content(batch.output_file_id)
        for line in content.text.splitlines():
            if line.strip():
                yield _map_batch_result(json.loads(line))


# --- Convenience constants ---

OPENAI_WEB_SEARCH_TOOL = ToolDefinition(
    name="web_search",
    raw_config={"type": "web_search"},
)
"""
OpenAI's built-in web search tool for the Responses API. Pass in LLMParams.tools
to give the model live web access.

    response = client.query(
        messages=[ChatMessage(role="user", content="Latest Python release?")],
        tools=[OPENAI_WEB_SEARCH_TOOL],
    )
"""


OPENAI_CODE_INTERPRETER_TOOL = ToolDefinition(
    name="code_interpreter",
    raw_config={"type": "code_interpreter", "container": {"type": "auto"}},
)
"""OpenAI's built-in code interpreter tool. Runs Python in a sandboxed container."""

OPENAI_FILE_SEARCH_TOOL = ToolDefinition(
    name="file_search",
    raw_config={"type": "file_search"},
)
"""OpenAI's built-in file search tool. Searches uploaded files via vector store."""

OPENAI_TOOL_SEARCH_TOOL = ToolDefinition(
    name="tool_search",
    raw_config={"type": "tool_search"},
)
"""OpenAI's built-in tool search tool (gpt-5.4+). Dynamic discovery from large tool catalogs."""


# --- Factory helpers ---


def create_openai_client(**kwargs: Any) -> OpenAI:
    return OpenAI(max_retries=1, **kwargs)


def create_async_openai_client(**kwargs: Any) -> AsyncOpenAI:
    return AsyncOpenAI(max_retries=1, **kwargs)
