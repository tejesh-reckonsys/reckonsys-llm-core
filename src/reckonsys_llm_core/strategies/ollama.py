import json
import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any

from ollama import AsyncClient, ChatResponse, Client, Options
from pydantic import BaseModel

from reckonsys_llm_core._utils import parse_json_response, validate_dict_response
from reckonsys_llm_core.types import (
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
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)

logger = logging.getLogger(__name__)

DONE_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
}

DEFAULT_MAX_TOKENS = 8000


# --- Module-level helpers ---


def _build_options(params: LLMParams, default_max_tokens: int) -> Options:
    opts: dict[str, Any] = {"num_predict": params.max_tokens or default_max_tokens}
    if params.temperature is not None:
        opts["temperature"] = params.temperature
    if params.top_p is not None:
        opts["top_p"] = params.top_p
    if params.stop:
        opts["stop"] = params.stop
    return Options(**opts)


def _message_to_ollama_messages(m: ChatMessage) -> list[dict[str, Any]]:
    """
    Convert a ChatMessage to one or more Ollama message dicts.

    Ollama differences from Anthropic:
    - Tool results use role "tool" (one message per result), not "user" + tool_result blocks.
    - Assistant tool calls use a "tool_calls" key, not content blocks.
    - Images go in a separate "images" list (base64 only; URL images are skipped).
    """
    if isinstance(m.content, str):
        return [{"role": m.role, "content": m.content}]

    # User message carrying tool results → emit one "tool" message per result.
    if m.role == "user" and all(
        isinstance(item, ToolResultContent) for item in m.content
    ):
        return [
            {"role": "tool", "content": item.content}
            for item in m.content
            if isinstance(item, ToolResultContent)
        ]

    # Assistant message that may contain text + tool_use blocks.
    if m.role == "assistant":
        text_parts = [item.text for item in m.content if isinstance(item, TextContent)]
        tool_uses = [item for item in m.content if isinstance(item, ToolUseContent)]
        msg: dict[str, Any] = {"role": "assistant", "content": " ".join(text_parts)}
        if tool_uses:
            msg["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.input}}
                for tc in tool_uses
            ]
        return [msg]

    # Regular user message with text / images.
    texts: list[str] = []
    images: list[str] = []
    for item in m.content:
        if isinstance(item, TextContent):
            texts.append(item.text)
        elif isinstance(item, DocumentContent):
            logger.warning(
                "Ollama does not support DocumentContent — skipping. "
                "Pass document text as a plain string or TextContent instead."
            )
        elif isinstance(item, ImageContent):
            if item.is_url:
                logger.warning(
                    "Ollama does not support URL images — skipping. "
                    "Convert to base64 before sending."
                )
            else:
                images.append(item.source)

    result: dict[str, Any] = {"role": m.role, "content": "\n".join(texts)}
    if images:
        result["images"] = images
    return [result]


def _build_messages(params: LLMParams) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if params.system:
        messages.append({"role": "system", "content": params.system})
    for m in params.messages:
        messages.extend(_message_to_ollama_messages(m))
    return messages


def _resolve_think(params: LLMParams) -> bool | None:
    """Return True/False when ThinkingConfig is set, None to use model default."""
    if params.thinking is None:
        return None
    return params.thinking.enabled


def _map_usage(response: ChatResponse) -> TokenUsage:
    return TokenUsage(
        input_tokens=getattr(response, "prompt_eval_count", 0) or 0,
        output_tokens=getattr(response, "eval_count", 0) or 0,
    )


def _build_tools(response_models: list[type[BaseModel]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": m.__name__,
                "description": f"Respond using the {m.__name__} schema.",
                "parameters": m.model_json_schema(),
            },
        }
        for m in response_models
    ]


def _build_tool_defs(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert ToolDefinition list to Ollama's tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
        if t.raw_config is None  # built-in tools are not supported by Ollama
    ]


# --- Shared base for sync and async Ollama strategies ---


class _OllamaBase:
    """Holds all config and response-parsing logic shared between sync and async."""

    def __init__(self, model: str, host: str, default_max_tokens: int) -> None:
        self.model = model
        self.host = host
        self.default_max_tokens = default_max_tokens

    def _parse_response(self, res: ChatResponse) -> LLMResponse:
        done_reason = getattr(res, "done_reason", None)
        tool_calls = [
            ToolCall(
                id=f"call_{i}",  # Ollama provides no IDs
                name=tc.function.name,
                input=dict(tc.function.arguments),
            )
            for i, tc in enumerate(res.message.tool_calls or [])
        ]
        return LLMResponse(
            content=res.message.content or "",
            usage=_map_usage(res),
            model=self.model,
            stop_reason=DONE_REASON_MAP.get(done_reason) if done_reason else None,
            thinking=res.message.thinking,
            tool_calls=tool_calls,
        )

    def _parse_format_output(
        self, res: ChatResponse, response_model: type[BaseModel]
    ) -> LLMStructuredResponse:
        """Parse a format-constrained response (single model, JSON schema generation)."""
        raw_content = res.message.content or ""
        done_reason = getattr(res, "done_reason", None)
        stop_reason = DONE_REASON_MAP.get(done_reason) if done_reason else None
        content, error = parse_json_response(raw_content, response_model)
        if error:
            stop_reason = StopReason.ERROR
        return LLMStructuredResponse(
            content=content,
            raw_content=raw_content,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=stop_reason,
            thinking=res.message.thinking,
            error=error,
        )

    def _parse_tools_output(
        self, res: ChatResponse, response_types: dict[str, type[BaseModel]]
    ) -> LLMStructuredResponse:
        """Parse a tool-use response (multiple models)."""
        done_reason = getattr(res, "done_reason", None)
        stop_reason = DONE_REASON_MAP.get(done_reason) if done_reason else None
        content = None
        raw_content = ""
        error = None

        for tool_call in res.message.tool_calls or []:
            name = tool_call.function.name
            if name in response_types:
                raw_content = json.dumps(tool_call.function.arguments)
                content, error = validate_dict_response(
                    tool_call.function.arguments, response_types[name]
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
            thinking=res.message.thinking,
            error=error,
        )


# --- Sync strategy ---


class OllamaLLMStrategy(_OllamaBase):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(model, host, default_max_tokens)
        self.client = Client(host=host)

    def __str__(self) -> str:
        return f"Ollama({self.model})"

    def send_query(self, params: LLMParams) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if params.tools:
            kwargs["tools"] = _build_tool_defs(params.tools)
        res = self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            think=_resolve_think(params),
            **kwargs,
        )
        return self._parse_response(res)

    def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        if len(params.response_models) == 1:
            res = self.client.chat(
                model=self.model,
                messages=_build_messages(params),
                options=_build_options(params, self.default_max_tokens),
                format=params.response_models[0].model_json_schema(),
                think=_resolve_think(params),
            )
            return self._parse_format_output(res, params.response_models[0])

        # Multiple models → tool-use
        # Requires a tool-capable model (qwen3, qwen2.5, llama3.2, mistral-nemo,
        # command-r, granite3, phi4). Models without tool training return content=None.
        res = self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            tools=_build_tools(params.response_models),
            think=_resolve_think(params),
        )
        return self._parse_tools_output(
            res, {m.__name__: m for m in params.response_models}
        )

    def stream_query(self, params: LLMParams) -> Generator[StreamEvent, None, None]:
        """Yields StreamToken per token, then a final StreamDone with full metadata."""
        full_content = ""
        thinking_text = ""
        last_chunk = None

        for chunk in self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            stream=True,
            think=_resolve_think(params),
        ):
            token = chunk.message.content or ""
            if token:
                full_content += token
                yield StreamToken(token=token)

            if chunk.message.thinking:
                thinking_text += chunk.message.thinking

            last_chunk = chunk

        done_reason = getattr(last_chunk, "done_reason", None) if last_chunk else None
        yield StreamDone(
            full_content=full_content,
            usage=_map_usage(last_chunk) if last_chunk else TokenUsage(),
            model=self.model,
            stop_reason=DONE_REASON_MAP.get(done_reason) if done_reason else None,
            thinking=thinking_text or None,
        )


# --- Async strategy ---


class AsyncOllamaLLMStrategy(_OllamaBase):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(model, host, default_max_tokens)
        self.client = AsyncClient(host=host)

    def __str__(self) -> str:
        return f"AsyncOllama({self.model})"

    async def send_query(self, params: LLMParams) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if params.tools:
            kwargs["tools"] = _build_tool_defs(params.tools)
        res = await self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            think=_resolve_think(params),
            **kwargs,
        )
        return self._parse_response(res)

    async def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        if len(params.response_models) == 1:
            res = await self.client.chat(
                model=self.model,
                messages=_build_messages(params),
                options=_build_options(params, self.default_max_tokens),
                format=params.response_models[0].model_json_schema(),
                think=_resolve_think(params),
            )
            return self._parse_format_output(res, params.response_models[0])

        res = await self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            tools=_build_tools(params.response_models),
            think=_resolve_think(params),
        )
        return self._parse_tools_output(
            res, {m.__name__: m for m in params.response_models}
        )

    async def stream_query(
        self, params: LLMParams
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yields StreamToken per token, then a final StreamDone with full metadata."""
        full_content = ""
        thinking_text = ""
        last_chunk = None

        async for chunk in await self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            stream=True,
            think=_resolve_think(params),
        ):
            token = chunk.message.content or ""
            if token:
                full_content += token
                yield StreamToken(token=token)

            # Thinking content streams separately from response tokens
            if chunk.message.thinking:
                thinking_text += chunk.message.thinking

            last_chunk = chunk

        done_reason = getattr(last_chunk, "done_reason", None) if last_chunk else None
        yield StreamDone(
            full_content=full_content,
            usage=_map_usage(last_chunk) if last_chunk else TokenUsage(),
            model=self.model,
            stop_reason=DONE_REASON_MAP.get(done_reason) if done_reason else None,
            thinking=thinking_text or None,
        )
