import json
import logging
from typing import Any, AsyncGenerator

from ollama import AsyncClient, ChatResponse, Client, Options
from pydantic import BaseModel

from reckonsys_llm_core._utils import parse_json_response, validate_dict_response
from reckonsys_llm_core.types import (
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


def _message_to_ollama(m: ChatMessage) -> dict[str, Any]:
    """Convert a ChatMessage to Ollama's message dict format.

    Ollama handles images via a separate `images` list (base64 strings only).
    URL images are not supported — a warning is logged and the image is skipped.
    """
    if isinstance(m.content, str):
        return {"role": m.role, "content": m.content}

    texts: list[str] = []
    images: list[str] = []
    for item in m.content:
        if isinstance(item, TextContent):
            texts.append(item.text)
        elif isinstance(item, ImageContent):
            if item.is_url:
                logger.warning(
                    "Ollama does not support URL images — skipping. "
                    "Convert to base64 before sending."
                )
            else:
                images.append(item.source)

    msg: dict[str, Any] = {"role": m.role, "content": "\n".join(texts)}
    if images:
        msg["images"] = images
    return msg


def _build_messages(params: LLMParams) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if params.system:
        messages.append({"role": "system", "content": params.system})
    messages.extend(_message_to_ollama(m) for m in params.messages)
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


# --- Shared base for sync and async Ollama strategies ---

class _OllamaBase:
    """Holds all config and response-parsing logic shared between sync and async."""

    def __init__(self, model: str, host: str, default_max_tokens: int) -> None:
        self.model = model
        self.host = host
        self.default_max_tokens = default_max_tokens

    def _options(self, params: LLMParams) -> Options:
        return _build_options(params, self.default_max_tokens)

    def _messages(self, params: LLMParams) -> list[dict[str, Any]]:
        return _build_messages(params)

    def _think(self, params: LLMParams) -> bool | None:
        return _resolve_think(params)

    def _parse_response(self, res: ChatResponse) -> LLMResponse:
        done_reason = getattr(res, "done_reason", None)
        return LLMResponse(
            content=res.message.content or "",
            usage=_map_usage(res),
            model=self.model,
            stop_reason=DONE_REASON_MAP.get(done_reason) if done_reason else None,
            thinking=res.message.thinking,
        )

    def _parse_format_output(self, res: ChatResponse, response_model: type[BaseModel]) -> LLMStructuredResponse:
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

    def _parse_tools_output(self, res: ChatResponse, response_types: dict[str, type[BaseModel]]) -> LLMStructuredResponse:
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
        res = self.client.chat(
            model=self.model,
            messages=self._messages(params),
            options=self._options(params),
            think=self._think(params),
        )
        return self._parse_response(res)

    def send_structured_query(self, params: LLMStructuredParams) -> LLMStructuredResponse:
        if len(params.response_models) == 1:
            res = self.client.chat(
                model=self.model,
                messages=self._messages(params),
                options=self._options(params),
                format=params.response_models[0].model_json_schema(),
                think=self._think(params),
            )
            return self._parse_format_output(res, params.response_models[0])

        # Multiple models → tool-use
        # Requires a tool-capable model (qwen3, qwen2.5, llama3.2, mistral-nemo,
        # command-r, granite3, phi4). Models without tool training return content=None.
        res = self.client.chat(
            model=self.model,
            messages=self._messages(params),
            options=self._options(params),
            tools=_build_tools(params.response_models),
            think=self._think(params),
        )
        return self._parse_tools_output(res, {m.__name__: m for m in params.response_models})


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
        res = await self.client.chat(
            model=self.model,
            messages=self._messages(params),
            options=self._options(params),
            think=self._think(params),
        )
        return self._parse_response(res)

    async def send_structured_query(self, params: LLMStructuredParams) -> LLMStructuredResponse:
        if len(params.response_models) == 1:
            res = await self.client.chat(
                model=self.model,
                messages=self._messages(params),
                options=self._options(params),
                format=params.response_models[0].model_json_schema(),
                think=self._think(params),
            )
            return self._parse_format_output(res, params.response_models[0])

        res = await self.client.chat(
            model=self.model,
            messages=self._messages(params),
            options=self._options(params),
            tools=_build_tools(params.response_models),
            think=self._think(params),
        )
        return self._parse_tools_output(res, {m.__name__: m for m in params.response_models})

    async def stream_query(self, params: LLMParams) -> AsyncGenerator[StreamEvent, None]:
        """Yields StreamToken per token, then a final StreamDone with full metadata."""
        full_content = ""
        thinking_text = ""
        last_chunk = None

        async for chunk in await self.client.chat(
            model=self.model,
            messages=self._messages(params),
            options=self._options(params),
            stream=True,
            think=self._think(params),
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
