import json
import logging
from typing import Any

import pydantic
from ollama import Client, Options

from reckonsys_llm_core.types import (
    LLMParams,
    LLMResponse,
    LLMStructuredParams,
    LLMStructuredResponse,
    StopReason,
    TokenUsage,
)

logger = logging.getLogger(__name__)

DONE_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
}

DEFAULT_MAX_TOKENS = 8000


def _build_options(params: LLMParams, default_max_tokens: int) -> Options:
    opts: dict[str, Any] = {
        "num_predict": params.max_tokens or default_max_tokens,
    }
    if params.temperature is not None:
        opts["temperature"] = params.temperature
    if params.top_p is not None:
        opts["top_p"] = params.top_p
    if params.stop:
        opts["stop"] = params.stop
    return Options(**opts)


def _build_messages(params: LLMParams) -> list[dict[str, Any]]:
    messages = []
    if params.system:
        messages.append({"role": "system", "content": params.system})
    messages.extend({"role": m.role, "content": m.content} for m in params.messages)
    return messages


def _resolve_think(params: LLMParams) -> bool | None:
    """Return True/False when ThinkingConfig is set, None to use model default."""
    if params.thinking is None:
        return None
    return params.thinking.enabled


def _map_usage(response: Any) -> TokenUsage:
    return TokenUsage(
        input_tokens=getattr(response, "prompt_eval_count", 0) or 0,
        output_tokens=getattr(response, "eval_count", 0) or 0,
    )


class OllamaLLMStrategy:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.client = Client(host=host)

    def __str__(self):
        return f"Ollama({self.model})"

    def send_query(self, params: LLMParams) -> LLMResponse:
        think = _resolve_think(params)

        res = self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            think=think,
        )

        done_reason = getattr(res, "done_reason", None)
        return LLMResponse(
            content=res.message.content or "",
            usage=_map_usage(res),
            model=self.model,
            stop_reason=DONE_REASON_MAP.get(done_reason) if done_reason else None,
            thinking=res.message.thinking,
        )

    def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        if len(params.response_models) == 1:
            return self._send_structured_query_format(params)
        return self._send_structured_query_tools(params)

    def _send_structured_query_format(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        """Use the format parameter with a JSON schema (single model)."""
        response_model = params.response_models[0]
        think = _resolve_think(params)

        res = self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            format=response_model.model_json_schema(),
            think=think,
        )

        content = None
        done_reason = getattr(res, "done_reason", None)
        stop_reason = DONE_REASON_MAP.get(done_reason) if done_reason else None

        try:
            content = response_model.model_validate(
                json.loads(res.message.content or "{}")
            )
        except (pydantic.ValidationError, json.JSONDecodeError) as e:
            logger.warning("Structured response validation failed: %s", e)
            stop_reason = StopReason.ERROR

        return LLMStructuredResponse(
            content=content,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=stop_reason,
            thinking=res.message.thinking,
        )

    def _send_structured_query_tools(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse:
        """Use tool-use for multiple response models.

        Requires a tool-capable model (e.g. llama3.2, qwen2.5, qwen3, mistral-nemo,
        command-r, granite3, phi4). Models without tool training will return
        content=None. Use a single response_model to avoid tool-use entirely.
        """
        response_types: dict[str, type] = {
            m.__name__: m for m in params.response_models
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Respond using the {name} schema.",
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model in response_types.items()
        ]

        think = _resolve_think(params)

        res = self.client.chat(
            model=self.model,
            messages=_build_messages(params),
            options=_build_options(params, self.default_max_tokens),
            tools=tools,
            think=think,
        )

        content = None
        done_reason = getattr(res, "done_reason", None)
        stop_reason = DONE_REASON_MAP.get(done_reason) if done_reason else None

        for tool_call in res.message.tool_calls or []:
            name = tool_call.function.name
            if name in response_types:
                try:
                    content = response_types[name].model_validate(
                        tool_call.function.arguments
                    )
                except pydantic.ValidationError as e:
                    logger.warning("Structured response validation failed: %s", e)
                    stop_reason = StopReason.ERROR
                break

        return LLMStructuredResponse(
            content=content,
            usage=_map_usage(res),
            model=self.model,
            stop_reason=stop_reason,
            thinking=res.message.thinking,
        )
