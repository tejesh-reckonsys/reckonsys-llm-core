from reckonsys_llm_core.strategy import LLMStrategy
from reckonsys_llm_core.types import (
    ChatMessage,
    LLMParams,
    LLMResponse,
    LLMStructuredParams,
    LLMStructuredResponse,
    ThinkingConfig,
)
from pydantic import BaseModel


class LLMClient:
    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy

    def query(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> LLMResponse:
        params = LLMParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
        )
        return self.strategy.send_query(params)

    def query_structured(
        self,
        messages: list[ChatMessage],
        response_models: list[type[BaseModel]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> LLMStructuredResponse:
        params = LLMStructuredParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
            response_models=response_models,
        )
        return self.strategy.send_structured_query(params)
