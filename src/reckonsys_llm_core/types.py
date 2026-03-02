from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

ResponseType = TypeVar("ResponseType", bound=BaseModel)


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    ERROR = "error"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ThinkingConfig:
    enabled: bool = False
    budget_tokens: int = 1024
    reasoning_effort: str | None = None  # "low" | "medium" | "high" for OpenAI


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class LLMParams:
    messages: list[ChatMessage]
    system: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    thinking: ThinkingConfig | None = None


@dataclass
class LLMResponse:
    content: str
    usage: TokenUsage
    model: str
    stop_reason: StopReason | None = None
    thinking: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMStructuredParams(LLMParams):
    response_models: list[type[BaseModel]] = field(default_factory=list)


@dataclass
class LLMStructuredResponse:
    content: BaseModel | None
    usage: TokenUsage
    model: str
    stop_reason: StopReason | None = None
    thinking: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
