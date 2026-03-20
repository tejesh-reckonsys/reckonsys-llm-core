from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel


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
    reasoning_effort: Literal["low", "medium", "high"] | None = None  # for OpenAI


# --- Content types ---

@dataclass
class TextContent:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ImageContent:
    """
    Represents an image for multimodal input.

    For base64: set source to the raw base64 string, is_url=False.
    For URLs:   set source to the URL string, is_url=True.

    Note: Ollama does not support URL images — only base64 is portable across all providers.
    """
    source: str        # base64 data or URL string
    media_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
    is_url: bool = False
    type: Literal["image"] = "image"


# A message's content is either a plain string or a list of text/image blocks.
ChatContent = str | list[TextContent | ImageContent]


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: ChatContent


# --- Request params ---

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
class LLMStructuredParams(LLMParams):
    response_models: list[type[BaseModel]] = field(default_factory=list)


# --- Responses ---

@dataclass
class LLMResponse:
    content: str
    usage: TokenUsage
    model: str
    stop_reason: StopReason | None = None
    thinking: str | None = None
    attempts: int = 1
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMStructuredResponse:
    content: BaseModel | None
    raw_content: str          # raw string before parsing — used by client for retry correction
    usage: TokenUsage
    model: str
    stop_reason: StopReason | None = None
    thinking: str | None = None
    attempts: int = 1
    error: str | None = None  # validation error message when stop_reason == ERROR
    provider_metadata: dict[str, Any] = field(default_factory=dict)


# --- Streaming ---

@dataclass
class StreamToken:
    """Yielded once per token during streaming."""
    token: str
    is_done: Literal[False] = False


@dataclass
class StreamDone:
    """Final event yielded after all tokens — carries full metadata."""
    full_content: str
    usage: TokenUsage
    model: str
    stop_reason: StopReason | None = None
    thinking: str | None = None
    is_done: Literal[True] = True
    provider_metadata: dict[str, Any] = field(default_factory=dict)


StreamEvent = StreamToken | StreamDone


# --- Batch processing ---

class BatchStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    CANCELING   = "canceling"
    ENDED       = "ended"


@dataclass
class BatchRequestCounts:
    processing: int = 0
    succeeded:  int = 0
    errored:    int = 0
    canceled:   int = 0
    expired:    int = 0

    @property
    def total(self) -> int:
        return self.processing + self.succeeded + self.errored + self.canceled + self.expired


@dataclass
class Batch:
    batch_id:   str
    status:     BatchStatus
    counts:     BatchRequestCounts
    created_at: datetime
    expires_at: datetime | None = None
    ended_at:   datetime | None = None


@dataclass
class BatchRequest:
    """
    A single request inside a batch submission.

    custom_id is your correlation key — used to match results back to requests.
    It must be unique within a batch and can be any string meaningful to your app
    (e.g. a DB row ID, a query hash, a user ID).

    Note: structured output (tools / output_config) is not supported in batch params
    here. If you need structured output from batch results, validate LLMResponse.content
    yourself after retrieving results.
    """
    custom_id: str
    params: LLMParams


@dataclass
class BatchResult:
    """Result for a single request in a completed batch."""
    custom_id: str
    response:  LLMResponse | None  # None when error/canceled/expired
    error:     str | None = None   # set when response is None


# --- Retry observability ---

@dataclass
class RetryContext:
    """Passed to the on_retry callback so callers can log/trace retry attempts."""
    attempt: int                      # which attempt just failed (1-based)
    error: str                        # validation error message
    raw_content: str                  # what the LLM actually returned
    params: LLMStructuredParams       # the params used for this attempt
