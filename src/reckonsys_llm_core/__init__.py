from importlib.resources import files

from reckonsys_llm_core.client import (
    AsyncBatchLLMClient,
    AsyncLLMClient,
    BatchLLMClient,
    LLMClient,
)
from reckonsys_llm_core.fn_tools import ToolKit, from_tools, tool_from_function
from reckonsys_llm_core.strategy import (
    AsyncBatchLLMStrategy,
    AsyncLLMStrategy,
    BatchLLMStrategy,
    LLMStrategy,
)
from reckonsys_llm_core.types import (
    Batch,
    BatchRequest,
    BatchRequestCounts,
    BatchResult,
    BatchStatus,
    ChatContent,
    ChatMessage,
    Citation,
    DocumentContent,
    ImageContent,
    LLMParams,
    LLMResponse,
    LLMStructuredParams,
    LLMStructuredResponse,
    RetryContext,
    StopReason,
    StreamDone,
    StreamEvent,
    StreamToken,
    TextContent,
    ThinkingConfig,
    TokenUsage,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)


def get_llms_txt() -> str:
    """Return the llms.txt content for use as LLM context.

    Example:
        >>> import reckonsys_llm_core
        >>> print(reckonsys_llm_core.get_llms_txt())
    """
    return files("reckonsys_llm_core").joinpath("llms.txt").read_text(encoding="utf-8")


__all__ = [
    # Clients
    "LLMClient",
    "AsyncLLMClient",
    "BatchLLMClient",
    "AsyncBatchLLMClient",
    # Protocols
    "LLMStrategy",
    "AsyncLLMStrategy",
    "BatchLLMStrategy",
    "AsyncBatchLLMStrategy",
    # Message / content types
    "ChatMessage",
    "ChatContent",
    "TextContent",
    "ImageContent",
    "DocumentContent",
    # Params
    "LLMParams",
    "LLMStructuredParams",
    "ThinkingConfig",
    # Responses
    "LLMResponse",
    "LLMStructuredResponse",
    "TokenUsage",
    "StopReason",
    # Streaming
    "StreamToken",
    "StreamDone",
    "StreamEvent",
    # Batch
    "Batch",
    "BatchRequest",
    "BatchRequestCounts",
    "BatchResult",
    "BatchStatus",
    # Retry observability
    "RetryContext",
    # Tools
    "ToolDefinition",
    "ToolCall",
    "ToolChoice",
    "ToolUseContent",
    "ToolResultContent",
    "Citation",
    # Function → tool helpers
    "tool_from_function",
    "from_tools",
    "ToolKit",
    # LLM context
    "get_llms_txt",
]
