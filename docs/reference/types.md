# Types & Data Models

All public types are exported from the top-level `reckonsys_llm_core` package.

---

## Request types

### `LLMParams`

Parameters for a plain text query.

| Field | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[ChatMessage]` | required | Conversation history |
| `system` | `str \| None` | `None` | System prompt |
| `cache_system` | `bool` | `True` | Cache the system prompt (Claude only) |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `max_tokens` | `int \| None` | `None` | Max output tokens (falls back to strategy default) |
| `top_p` | `float \| None` | `None` | Nucleus sampling |
| `stop` | `list[str] \| None` | `None` | Stop sequences (Claude / Ollama only) |
| `thinking` | `ThinkingConfig \| None` | `None` | Enable extended thinking / reasoning |
| `tools` | `list[ToolDefinition]` | `[]` | Tools available to the model |
| `tool_choice` | `ToolChoice \| None` | `None` | How the model selects tools |

### `LLMStructuredParams`

Extends `LLMParams` with:

| Field | Type | Default | Description |
|---|---|---|---|
| `response_models` | `list[type[BaseModel]]` | `[]` | Pydantic models the LLM should fill |

### `ThinkingConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Enable thinking / reasoning |
| `budget_tokens` | `int` | `1024` | Token budget (Claude only) |
| `reasoning_effort` | `"low" \| "medium" \| "high" \| None` | `None` | Effort level (OpenAI o-series only) |

---

## Message types

### `ChatMessage`

| Field | Type | Default | Description |
|---|---|---|---|
| `role` | `"user" \| "assistant"` | required | Speaker |
| `content` | `ChatContent` | required | Message body |
| `cache` | `bool` | `False` | Insert cache breakpoint after this message (Claude only) |

### `ChatContent`

A `ChatMessage`'s content is either a plain `str` or a list of content blocks:

```
str
| list[TextContent | ImageContent | DocumentContent | ToolUseContent | ToolResultContent]
```

### `TextContent`

| Field | Type |
|---|---|
| `text` | `str` |

### `ImageContent`

| Field | Type | Description |
|---|---|---|
| `source` | `str` | Base64 data or URL string |
| `media_type` | `"image/png" \| "image/jpeg" \| "image/gif" \| "image/webp"` | |
| `is_url` | `bool` | `True` if `source` is a URL |

> Ollama does not support URL images — use base64 for cross-provider compatibility.

### `DocumentContent`

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | required | Document body |
| `title` | `str \| None` | `None` | Optional title |
| `citations_enabled` | `bool` | `True` | Allow Claude to cite passages (Claude only) |

### `ToolUseContent`

Represents a tool call in an **assistant** message. Append this to the conversation after receiving a `tool_use` response.

| Field | Type |
|---|---|
| `id` | `str` |
| `name` | `str` |
| `input` | `dict[str, Any]` |

### `ToolResultContent`

Represents tool output in a **user** message. Append after executing a tool.

| Field | Type | Default | Description |
|---|---|---|---|
| `tool_use_id` | `str` | required | Matches `ToolUseContent.id` |
| `content` | `str` | required | Tool output |
| `is_error` | `bool` | `False` | Mark as error result |

---

## Tool types

### `ToolDefinition`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Tool name |
| `description` | `str` | `""` | What the tool does |
| `input_schema` | `dict[str, Any]` | `{}` | JSON Schema for the tool's input |
| `raw_config` | `dict[str, Any] \| None` | `None` | Pass-through for provider built-in tools |

When `raw_config` is set the other fields are ignored and the dict is forwarded directly to the API.

### `ToolCall`

Returned inside `LLMResponse.tool_calls` when the model calls a tool.

| Field | Type |
|---|---|
| `id` | `str` |
| `name` | `str` |
| `input` | `dict[str, Any]` |

### `ToolChoice`

| Field | Type | Description |
|---|---|---|
| `type` | `"auto" \| "any" \| "tool" \| "none"` | Selection mode |
| `name` | `str \| None` | Required when `type="tool"` |

---

## Response types

### `LLMResponse`

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Model output text |
| `usage` | `TokenUsage` | Token counts |
| `model` | `str` | Model ID used |
| `stop_reason` | `StopReason \| None` | Why generation stopped |
| `thinking` | `str \| None` | Extended thinking trace |
| `attempts` | `int` | Number of attempts (async client with retry) |
| `tool_calls` | `list[ToolCall]` | Tool calls made by the model |
| `citations` | `list[Citation]` | Source citations (Claude only) |

### `LLMStructuredResponse`

| Field | Type | Description |
|---|---|---|
| `content` | `BaseModel \| None` | Parsed Pydantic instance, or `None` on error |
| `raw_content` | `str` | Raw string before parsing |
| `usage` | `TokenUsage` | Token counts |
| `model` | `str` | Model ID |
| `stop_reason` | `StopReason \| None` | `StopReason.ERROR` on parse failure |
| `thinking` | `str \| None` | Thinking trace |
| `attempts` | `int` | Number of attempts |
| `error` | `str \| None` | Validation error message when `content` is `None` |

### `TokenUsage`

| Field | Type | Description |
|---|---|---|
| `input_tokens` | `int` | Tokens in the prompt |
| `output_tokens` | `int` | Tokens in the response |
| `cache_read_tokens` | `int` | Tokens read from cache (Claude only) |
| `cache_creation_tokens` | `int` | Tokens written to cache (Claude only) |
| `reasoning_tokens` | `int` | Tokens spent on reasoning (OpenAI o-series only) |
| `total_tokens` | `int` | `input_tokens + output_tokens` (property) |

### `StopReason`

| Value | Meaning |
|---|---|
| `end_turn` | Model finished normally |
| `tool_use` | Model is waiting for tool results |
| `max_tokens` | Output token limit reached |
| `stop_sequence` | A stop sequence was matched (or content filtered) |
| `error` | Parsing / validation error (structured output) |

### `Citation`

| Field | Type | Description |
|---|---|---|
| `cited_text` | `str` | The exact passage cited |
| `url` | `str \| None` | Web search citation URL |
| `title` | `str \| None` | Web search source title |
| `document_title` | `str \| None` | `DocumentContent` citation |
| `document_index` | `int \| None` | Which document in the message (0-indexed) |

---

## Streaming types

### `StreamToken`

| Field | Type |
|---|---|
| `token` | `str` |
| `is_done` | `Literal[False]` |

### `StreamDone`

| Field | Type |
|---|---|
| `full_content` | `str` |
| `usage` | `TokenUsage` |
| `model` | `str` |
| `stop_reason` | `StopReason \| None` |
| `thinking` | `str \| None` |
| `citations` | `list[Citation]` |
| `is_done` | `Literal[True]` |

```python
StreamEvent = StreamToken | StreamDone
```

---

## Batch types

### `BatchRequest`

| Field | Type | Description |
|---|---|---|
| `custom_id` | `str` | Your correlation key — must be unique within a batch |
| `params` | `LLMParams` | Request parameters |

### `BatchResult`

| Field | Type | Description |
|---|---|---|
| `custom_id` | `str` | Echoes back the `custom_id` from `BatchRequest` |
| `response` | `LLMResponse \| None` | Set on success |
| `error` | `str \| None` | Set on failure, cancellation, or expiry |

### `Batch`

| Field | Type | Description |
|---|---|---|
| `batch_id` | `str` | Provider-assigned batch identifier |
| `status` | `BatchStatus` | Current status |
| `counts` | `BatchRequestCounts` | Request counts by state |
| `created_at` | `datetime` | |
| `expires_at` | `datetime \| None` | |
| `ended_at` | `datetime \| None` | |

### `BatchStatus`

| Value | Meaning |
|---|---|
| `in_progress` | Processing |
| `canceling` | Cancellation requested |
| `ended` | Completed, failed, cancelled, or expired |

### `BatchRequestCounts`

| Field | Type |
|---|---|
| `processing` | `int` |
| `succeeded` | `int` |
| `errored` | `int` |
| `canceled` | `int` |
| `expired` | `int` |
| `total` | `int` (property) |

---

## Retry observability

### `RetryContext`

Passed to the `on_retry` callback of `AsyncLLMClient`.

| Field | Type | Description |
|---|---|---|
| `attempt` | `int` | Which attempt just failed (1-based) |
| `error` | `str` | Validation error message |
| `raw_content` | `str` | What the LLM returned |
| `params` | `LLMStructuredParams` | Params used for this attempt |
