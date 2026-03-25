# Claude (Anthropic + AWS Bedrock)

> Last reviewed: March 2026

## Installation

```bash
pip install "reckonsys-llm-core[claude]"   # anthropic>=0.84.0
```

Set `ANTHROPIC_API_KEY` in your environment (or pass it explicitly via `create_claude_client(api_key=...)`).

---

## Current models

| Model | API ID | Context | Max output | Best for |
|---|---|---|---|---|
| Claude Opus 4.6 | `claude-opus-4-6` | **1M tokens** | 128k | Most capable, complex reasoning |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | **1M tokens** | 64k | Balanced speed / quality |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 200k | 64k | Fast, cost-efficient tasks |

> The 1M context window for Opus 4.6 and Sonnet 4.6 is **GA** as of March 2026 — no beta header required. Up to 600 images/PDFs per request (up from 100).

### Deprecated / retiring soon

| Model | Retires |
|---|---|
| Claude Haiku 3 (`claude-3-haiku-20240307`) | **April 19, 2026** |
| Claude Sonnet 3.7, Haiku 3.5 | Retired Feb 19, 2026 |

---

## Strategies

| Class | Client type | Use case |
|---|---|---|
| `ClaudeLLMStrategy` | `Anthropic` / `AnthropicBedrock` | Sync |
| `AsyncClaudeLLMStrategy` | `AsyncAnthropic` / `AsyncAnthropicBedrock` | Async |
| `ClaudeBatchStrategy` | `Anthropic` / `AnthropicBedrock` | Sync batch |
| `AsyncClaudeBatchStrategy` | `AsyncAnthropic` / `AsyncAnthropicBedrock` | Async batch |

---

## Quick start

```python
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

strategy = ClaudeLLMStrategy(
    client=create_claude_client(),   # reads ANTHROPIC_API_KEY
    model="claude-sonnet-4-6",
)
client = LLMClient(strategy)

response = client.query(
    messages=[ChatMessage(role="user", content="Hello!")],
    system="You are a helpful assistant.",
)
print(response.content)
```

---

## Async

```python
from reckonsys_llm_core import AsyncLLMClient, ChatMessage
from reckonsys_llm_core.strategies.claude import AsyncClaudeLLMStrategy, create_async_claude_client

strategy = AsyncClaudeLLMStrategy(
    client=create_async_claude_client(),
    model="claude-sonnet-4-6",
)
client = AsyncLLMClient(strategy)
```

---

## AWS Bedrock

```python
from reckonsys_llm_core.strategies.claude import (
    ClaudeLLMStrategy,
    create_bedrock_client,
    create_async_bedrock_client,
)

# Reads AWS_IAM_ACCESS_KEY and AWS_IAM_SECRET_KEY from env by default
strategy = ClaudeLLMStrategy(
    client=create_bedrock_client(region="us-west-2"),
    model="anthropic.claude-opus-4-6-20251101-v1:0",
)
```

---

## Structured output

Structured outputs are **GA** as of January 29, 2026 — no beta header required.

### Tool-use approach (default, all models)

Works with all Claude models. Supports multiple response models in one call.

```python
from pydantic import BaseModel
from reckonsys_llm_core import LLMClient, ChatMessage

class Person(BaseModel):
    name: str
    age: int

response = client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Alice is 30.")],
    response_models=[Person],
)
print(response.content)  # Person(name='Alice', age=30)
```

### Native JSON schema — strict mode

Pass `strict=True` to use Claude's `output_config.format` with `json_schema`. Guarantees valid JSON without tool-use overhead.

- **Supported on:** `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5`
- With a single `response_model` → uses `output_config` (no tools)
- With multiple `response_models` → falls back to tool-use with `additionalProperties: false`

```python
strategy = ClaudeLLMStrategy(
    client=create_claude_client(),
    model="claude-opus-4-6",
    strict=True,
)
```

> **Note (Feb 5, 2026 change):** The internal parameter was renamed from `output_format` to `output_config.format`. The library already uses `output_config` — no action required.

---

## Extended thinking

### `effort` parameter (recommended for Opus 4.6+)

Claude Opus 4.6 and Sonnet 4.6 use **adaptive thinking** by default. The `effort` parameter provides a higher-level control over thinking depth and is the recommended approach for new models. `budget_tokens` still works but is considered legacy.

> ⚠️ **The library currently passes `budget_tokens` in the API call.** For Opus 4.6 / Sonnet 4.6, switch to the `effort` parameter when the library is updated. Workaround: pass `reasoning_effort` via `ThinkingConfig` — the `effort` mapping is not yet implemented.

```python
from reckonsys_llm_core import ThinkingConfig

# Current approach (works but legacy on new models)
response = client.query(
    messages=[ChatMessage(role="user", content="Solve step by step...")],
    thinking=ThinkingConfig(enabled=True, budget_tokens=5000),
)
print(response.thinking)
print(response.content)
```

### Interleaved thinking (beta)

Think between tool calls in multi-step workflows. Requires beta header `interleaved-thinking-2025-05-14`. Not yet exposed in this library — pass extra headers via the raw Anthropic client if needed.

---

## Prompt caching

### Manual cache breakpoints

Add `cache=True` to any `ChatMessage`, or `cache_system=True` on `LLMParams` (default).

```python
ChatMessage(role="user", content="Long document...", cache=True)
```

### Automatic caching (new, Feb 2026)

A single `cache_control` on the request body auto-advances the cache point as the conversation grows. **Not yet exposed in the library** — the library uses manual breakpoints only.

Cache TTL of 1 hour is GA — no beta header required.

> ⚠️ **Breaking change (May 1, 2025):** `cache_control` must be on the parent content block, not nested inside `tool_result.content` or `document.source.content`. The library constructs these correctly.

---

## Web search (built-in tool)

Web search is **GA** as of February 17, 2026 — no beta header required.

```python
from reckonsys_llm_core.strategies.claude import WEB_SEARCH_TOOL

response = client.query(
    messages=[ChatMessage(role="user", content="Latest Python release?")],
    tools=[WEB_SEARCH_TOOL],
)
print(response.content)
```

> ⚠️ **Library note:** The library still injects the beta header `web-search-2025-03-05`. This is harmless (the API ignores unknown beta headers) but will be cleaned up in a future release.

---

## New tools available (not yet in library)

These tools are available via the Anthropic API but require raw `raw_config` passthrough or direct API usage:

| Tool | Type string | Notes |
|---|---|---|
| Web fetch | `web_fetch_20250305` | Fetch full content from a URL or PDF |
| Code execution v2 | `computer_20250124` | Bash, multi-language, file manipulation |
| Memory | `memory_20250110` | Store/retrieve info across conversations |
| Tool search | `tool_search_20250305` | Dynamic tool discovery from large catalogs |
| MCP connector | `mcp_20250410` | Connect to remote MCP servers |

Pass any of these via `ToolDefinition(name=..., raw_config={"type": "..."})`.

---

## Document citations

```python
from reckonsys_llm_core import ChatMessage, DocumentContent, TextContent

response = client.query(
    messages=[
        ChatMessage(
            role="user",
            content=[
                DocumentContent(
                    text="The sky appears blue due to Rayleigh scattering...",
                    title="Optics Notes",
                    citations_enabled=True,
                ),
                TextContent(text="Why is the sky blue? Cite your source."),
            ],
        )
    ],
)
for c in response.citations:
    print(c.cited_text, "—", c.document_title)
```

---

## Batch processing

See [Batch Processing guide](../guides/batch-processing.md).

```python
from reckonsys_llm_core import BatchLLMClient, BatchRequest, LLMParams, ChatMessage
from reckonsys_llm_core.strategies.claude import ClaudeBatchStrategy, create_claude_client

strategy = ClaudeBatchStrategy(create_claude_client(), model="claude-sonnet-4-6")
client = BatchLLMClient(strategy)

batch = client.submit([
    BatchRequest("req-1", LLMParams(messages=[ChatMessage(role="user", content="Hello")])),
])
print(batch.batch_id)
```
