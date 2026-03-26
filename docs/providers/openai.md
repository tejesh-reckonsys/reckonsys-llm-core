# OpenAI (Responses API)

> Last reviewed: March 2026

This strategy uses OpenAI's **Responses API** (`client.responses.create`), which replaced the Chat Completions API as the primary recommended interface in March 2025.

## Installation

```bash
pip install "reckonsys-llm-core[openai]"   # openai>=2.0.0
```

Set `OPENAI_API_KEY` in your environment (or pass it explicitly via `create_openai_client(api_key=...)`).

> **Tip:** For the quickest setup, use the [Client Helpers](../guides/helpers.md) — e.g. `create_llm("openai", "gpt-5.4-mini")`.

---

## Current models

### GPT-5 family (recommended)

| Model | Context | Best for |
|---|---|---|
| `gpt-5.4` | 1M tokens | Complex tasks, built-in computer use, tool search |
| `gpt-5.4-mini` | 400k tokens | High-volume, lower-latency |
| `gpt-5.4-nano` | 400k tokens | Cost-efficient, fastest |
| `gpt-5.2` | — | Reasoning with `xhigh` effort |
| `gpt-5.1` | — | Steerability, code, agentic workflows |

### GPT-4.1 family

| Model | Context | Notes |
|---|---|---|
| `gpt-4.1` | 1M tokens | Strong coding and instruction following |
| `gpt-4.1-mini` | 1M tokens | Cost-efficient |
| `gpt-4.1-nano` | 1M tokens | Fastest in the family |

### o-series (reasoning models)

| Model | Best for |
|---|---|
| `o4-mini` | Fast reasoning — math, code (best AIME 2024/2025) |
| `o3` | Deep reasoning — tools, visual reasoning |
| `o3-pro` | Hardest problems, higher consistency |

### Deprecated

| Model | Status |
|---|---|
| `gpt-4o`, `o4-mini` (old snapshots) | Returning 404 as of Feb 16, 2026 |
| `gpt-4.5-preview` | Deprecated Apr 2025 |

---

## Strategies

| Class | Client type | Use case | `provider_name` |
|---|---|---|---|
| `OpenAILLMStrategy` | `OpenAI` | Sync | `"openai"` |
| `AsyncOpenAILLMStrategy` | `AsyncOpenAI` | Async | `"openai"` |
| `OpenAIBatchStrategy` | `OpenAI` | Sync batch | `"openai"` |
| `AsyncOpenAIBatchStrategy` | `AsyncOpenAI` | Async batch | `"openai"` |

---

## Quick start

```python
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.strategies.openai import OpenAILLMStrategy, create_openai_client

strategy = OpenAILLMStrategy(
    client=create_openai_client(),   # reads OPENAI_API_KEY
    model="gpt-5.4-mini",
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
from reckonsys_llm_core import AsyncLLMClient
from reckonsys_llm_core.strategies.openai import AsyncOpenAILLMStrategy, create_async_openai_client

strategy = AsyncOpenAILLMStrategy(
    client=create_async_openai_client(),
    model="gpt-5.4-mini",
)
client = AsyncLLMClient(strategy)
```

---

## Structured output

### Tool-use approach (default, all models)

```python
from pydantic import BaseModel

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

Pass `strict=True` to use the Responses API `text.format` with `json_schema`.

- Supported on `gpt-4.1` and later, all GPT-5 models, o-series
- With a single `response_model` → uses `text.format` (no tools)
- With multiple `response_models` → falls back to function calling with `strict: true`

```python
strategy = OpenAILLMStrategy(
    client=create_openai_client(),
    model="gpt-5.4-mini",
    strict=True,
)
```

---

## Reasoning models (o-series)

```python
from reckonsys_llm_core import ThinkingConfig

response = client.query(
    messages=[ChatMessage(role="user", content="Solve this step by step...")],
    thinking=ThinkingConfig(enabled=True, reasoning_effort="high"),
)
print(response.thinking)
print(response.content)
print(response.usage.reasoning_tokens)
```

### `reasoning_effort` levels

| Value | Description |
|---|---|
| `"minimal"` | Least reasoning — new, added with GPT-5 family |
| `"low"` | Fast, minimal reasoning |
| `"medium"` | Balanced (default when `enabled=True`) |
| `"high"` | Deep reasoning |
| `"xhigh"` | Maximum effort — added with GPT-5.2 |

> ⚠️ `enabled=True` on non-reasoning models (e.g. `gpt-4.1`) causes an API error.

---

## Web search (built-in tool)

```python
from reckonsys_llm_core.strategies.openai import OPENAI_WEB_SEARCH_TOOL

response = client.query(
    messages=[ChatMessage(role="user", content="Latest Python release?")],
    tools=[OPENAI_WEB_SEARCH_TOOL],
)
print(response.content)
```

`OPENAI_WEB_SEARCH_TOOL` passes `{"type": "web_search"}`. An updated type string `"web_search_2025_08_26"` is also accepted by the API — use via `raw_config` if you need the newer snapshot.

---

## New built-in tools (not yet in library)

These tools are available in the Responses API but require `raw_config` passthrough via `ToolDefinition`:

| Tool | `type` string | Added |
|---|---|---|
| File search | `file_search` | Mar 2025 (launch) |
| Computer use | `computer_use_preview` | Mar 2025 (launch) |
| Code interpreter | `code_interpreter` | May 2025 |
| Image generation | `image_generation` | Dec 2025 |
| Tool search | `tool_search` | Mar 2026 |
| MCP connector | `mcp` (with `server_label`, `type`, `allowed_tools`) | May 2025 |

Pass any via `ToolDefinition(name=..., raw_config={"type": "..."})`.

---

## Custom tools

```python
from reckonsys_llm_core import ToolDefinition

TOOLS = [
    ToolDefinition(
        name="get_weather",
        description="Return the current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
]

response = client.run_agent(
    messages=[ChatMessage(role="user", content="Weather in Paris?")],
    tools=TOOLS,
    tool_executor=lambda name, inputs: f"Sunny in {inputs['city']}",
)
```

---

## Batch processing

```python
from reckonsys_llm_core import BatchLLMClient, BatchRequest, LLMParams, ChatMessage
from reckonsys_llm_core.strategies.openai import OpenAIBatchStrategy, create_openai_client

strategy = OpenAIBatchStrategy(create_openai_client(), model="gpt-5.4-mini")
client = BatchLLMClient(strategy)

batch = client.submit([
    BatchRequest("req-1", LLMParams(messages=[ChatMessage(role="user", content="Hello")])),
])
print(batch.batch_id)
```

See [Batch Processing guide](../guides/batch-processing.md) for the full poll-and-retrieve pattern.

---

## Gotchas

### `stop` sequences are ignored
The Responses API has no `stop` parameter. Values in `LLMParams.stop` are silently dropped.

### `cache_system` and `ChatMessage.cache` have no effect
Prompt caching flags are Claude-specific. They are ignored here.

### `DocumentContent` is flattened to plain text
No native document block type exists in the Responses API. `DocumentContent` becomes an `input_text` block with the title prepended. `LLMResponse.citations` will always be empty.

### `ThinkingConfig.budget_tokens` is ignored
Use `reasoning_effort` instead. `budget_tokens` is a Claude concept.

### `ThinkingConfig.enabled=True` crashes on non-reasoning models
Only set it for o-series and GPT-5 models that support reasoning (o3, o4-mini, o3-pro, gpt-5.2+). Using it on `gpt-4.1` or `gpt-5.4` (non-reasoning paths) will cause an API error.

### `reasoning_effort="minimal"` and `"xhigh"` require newer models
- `"minimal"` — added with the GPT-5 family (Aug 2025)
- `"xhigh"` — added with GPT-5.2 (Dec 2025)
Passing these to earlier models will cause an API error.

### Claude built-in `raw_config` tools are passed through as-is
If you accidentally pass Claude's `WEB_SEARCH_TOOL` (type `web_search_20250305`), the dict is forwarded directly and will cause an API error. Use `OPENAI_WEB_SEARCH_TOOL` instead.

### Tool-call multi-turn conversation layout
`ToolUseContent` becomes a top-level `function_call` input item; `ToolResultContent` becomes a top-level `function_call_output` item. Both are lifted out of their parent `ChatMessage` automatically — no change needed in your code.

### Streaming only yields text tokens
Tool-call delta events are not surfaced as `StreamToken` events. Full tool-call data is only available on the `LLMResponse` from `send_query` / `query`.

### Batch endpoint support may vary by tier
Batch requests target `/v1/responses`. Verify your account tier supports this endpoint before using in production.

### Avoid setting both `temperature` and `top_p`
OpenAI recommends changing only one at a time. Setting both may produce unexpected results.

### Assistants API is deprecated
If you are migrating from the Assistants API: it sunsets **August 26, 2026**. Migrate to the Responses API + Conversations API. Threads → Conversations, Assistants → Prompts.
