# Client Helpers

> Last reviewed: March 2026

Instead of manually instantiating a strategy and wrapping it in a client, the four helper functions let you get a ready-to-use client in one call.

```python
from reckonsys_llm_core import create_llm, create_async_llm, create_batch_llm, create_async_batch_llm
```

---

## Functions

| Helper | Returns | Use case |
|---|---|---|
| `create_llm` | `LLMClient` | Sync scripts, CLI tools, eval pipelines |
| `create_async_llm` | `AsyncLLMClient` | FastAPI, async pipelines |
| `create_batch_llm` | `BatchLLMClient` | Sync batch submission |
| `create_async_batch_llm` | `AsyncBatchLLMClient` | Async batch submission |

All four share the same provider / model / credential interface. Differences are noted per-parameter below.

---

## Common parameters

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `provider` | `str` | — | all |
| `model` | `str` | — | all |
| `api_key` | `str \| None` | `None` (reads env) | claude, openai |
| `access_key` | `str \| None` | `None` (reads env) | claude_bedrock |
| `secret_key` | `str \| None` | `None` (reads env) | claude_bedrock |
| `region` | `str` | `"us-west-2"` | claude_bedrock |
| `host` | `str` | `"http://localhost:11434"` | ollama |
| `default_max_tokens` | `int \| None` | `None` (provider default) | all |
| `strict` | `bool` | `False` | claude, openai |
| `max_retries` | `int` | `2` | create_llm, create_async_llm |
| `on_retry` | `Callable \| None` | `None` | create_llm, create_async_llm |

`default_max_tokens` provider defaults: `8000` (claude, ollama), `4096` (openai).

Batch helpers (`create_batch_llm`, `create_async_batch_llm`) do **not** accept `strict`, `max_retries`, or `on_retry` — those concepts don't apply to the batch API.

---

## Supported providers

| `provider` | LLM helpers | Batch helpers | Docs |
|---|---|---|---|
| `"claude"` | ✅ | ✅ | [providers/claude.md](../providers/claude.md) |
| `"claude_bedrock"` | ✅ | ✅ | [providers/claude.md](../providers/claude.md) |
| `"openai"` | ✅ | ✅ | [providers/openai.md](../providers/openai.md) |
| `"ollama"` | ✅ | ❌ | [providers/ollama.md](../providers/ollama.md) |

---

## Examples

### Sync — Claude

```python
from reckonsys_llm_core import create_llm, ChatMessage

client = create_llm("claude", "claude-sonnet-4-6")
response = client.query(
    messages=[ChatMessage(role="user", content="Hello!")],
    system="You are a helpful assistant.",
)
print(response.content)
```

Credentials are read from `ANTHROPIC_API_KEY`. Pass `api_key=` to override.

See [providers/claude.md](../providers/claude.md) for available models and `strict` mode details.

---

### Async — OpenAI

```python
import asyncio
from reckonsys_llm_core import create_async_llm, ChatMessage

async def main():
    client = create_async_llm("openai", "gpt-5.4-mini")
    response = await client.query(
        messages=[ChatMessage(role="user", content="Hello!")],
    )
    print(response.content)

asyncio.run(main())
```

Credentials are read from `OPENAI_API_KEY`. Pass `api_key=` to override.

See [providers/openai.md](../providers/openai.md) for available models, reasoning effort levels, and built-in tools.

---

### Sync — Claude on Bedrock

```python
from reckonsys_llm_core import create_llm, ChatMessage

client = create_llm(
    "claude_bedrock",
    "anthropic.claude-sonnet-4-6-20251101-v1:0",
    region="us-east-1",
)
response = client.query(
    messages=[ChatMessage(role="user", content="Hello!")],
)
print(response.content)
```

Credentials are read from `AWS_IAM_ACCESS_KEY` / `AWS_IAM_SECRET_KEY`. Pass `access_key=` and `secret_key=` to override.

See [providers/claude.md](../providers/claude.md) for Bedrock model IDs and region notes.

---

### Async — Ollama

```python
import asyncio
from reckonsys_llm_core import create_async_llm, ChatMessage

async def main():
    client = create_async_llm("ollama", "llama3.2")
    response = await client.query(
        messages=[ChatMessage(role="user", content="Hello!")],
    )
    print(response.content)

asyncio.run(main())
```

Point at a non-default server with `host="http://my-server:11434"`.

See [providers/ollama.md](../providers/ollama.md) for model names and extended-thinking support.

---

### Sync batch — OpenAI

```python
from reckonsys_llm_core import create_batch_llm, BatchRequest, LLMParams, ChatMessage

client = create_batch_llm("openai", "gpt-5.4-mini")

batch = client.submit([
    BatchRequest("req-1", LLMParams(messages=[ChatMessage(role="user", content="Hello")])),
    BatchRequest("req-2", LLMParams(messages=[ChatMessage(role="user", content="World")])),
])
print(batch.batch_id)  # persist this
```

---

### Async batch — Claude

```python
import asyncio
from reckonsys_llm_core import create_async_batch_llm, BatchRequest, LLMParams, ChatMessage

async def main():
    client = create_async_batch_llm("claude", "claude-sonnet-4-6")
    batch = await client.submit([
        BatchRequest("req-1", LLMParams(messages=[ChatMessage(role="user", content="Hello")])),
    ])
    print(batch.batch_id)

asyncio.run(main())
```

See [guides/batch-processing.md](batch-processing.md) for the full poll-and-retrieve pattern.

---

### Strict mode + retry callback

```python
from reckonsys_llm_core import create_llm, RetryContext

def log_retry(ctx: RetryContext) -> None:
    print(f"Attempt {ctx.attempt} failed: {ctx.error}")

client = create_llm(
    "openai", "gpt-5.4-mini",
    strict=True,
    max_retries=3,
    on_retry=log_retry,
)
```

`strict=True` enables native JSON-schema structured output on supported models.
See [guides/structured-output.md](structured-output.md) for details.
