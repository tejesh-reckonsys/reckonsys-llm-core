# Batch Processing

Batch processing submits many requests in one call and retrieves results asynchronously — typically at lower cost and higher throughput than individual requests.

---

## Provider support

| Provider | Batch support | Discount |
|---|---|---|
| Claude (Anthropic) | Yes — Message Batches API | Up to 50% cheaper |
| OpenAI | Yes — Batch API via `/v1/responses` | Up to 50% cheaper |
| Ollama | No | — |

---

## Pattern

All batch strategies follow the same three-step pattern:

1. **Submit** — create a batch and get a `batch_id`. Persist this ID.
2. **Poll** — periodically call `status(batch_id)` until `batch.status == BatchStatus.ENDED`.
3. **Retrieve** — stream results with `results(batch_id)`.

Batches expire after 24 hours (both Claude and OpenAI).

---

## Claude

### Sync

```python
from reckonsys_llm_core import (
    BatchLLMClient, BatchRequest, BatchStatus, ChatMessage, LLMParams,
)
from reckonsys_llm_core.strategies.claude import ClaudeBatchStrategy, create_claude_client

strategy = ClaudeBatchStrategy(
    client=create_claude_client(),
    model="claude-opus-4-6",
)
client = BatchLLMClient(strategy)

requests = [
    BatchRequest(
        custom_id="item-1",
        params=LLMParams(
            messages=[ChatMessage(role="user", content="Summarise: ...")],
            system="You are a summariser.",
        ),
    ),
    BatchRequest(
        custom_id="item-2",
        params=LLMParams(
            messages=[ChatMessage(role="user", content="Summarise: ...")],
        ),
    ),
]

# Step 1 — submit
batch = client.submit(requests)
print("batch_id:", batch.batch_id)   # save this

# Step 2 — poll (in a cron job, background task, etc.)
batch = client.status(batch.batch_id)
if batch.status == BatchStatus.ENDED:
    # Step 3 — retrieve
    for result in client.results(batch.batch_id):
        if result.response:
            print(result.custom_id, result.response.content)
        else:
            print(result.custom_id, "ERROR:", result.error)
```

### Async

```python
from reckonsys_llm_core import AsyncBatchLLMClient
from reckonsys_llm_core.strategies.claude import AsyncClaudeBatchStrategy, create_async_claude_client

strategy = AsyncClaudeBatchStrategy(create_async_claude_client(), model="claude-opus-4-6")
client = AsyncBatchLLMClient(strategy)

batch = await client.submit(requests)

# Later...
batch = await client.status(batch.batch_id)
if batch.status == BatchStatus.ENDED:
    async for result in client.results(batch.batch_id):
        print(result.custom_id, result.response.content if result.response else result.error)
```

---

## OpenAI

The OpenAI batch strategy uploads a JSONL file, then creates a batch pointing at `/v1/responses`.

```python
from reckonsys_llm_core import BatchLLMClient, BatchRequest, BatchStatus, LLMParams, ChatMessage
from reckonsys_llm_core.strategies.openai import OpenAIBatchStrategy, create_openai_client

strategy = OpenAIBatchStrategy(create_openai_client(), model="gpt-4o")
client = BatchLLMClient(strategy)

batch = client.submit([
    BatchRequest("req-1", LLMParams(messages=[ChatMessage(role="user", content="Hello")])),
])
print("batch_id:", batch.batch_id)

# Poll
batch = client.status(batch.batch_id)
if batch.status == BatchStatus.ENDED:
    for result in client.results(batch.batch_id):
        print(result.custom_id, result.response.content if result.response else result.error)
```

> **Note:** The `/v1/responses` batch endpoint may not be available on all OpenAI account tiers. Verify access before using in production.

---

## Batch result fields

| Field | Type | Description |
|---|---|---|
| `custom_id` | `str` | Your correlation key from `BatchRequest` |
| `response` | `LLMResponse \| None` | Set on success |
| `error` | `str \| None` | Set when the request failed, was cancelled, or expired |

---

## Cancellation

```python
# Sync
batch = client.strategy.cancel_batch(batch_id)

# Async
batch = await async_strategy.cancel_batch(batch_id)
```

Cancellation is best-effort. Requests already processed are not reversed.

---

## Notes

- Structured output is not supported in batch params. Validate `LLMResponse.content` yourself after retrieving results.
- Results are streamed one by one from the provider to avoid loading large result sets into memory.
- `batch.counts` tracks `processing`, `succeeded`, `errored`, `canceled`, and `expired` counts during polling.
