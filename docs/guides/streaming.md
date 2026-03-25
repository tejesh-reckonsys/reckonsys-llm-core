# Streaming

Streaming yields tokens incrementally as the model generates them, followed by a final event with full metadata.

---

## Sync streaming

```python
from reckonsys_llm_core import LLMClient, ChatMessage, StreamToken, StreamDone

for event in client.stream_query(
    messages=[ChatMessage(role="user", content="Tell me a story.")],
    system="You are a storyteller.",
):
    if isinstance(event, StreamToken):
        print(event.token, end="", flush=True)
    elif isinstance(event, StreamDone):
        print()
        print(f"Total tokens: {event.usage.total_tokens}")
        print(f"Stop reason: {event.stop_reason}")
```

---

## Async streaming

```python
from reckonsys_llm_core import AsyncLLMClient, StreamToken, StreamDone

async def stream():
    async for event in client.stream_query(
        messages=[ChatMessage(role="user", content="Tell me a story.")],
    ):
        if isinstance(event, StreamToken):
            print(event.token, end="", flush=True)
        elif isinstance(event, StreamDone):
            print()
            print(f"Total tokens: {event.usage.total_tokens}")
```

---

## Event types

### `StreamToken`

Yielded once per token during generation.

| Field | Type | Description |
|---|---|---|
| `token` | `str` | The token text |
| `is_done` | `Literal[False]` | Always `False` — use to distinguish from `StreamDone` |

### `StreamDone`

Final event, yielded after all tokens.

| Field | Type | Description |
|---|---|---|
| `full_content` | `str` | Complete assembled text |
| `usage` | `TokenUsage` | Input/output token counts |
| `model` | `str` | Model ID |
| `stop_reason` | `StopReason \| None` | Why generation stopped |
| `thinking` | `str \| None` | Reasoning trace (Claude / Ollama reasoning models) |
| `citations` | `list[Citation]` | Source citations (Claude only) |
| `is_done` | `Literal[True]` | Always `True` |

---

## Notes

- Streaming is **not retried** on failure. Tokens are forwarded to the caller before the response is complete, so there is no opportunity to retry transparently.
- **Tool calls are not streamed.** If the model calls a tool during a streamed request, tool-call data appears only in `StreamDone` (via the final response metadata from the provider). Use `send_query` / `query` instead if you need to act on tool calls.
- **OpenAI**: only `response.output_text.delta` events are forwarded as `StreamToken`. Other event types (reasoning deltas, tool call deltas) are consumed internally.
- **Thinking**: the `thinking` field on `StreamDone` is populated for Claude and Ollama reasoning models. It is not streamed token-by-token — it appears only in the final event.
