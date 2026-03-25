# Extended Thinking & Reasoning

> Last reviewed: March 2026

Extended thinking lets the model reason step-by-step before producing its final answer. The reasoning trace is returned separately from the response content.

---

## Provider support

| Provider | Supported models | Config to use |
|---|---|---|
| Claude | Opus 4.6, Sonnet 4.6, Haiku 4.5 | `budget_tokens` (legacy) or `effort` (new, not yet in library) |
| OpenAI | o3, o4-mini, o3-pro, GPT-5.2+, GPT-5.4+ | `reasoning_effort` |
| Ollama | deepseek-r1, qwen3, other reasoning models | `enabled=True` |

---

## Claude

### Current approach — `budget_tokens`

```python
from reckonsys_llm_core import LLMClient, ChatMessage, ThinkingConfig
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

strategy = ClaudeLLMStrategy(create_claude_client(), model="claude-opus-4-6")
client = LLMClient(strategy)

response = client.query(
    messages=[
        ChatMessage(
            role="user",
            content="A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        )
    ],
    thinking=ThinkingConfig(enabled=True, budget_tokens=5000),
)

print("Thinking:", response.thinking)
print("Answer:", response.content)
```

### New `effort` parameter (Opus 4.6 / Sonnet 4.6)

Anthropic introduced a higher-level `effort` parameter for Claude Opus 4.6 and Sonnet 4.6. These models use **adaptive thinking** by default — they decide how much thinking is needed.

> ⚠️ **Not yet implemented in this library.** `budget_tokens` still works but is considered legacy for new models. The `effort` API parameter is not yet passed through. Track this as a planned update.

When implemented, the API call will look like:
```json
{
  "thinking": { "type": "enabled", "effort": "high" }
}
```

Effort levels: `"low"`, `"medium"`, `"high"`.

### Thinking display modes (new, Mar 2026)

Anthropic added `thinking.display: "omitted"` — you can request that the thinking blocks have their content stripped from the response (the encrypted `signature` is preserved for multi-turn correctness, but the raw text is not returned). Useful for reducing streaming latency. Not yet exposed in this library.

### Claude thinking + tools

When `thinking` is enabled alongside tools, the strategy automatically sets `temperature=1` (required by the API). You do not need to set this yourself.

---

## OpenAI (o-series and GPT-5)

```python
from reckonsys_llm_core.strategies.openai import OpenAILLMStrategy, create_openai_client

strategy = OpenAILLMStrategy(create_openai_client(), model="o4-mini")
client = LLMClient(strategy)

response = client.query(
    messages=[ChatMessage(role="user", content="Solve this step by step...")],
    thinking=ThinkingConfig(enabled=True, reasoning_effort="high"),
)

print("Thinking:", response.thinking)
print("Answer:", response.content)
print("Reasoning tokens:", response.usage.reasoning_tokens)
```

### `reasoning_effort` levels

| Value | Notes |
|---|---|
| `"minimal"` | Added with GPT-5 family (Aug 2025) — not supported on earlier models |
| `"low"` | Fast, minimal reasoning |
| `"medium"` | Default when `enabled=True` |
| `"high"` | Deep, thorough reasoning |
| `"xhigh"` | Added with GPT-5.2 (Dec 2025) — not supported on earlier models |

> `budget_tokens` is ignored for OpenAI — set `reasoning_effort` via `ThinkingConfig.reasoning_effort` instead.

> ⚠️ Setting `enabled=True` on a non-reasoning model (e.g. `gpt-4.1`) causes an API error.

### o3 / o4-mini reasoning across tool calls

In the Responses API, `o3` and `o4-mini` preserve reasoning tokens across tool calls within a single request — they reason about tool results without re-reading the full conversation. This is automatic when using the Responses API.

---

## Ollama

```python
from reckonsys_llm_core.strategies.ollama import OllamaLLMStrategy

strategy = OllamaLLMStrategy(model="deepseek-r1")
client = LLMClient(strategy)

response = client.query(
    messages=[ChatMessage(role="user", content="Explain your reasoning...")],
    thinking=ThinkingConfig(enabled=True),
)

print("Thinking:", response.thinking)   # extracted from <think>...</think> tags
print("Answer:", response.content)
```

---

## Response fields

| Field | Location | Description |
|---|---|---|
| `response.thinking` | `LLMResponse` | Reasoning trace, or `None` if not enabled / not returned |
| `response.usage.reasoning_tokens` | `TokenUsage` | Tokens spent on reasoning (OpenAI o-series / GPT-5 only) |
| `StreamDone.thinking` | streaming | Populated in final event; not streamed token-by-token |
