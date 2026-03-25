# Structured Output

All strategies support structured output via Pydantic models. The mechanism used depends on the provider and the `strict` flag.

---

## Basic usage

```python
from pydantic import BaseModel
from reckonsys_llm_core import LLMClient, ChatMessage

class Person(BaseModel):
    name: str
    age: int

response = client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Alice is 30 years old.")],
    response_models=[Person],
)

print(response.content)       # Person(name='Alice', age=30)
print(response.raw_content)   # raw JSON string before parsing
print(response.error)         # None on success, validation error string on failure
```

---

## Multiple response models

Pass more than one model to let the LLM choose which schema to fill. Useful for routing or union-type extraction.

```python
class Person(BaseModel):
    name: str
    age: int

class Company(BaseModel):
    name: str
    industry: str

response = client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Acme Corp, a tech company.")],
    response_models=[Person, Company],
)
# response.content will be a Company instance
```

Multiple models always use the tool-use approach regardless of `strict`.

---

## Strict mode

### Claude

`strict=True` with a single model uses `output_config` with `json_schema` — guaranteed valid JSON, no tool overhead.

```python
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

strategy = ClaudeLLMStrategy(
    client=create_claude_client(),
    model="claude-opus-4-6",
    strict=True,
)
```

Supported models: `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5`.

### OpenAI

`strict=True` with a single model uses `text.format` with `json_schema` in the Responses API — same guarantee.

```python
from reckonsys_llm_core.strategies.openai import OpenAILLMStrategy, create_openai_client

strategy = OpenAILLMStrategy(
    client=create_openai_client(),
    model="gpt-4o",
    strict=True,
)
```

### Ollama

Single-model structured output always uses Ollama's `format` parameter (constrained decoding). The `strict` flag has no additional effect.

---

## Auto-retry on validation failure (async)

`AsyncLLMClient` retries failed structured queries automatically. On each failure it appends the model's raw output and the exact validation error to the conversation, giving the model the context it needs to self-correct.

```python
from reckonsys_llm_core import AsyncLLMClient, RetryContext

def on_retry(ctx: RetryContext) -> None:
    print(f"Attempt {ctx.attempt} failed: {ctx.error}")

client = AsyncLLMClient(strategy, max_retries=3, on_retry=on_retry)

response = await client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Alice is 30.")],
    response_models=[Person],
)
print(response.attempts)  # 1 if successful on first try
```

The sync `LLMClient` does not retry — wrap it in your own loop if needed.

---

## Response fields

| Field | Description |
|---|---|
| `content` | Parsed `BaseModel` instance, or `None` on error |
| `raw_content` | Raw string before parsing |
| `error` | Validation error message when `content` is `None` |
| `stop_reason` | `StopReason.ERROR` on parse failure, otherwise normal |
| `usage` | Token counts |
| `attempts` | Number of attempts (async client) |
