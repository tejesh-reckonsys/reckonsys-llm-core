# Ollama

Run models locally via a running [Ollama](https://ollama.com) server.

## Installation

```bash
pip install "reckonsys-llm-core[ollama]"
```

Ollama must be running locally (or on a reachable host). Default endpoint: `http://localhost:11434`.

```bash
# Install and start Ollama
ollama serve

# Pull a model
ollama pull llama3.2
```

---

## Strategies

| Class | Use case | `provider_name` |
|---|---|---|
| `OllamaLLMStrategy` | Sync | `"ollama"` |
| `AsyncOllamaLLMStrategy` | Async | `"ollama"` |

No batch strategy — Ollama has no native batch API.

---

## Quick start

```python
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.strategies.ollama import OllamaLLMStrategy

strategy = OllamaLLMStrategy(model="llama3.2")
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
from reckonsys_llm_core.strategies.ollama import AsyncOllamaLLMStrategy

strategy = AsyncOllamaLLMStrategy(model="llama3.2")
client = AsyncLLMClient(strategy)
```

---

## Custom host

```python
strategy = OllamaLLMStrategy(
    model="llama3.2",
    host="http://my-server:11434",
)
```

---

## Structured output

### Single response model — constrained decoding

Uses Ollama's `format` parameter with a JSON schema. Produces guaranteed-valid JSON without tool-use overhead.

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

### Multiple response models — tool-use

Requires a model with tool support (e.g. `llama3.2`, `qwen2.5`).

```python
response = client.query_structured(
    messages=[...],
    response_models=[Person, Company],
)
```

---

## Extended thinking

Supported on reasoning-capable models such as `deepseek-r1` and `qwen3`.

```python
from reckonsys_llm_core import ThinkingConfig

response = client.query(
    messages=[ChatMessage(role="user", content="Solve this step by step...")],
    thinking=ThinkingConfig(enabled=True),
)
print(response.thinking)  # reasoning trace (extracted from <think> tags)
print(response.content)   # final answer
```

---

## Notes

- **Images**: base64 images work; URL images are not supported — use `ImageContent(is_url=False, ...)`.
- **DocumentContent**: not natively supported; passed as plain text.
- **Prompt caching** (`cache`, `cache_system`): ignored — Ollama has no caching API.
- **Citations**: always empty.
- **Tool availability**: not all Ollama models support tools. Check the model's Ollama page before using tool calling.
