# reckonsys-llm-core

A lightweight, provider-agnostic LLM client library using the strategy pattern.
Supports sync/async execution, streaming, structured output, batch processing, tool calling, and extended thinking.

---

## Navigation

### Providers
- [Claude (Anthropic + Bedrock)](providers/claude.md)
- [OpenAI (Responses API)](providers/openai.md)
- [Ollama](providers/ollama.md)

### Guides
- [Client Helpers](guides/helpers.md)
- [Structured Output](guides/structured-output.md)
- [Streaming](guides/streaming.md)
- [Tool Calling & Agentic Loop](guides/tool-calling.md)
- [Batch Processing](guides/batch-processing.md)
- [Extended Thinking & Reasoning](guides/thinking-reasoning.md)

### Reference
- [Types & Data Models](reference/types.md)
- [API Changes & Library Gaps](api-changes.md)

---

## Architecture

```
LLMClient / AsyncLLMClient
    └── uses  LLMStrategy (Protocol)
                ├── ClaudeLLMStrategy      (Anthropic API / Bedrock)
                ├── OpenAILLMStrategy      (OpenAI Responses API)
                └── OllamaLLMStrategy      (local models)

BatchLLMClient / AsyncBatchLLMClient
    └── uses  BatchLLMStrategy (Protocol)
                ├── ClaudeBatchStrategy
                ├── OpenAIBatchStrategy
                └── (no Ollama batch)
```

Strategies are plain classes — instantiate one, wrap it in a client, call methods. No plugin system, no magic.

---

## Five-minute start

```bash
pip install "reckonsys-llm-core[claude]"
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from reckonsys_llm_core import create_llm, ChatMessage

client = create_llm("claude", "claude-opus-4-6")

response = client.query(
    messages=[ChatMessage(role="user", content="Hello!")],
    system="You are a helpful assistant.",
)
print(response.content)
```

See [Client Helpers](guides/helpers.md) for all providers and async/batch variants, or the [examples/](../examples/) directory for runnable scripts covering every feature.
