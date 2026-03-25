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
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

client = LLMClient(ClaudeLLMStrategy(create_claude_client(), model="claude-opus-4-6"))

response = client.query(
    messages=[ChatMessage(role="user", content="Hello!")],
    system="You are a helpful assistant.",
)
print(response.content)
```

See the [examples/](../examples/) directory for runnable scripts covering every feature.
