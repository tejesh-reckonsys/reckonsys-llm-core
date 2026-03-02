# reckonsys-llm-core

A lightweight, provider-agnostic LLM client library with a strategy pattern. Currently supports Claude (Anthropic API and AWS Bedrock).

## Installation

```bash
# Base package (no provider dependencies)
pip install reckonsys-llm-core

# With Claude support
pip install "reckonsys-llm-core[claude]"
```

## Building a wheel

```bash
# Install uv if you don't have it
pip install uv

# Build — produces dist/reckonsys_llm_core-<version>-py3-none-any.whl
uv build
```

The wheel file will appear in the `dist/` directory.

## Installing from a wheel

```bash
# Base only
pip install dist/reckonsys_llm_core-*.whl

# With Claude support
pip install "dist/reckonsys_llm_core-*.whl[claude]"

# Using uv
uv pip install dist/reckonsys_llm_core-*.whl
uv pip install "dist/reckonsys_llm_core-*.whl[claude]"
```

## Quick start

### Plain text query

```python
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.stratagies.claude import ClaudeLLMStrategy, create_claude_client

strategy = ClaudeLLMStrategy(
    client=create_claude_client(),  # reads ANTHROPIC_API_KEY from env
    model="claude-opus-4-6",
)
client = LLMClient(strategy)

response = client.query(
    messages=[ChatMessage(role="user", content="Hello!")],
    system="You are a helpful assistant.",
)
print(response.content)
```

### Structured output (tool-use)

```python
from pydantic import BaseModel
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.stratagies.claude import ClaudeLLMStrategy, create_claude_client

class Person(BaseModel):
    name: str
    age: int

strategy = ClaudeLLMStrategy(
    client=create_claude_client(),
    model="claude-opus-4-6",
)
client = LLMClient(strategy)

response = client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Alice is 30 years old.")],
    response_models=[Person],
)
print(response.content)  # Person(name='Alice', age=30)
```

### Structured output (native JSON schema — strict mode)

Pass `strict=True` when creating the strategy to use Claude's native `output_config` structured outputs. Supported on `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5`.

- **Single response model** → uses `output_config` with `json_schema` (guaranteed valid JSON, no tools)
- **Multiple response models** → uses tool-use with `strict: true` on each tool

```python
strategy = ClaudeLLMStrategy(
    client=create_claude_client(),
    model="claude-opus-4-6",
    strict=True,
)
client = LLMClient(strategy)

response = client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Alice is 30 years old.")],
    response_models=[Person],
)
print(response.content)  # Person(name='Alice', age=30)
```

### AWS Bedrock

```python
from reckonsys_llm_core.stratagies.claude import ClaudeLLMStrategy, create_bedrock_client

strategy = ClaudeLLMStrategy(
    client=create_bedrock_client(region="us-west-2"),  # reads AWS_IAM_ACCESS_KEY / AWS_IAM_SECRET_KEY from env
    model="anthropic.claude-opus-4-6-20251101-v1:0",
)
```

### Extended thinking

```python
from reckonsys_llm_core import ThinkingConfig

response = client.query(
    messages=[ChatMessage(role="user", content="Solve this step by step...")],
    thinking=ThinkingConfig(enabled=True, budget_tokens=5000),
)
print(response.thinking)  # reasoning trace
print(response.content)   # final answer
```

## Response fields

`LLMResponse` and `LLMStructuredResponse` both expose:

| Field | Type | Description |
|---|---|---|
| `content` | `str` / `BaseModel \| None` | Model output |
| `usage` | `TokenUsage` | Input, output, cache tokens |
| `model` | `str` | Model ID |
| `stop_reason` | `StopReason \| None` | `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`, `error` |
| `thinking` | `str \| None` | Extended thinking trace (if enabled) |
