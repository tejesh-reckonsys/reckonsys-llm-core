# reckonsys-llm-core

A lightweight, provider-agnostic LLM client library with a strategy pattern. Currently supports Claude (Anthropic API and AWS Bedrock) and Ollama.

## Installation

```bash
# Base package (no provider dependencies)
pip install reckonsys-llm-core

# With Claude support
pip install "reckonsys-llm-core[claude]"

# With Ollama support
pip install "reckonsys-llm-core[ollama]"

# With Jinja2 template support
pip install "reckonsys-llm-core[templates]"

# Everything
pip install "reckonsys-llm-core[all]"
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

# With Ollama support
pip install "dist/reckonsys_llm_core-*.whl[ollama]"

# All providers
pip install "dist/reckonsys_llm_core-*.whl[all]"

# Using uv
uv pip install "dist/reckonsys_llm_core-*.whl[all]"
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

### Ollama

Requires a running [Ollama](https://ollama.com) server (defaults to `http://localhost:11434`).

```python
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.stratagies.ollama import OllamaLLMStrategy

strategy = OllamaLLMStrategy(model="llama3.2")
client = LLMClient(strategy)

response = client.query(
    messages=[ChatMessage(role="user", content="Hello!")],
    system="You are a helpful assistant.",
)
print(response.content)
```

Custom host:

```python
strategy = OllamaLLMStrategy(model="llama3.2", host="http://my-server:11434")
```

#### Ollama structured output

```python
response = client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Alice is 30 years old.")],
    response_models=[Person],
)
print(response.content)  # Person(name='Alice', age=30)
```

- **Single response model** → uses Ollama's `format` parameter with a JSON schema (constrained decoding)
- **Multiple response models** → uses tool-use (requires a model with tool support, e.g. `llama3.2`, `qwen2.5`)

#### Ollama extended thinking

Supported on models with reasoning capability (e.g. `deepseek-r1`, `qwen3`).

```python
from reckonsys_llm_core import ThinkingConfig

response = client.query(
    messages=[ChatMessage(role="user", content="Solve this step by step...")],
    thinking=ThinkingConfig(enabled=True),
)
print(response.thinking)  # reasoning trace
print(response.content)   # final answer
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

## Prompt templates

Install the `templates` extra, then call `configure_templates()` once at startup (e.g. in `settings.py` or `main.py`):

```python
from reckonsys_llm_core.templates import configure_templates, render_prompt

configure_templates("/path/to/prompts")  # accepts any jinja2.Environment kwargs too
```

Then render anywhere:

```python
system = render_prompt("system.md.j2", {"role": "analyst"})
user   = render_prompt("user.md.j2",   {"query": query})

response = client.query(
    messages=[ChatMessage(role="user", content=user)],
    system=system,
)
```

`configure_templates` accepts any extra `jinja2.Environment` kwargs:

```python
configure_templates("/path/to/prompts", trim_blocks=True, lstrip_blocks=True)
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
