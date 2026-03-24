# reckonsys-llm-core

A lightweight, provider-agnostic LLM client library with a strategy pattern. Supports **sync and async** clients, **streaming**, **structured output**, **batch processing**, and **extended thinking**. Providers: Claude (Anthropic API and AWS Bedrock) and Ollama.

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

---

## Clients at a glance

| Client | Use case |
|---|---|
| `LLMClient` | Synchronous — scripts, CLI tools, eval pipelines |
| `AsyncLLMClient` | Async — FastAPI, async pipelines; retry-on-failure for structured queries |
| `BatchLLMClient` | Sync batch submission via Claude's Message Batches API |
| `AsyncBatchLLMClient` | Async batch submission (up to 50% cheaper, higher rate limits) |

---

## Quick start

### Synchronous plain text query

```python
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

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

### Async plain text query

```python
import asyncio
from reckonsys_llm_core import AsyncLLMClient, ChatMessage
from reckonsys_llm_core.strategies.claude import AsyncClaudeLLMStrategy, create_async_claude_client

strategy = AsyncClaudeLLMStrategy(
    client=create_async_claude_client(),
    model="claude-opus-4-6",
)
client = AsyncLLMClient(strategy)

async def main():
    response = await client.query(
        messages=[ChatMessage(role="user", content="Hello!")],
        system="You are a helpful assistant.",
    )
    print(response.content)

asyncio.run(main())
```

### Async streaming

```python
from reckonsys_llm_core import StreamToken, StreamDone

async def stream():
    async for event in client.stream_query(
        messages=[ChatMessage(role="user", content="Tell me a story.")],
    ):
        if isinstance(event, StreamToken):
            print(event.token, end="", flush=True)
        elif isinstance(event, StreamDone):
            print()  # newline after stream
            print(f"Tokens used: {event.usage.total_tokens}")
```

> Streaming is not retried on failure — tokens are forwarded to the caller before the response is complete.

---

## Structured output

### Tool-use (all providers)

```python
from pydantic import BaseModel
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

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

### Native JSON schema — strict mode (Claude only)

Pass `strict=True` to use Claude's native `output_config` structured outputs. Supported on `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-haiku-4-5`.

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

### Async structured query with retry

`AsyncLLMClient` automatically retries failed structured queries. On each validation failure, the LLM's raw output and the exact error are appended to the conversation so the model can self-correct — far more effective than blind retries.

```python
from reckonsys_llm_core import AsyncLLMClient, RetryContext

def on_retry(ctx: RetryContext) -> None:
    print(f"Attempt {ctx.attempt} failed: {ctx.error}")
    # integrate with OTel, Prometheus, or your eval dataset here

client = AsyncLLMClient(strategy, max_retries=2, on_retry=on_retry)

async def main():
    response = await client.query_structured(
        messages=[ChatMessage(role="user", content="Extract: Alice is 30 years old.")],
        response_models=[Person],
    )
    print(response.content)   # Person(name='Alice', age=30)
    print(response.attempts)  # number of attempts made
```

---

## Batch processing (Claude only)

Claude's Message Batches API processes up to 100k requests per batch at up to 50% lower cost and higher rate limits. Batches expire after 24 hours if not completed.

The caller is responsible for persisting `batch_id`, deciding when to poll, and acting on results.

### Synchronous batch

```python
from reckonsys_llm_core import BatchLLMClient, BatchRequest, BatchStatus, ChatMessage, LLMParams
from reckonsys_llm_core.strategies.claude import ClaudeBatchStrategy, create_claude_client

strategy = ClaudeBatchStrategy(
    client=create_claude_client(),
    model="claude-opus-4-6",
)
client = BatchLLMClient(strategy)

requests = [
    BatchRequest(
        custom_id="row-1",
        params=LLMParams(
            messages=[ChatMessage(role="user", content="Summarize: ...")],
            system="You are a summarizer.",
        ),
    ),
    BatchRequest(
        custom_id="row-2",
        params=LLMParams(
            messages=[ChatMessage(role="user", content="Summarize: ...")],
        ),
    ),
]

# Submit
batch = client.submit(requests)
print(batch.batch_id)  # persist this to your DB

# Poll (in a cron job or background task)
batch = client.status(batch.batch_id)
if batch.status == BatchStatus.ENDED:
    for result in client.results(batch.batch_id):
        if result.response:
            print(result.custom_id, result.response.content)
        else:
            print(result.custom_id, "ERROR:", result.error)
```

### Async batch

```python
from reckonsys_llm_core import AsyncBatchLLMClient
from reckonsys_llm_core.strategies.claude import AsyncClaudeBatchStrategy, create_async_claude_client

strategy = AsyncClaudeBatchStrategy(
    client=create_async_claude_client(),
    model="claude-opus-4-6",
)
client = AsyncBatchLLMClient(strategy)

async def run():
    batch = await client.submit(requests)

    # Later, poll and retrieve
    batch = await client.status(batch.batch_id)
    if batch.status == BatchStatus.ENDED:
        async for result in client.results(batch.batch_id):
            if result.response:
                print(result.custom_id, result.response.content)
            else:
                print(result.custom_id, "ERROR:", result.error)
```

---

## Ollama

Requires a running [Ollama](https://ollama.com) server (defaults to `http://localhost:11434`).

```python
from reckonsys_llm_core import LLMClient, AsyncLLMClient, ChatMessage
from reckonsys_llm_core.strategies.ollama import OllamaLLMStrategy, AsyncOllamaLLMStrategy

# Synchronous
strategy = OllamaLLMStrategy(model="llama3.2")
client = LLMClient(strategy)

# Async
async_strategy = AsyncOllamaLLMStrategy(model="llama3.2")
async_client = AsyncLLMClient(async_strategy)

# Custom host
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

---

## AWS Bedrock

```python
from reckonsys_llm_core.strategies.claude import (
    ClaudeLLMStrategy,
    AsyncClaudeLLMStrategy,
    create_bedrock_client,
    create_async_bedrock_client,
)

# Sync
strategy = ClaudeLLMStrategy(
    client=create_bedrock_client(region="us-west-2"),  # reads AWS_IAM_ACCESS_KEY / AWS_IAM_SECRET_KEY from env
    model="anthropic.claude-opus-4-6-20251101-v1:0",
)

# Async
async_strategy = AsyncClaudeLLMStrategy(
    client=create_async_bedrock_client(region="us-west-2"),
    model="anthropic.claude-opus-4-6-20251101-v1:0",
)
```

---

## Extended thinking (Claude)

```python
from reckonsys_llm_core import ThinkingConfig

response = client.query(
    messages=[ChatMessage(role="user", content="Solve this step by step...")],
    thinking=ThinkingConfig(enabled=True, budget_tokens=5000),
)
print(response.thinking)  # reasoning trace
print(response.content)   # final answer
```

---

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

---

## Multi-turn conversations

The clients are stateless — you own the message history. Append each assistant reply to the list and pass it all on the next call:

```python
messages: list[ChatMessage] = []

for user_text in ["Tell me a joke.", "Explain why it's funny."]:
    messages.append(ChatMessage(role="user", content=user_text))
    response = client.query(messages=messages, system=SYSTEM)
    messages.append(ChatMessage(role="assistant", content=response.content))
    print(response.content)
```

See [examples/09_multi_turn.py](examples/09_multi_turn.py) for a scripted pipeline and an interactive REPL.

---

## Custom tool calling (agentic loop)

Define tools with `ToolDefinition` and pass a `tool_executor` callable to `run_agent` / `arun_agent`. The library handles the full loop.

```python
from reckonsys_llm_core import LLMClient, ChatMessage, ToolDefinition

TOOLS = [
    ToolDefinition(
        name="get_weather",
        description="Return the current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
]

def tool_executor(name: str, inputs: dict) -> str:
    if name == "get_weather":
        return f"Sunny, 22°C in {inputs['city']}"
    return f"Unknown tool: {name}"

response = client.run_agent(
    messages=[ChatMessage(role="user", content="What's the weather in Paris?")],
    tools=TOOLS,
    tool_executor=tool_executor,
)
print(response.content)
```

For async use `arun_agent` — it accepts both sync and async `tool_executor` callables:

```python
response = await async_client.arun_agent(
    messages=[ChatMessage(role="user", content="What's the weather in Tokyo?")],
    tools=TOOLS,
    tool_executor=tool_executor,   # or an async def
    max_iterations=10,
)
```

See [examples/10_custom_tools.py](examples/10_custom_tools.py) for a full working calculator agent.

---

## Functions as tools

Instead of writing `ToolDefinition` + JSON schema by hand, derive everything from the function's name, type annotations, defaults, and docstring.

`tool_from_function(fn)` converts a single function:

```python
from typing import Literal
from reckonsys_llm_core import tool_from_function

def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Return the current weather for a city.

    Args:
        city: The city name.
        unit: Temperature unit.
    """
    ...

tool_def = tool_from_function(get_weather)
# ToolDefinition(
#   name="get_weather",
#   description="Return the current weather for a city.",
#   input_schema={
#     "type": "object",
#     "properties": {
#       "city": {"type": "string", "description": "The city name."},
#       "unit": {"enum": ["celsius", "fahrenheit"], "default": "celsius", ...},
#     },
#     "required": ["city"],
#   }
# )
```

`from_tools(*fns)` converts multiple functions and returns a `ToolKit(tools, executor)` that unpacks directly into `run_agent`:

```python
from reckonsys_llm_core import from_tools

tools, executor = from_tools(get_weather, calculate, get_current_time)

response = client.run_agent(
    messages=[ChatMessage(role="user", content="What's the weather in Paris?")],
    tools=tools,
    tool_executor=executor,
)
```

The executor dispatches by name and passes the LLM's input dict as keyword arguments. Exceptions propagate so `run_agent` can mark them as error results.

See [examples/12_fn_tools.py](examples/12_fn_tools.py) for a full example.

---

## Claude built-in tools — web search

`WEB_SEARCH_TOOL` is a pre-built `ToolDefinition` for Claude's native web search. The library auto-injects the required beta header.

```python
from reckonsys_llm_core import LLMClient, ChatMessage
from reckonsys_llm_core.strategies.claude import WEB_SEARCH_TOOL

response = client.query(
    messages=[ChatMessage(role="user", content="What is the latest version of Python?")],
    tools=[WEB_SEARCH_TOOL],
)
print(response.content)
```

See [examples/11_claude_web_search.py](examples/11_claude_web_search.py) for single-question and multi-turn patterns.

---

## Response fields

`LLMResponse` and `LLMStructuredResponse` both expose:

| Field | Type | Description |
|---|---|---|
| `content` | `str` / `BaseModel \| None` | Model output |
| `usage` | `TokenUsage` | Input, output, cache tokens |
| `model` | `str` | Model ID |
| `stop_reason` | `StopReason \| None` | `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`, `error` |
| `thinking` | `str \| None` | Extended thinking trace (if enabled) |
| `attempts` | `int` | Number of attempts made (async client with retry) |

`StreamDone` (final event in a stream) exposes the same fields as `LLMResponse` except `attempts`.

`BatchResult` exposes:

| Field | Type | Description |
|---|---|---|
| `custom_id` | `str` | Your correlation key |
| `response` | `LLMResponse \| None` | Set on success |
| `error` | `str \| None` | Set on failure / cancellation / expiry |
