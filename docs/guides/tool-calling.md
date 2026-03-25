# Tool Calling & Agentic Loop

The library handles the full agentic loop via `run_agent` (sync) and `arun_agent` (async). You define tools and an executor; the library iterates until the model stops calling tools.

---

## Defining tools

```python
from reckonsys_llm_core import ToolDefinition

TOOLS = [
    ToolDefinition(
        name="get_weather",
        description="Return the current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name."},
            },
            "required": ["city"],
        },
    ),
]
```

---

## Running the agent (sync)

```python
def tool_executor(name: str, inputs: dict) -> str:
    if name == "get_weather":
        return f"Sunny, 22°C in {inputs['city']}"
    return f"Unknown tool: {name}"

response = client.run_agent(
    messages=[ChatMessage(role="user", content="What's the weather in Paris?")],
    tools=TOOLS,
    tool_executor=tool_executor,
    max_iterations=10,   # default: 10; prevents infinite loops
)
print(response.content)
```

---

## Running the agent (async)

`arun_agent` accepts both sync and async executor callables.

```python
async def async_executor(name: str, inputs: dict) -> str:
    # can await other coroutines here
    return f"Result for {name}"

response = await async_client.arun_agent(
    messages=[ChatMessage(role="user", content="What's the weather in Tokyo?")],
    tools=TOOLS,
    tool_executor=async_executor,
)
```

---

## Functions as tools

Derive `ToolDefinition` automatically from a function's name, type annotations, defaults, and docstring:

```python
from typing import Literal
from reckonsys_llm_core import tool_from_function, from_tools

def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Return the current weather for a city.

    Args:
        city: The city name.
        unit: Temperature unit.
    """
    return f"Sunny, 22°C in {city}"

# Single tool
tool_def = tool_from_function(get_weather)

# Multiple tools — returns ToolKit(tools, executor)
tools, executor = from_tools(get_weather, calculate, get_current_time)

response = client.run_agent(
    messages=[ChatMessage(role="user", content="What's the weather in Paris?")],
    tools=tools,
    tool_executor=executor,
)
```

The `executor` dispatches by name and passes the LLM's input dict as keyword arguments. Exceptions propagate as error tool results.

---

## Tool choice

Control how the model selects tools via `ToolChoice`:

```python
from reckonsys_llm_core import ToolChoice

# Default — model decides
tool_choice = ToolChoice(type="auto")

# Must call at least one tool
tool_choice = ToolChoice(type="any")

# Must call a specific tool
tool_choice = ToolChoice(type="tool", name="get_weather")

# No tool calls allowed
tool_choice = ToolChoice(type="none")

response = client.query(
    messages=[...],
    tools=TOOLS,
    tool_choice=tool_choice,
)
```

> **OpenAI note:** `type="any"` maps to OpenAI's `"required"` tool choice.

---

## Manual tool loop

If you need custom logic between tool calls, manage the loop yourself:

```python
from reckonsys_llm_core import ChatMessage, ToolUseContent, ToolResultContent

messages = [ChatMessage(role="user", content="What's the weather in Paris?")]

while True:
    response = client.query(messages=messages, tools=TOOLS)

    if response.stop_reason != StopReason.TOOL_USE:
        break

    # Append the assistant's tool call message
    messages.append(ChatMessage(
        role="assistant",
        content=[ToolUseContent(id=tc.id, name=tc.name, input=tc.input)
                 for tc in response.tool_calls],
    ))

    # Execute tools and append results
    results = [
        ToolResultContent(
            tool_use_id=tc.id,
            content=tool_executor(tc.name, tc.input),
        )
        for tc in response.tool_calls
    ]
    messages.append(ChatMessage(role="user", content=results))

print(response.content)
```

---

## Built-in provider tools

### Claude — web search

```python
from reckonsys_llm_core.strategies.claude import WEB_SEARCH_TOOL

response = client.query(
    messages=[ChatMessage(role="user", content="Latest Python release?")],
    tools=[WEB_SEARCH_TOOL],
)
```

### OpenAI — web search

```python
from reckonsys_llm_core.strategies.openai import OPENAI_WEB_SEARCH_TOOL

response = client.query(
    messages=[ChatMessage(role="user", content="Latest Python release?")],
    tools=[OPENAI_WEB_SEARCH_TOOL],
)
```

Do not mix provider-specific built-in tools with the wrong strategy — Claude's `WEB_SEARCH_TOOL` will error on the OpenAI strategy and vice versa.
