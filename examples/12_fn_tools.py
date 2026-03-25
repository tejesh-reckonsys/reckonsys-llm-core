"""
Function-to-tool helpers: tool_from_function and from_tools.

Instead of writing ToolDefinition + JSON schema by hand, derive everything
from the function's name, type annotations, defaults, and docstring.

    python examples/12_fn_tools.py

Requires: ANTHROPIC_API_KEY
"""

import asyncio
import json
from datetime import UTC, datetime
from typing import Literal

from reckonsys_llm_core import (
    AsyncLLMClient,
    ChatMessage,
    LLMClient,
    from_tools,
    tool_from_function,
)
from reckonsys_llm_core.strategies.claude import (
    AsyncClaudeLLMStrategy,
    ClaudeLLMStrategy,
    create_async_claude_client,
    create_claude_client,
)

# ---------------------------------------------------------------------------
# 1. Inspecting what tool_from_function produces
# ---------------------------------------------------------------------------


def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Return the current weather for a city.

    Args:
        city: The city name.
        unit: Temperature unit — celsius or fahrenheit.
    """
    return f"Sunny, 22°C in {city}"


td = tool_from_function(get_weather)
print("[tool_from_function]")
print(f"  name:        {td.name}")
print(f"  description: {td.description}")
print(f"  schema:      {json.dumps(td.input_schema, indent=4)}")
print()


# ---------------------------------------------------------------------------
# 2. from_tools — multiple functions, auto-generates tools + executor
# ---------------------------------------------------------------------------


def calculate(expression: str) -> str:
    """Evaluate a safe arithmetic expression.

    Args:
        expression: A Python arithmetic expression, e.g. '(3 + 4) * 2'.
    """
    import ast
    import operator as op

    _ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ops:
            return _ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ops:
            return _ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported: {ast.dump(node)}")

    return str(_eval(ast.parse(expression, mode="eval")))


def get_current_time() -> str:
    """Return the current UTC date and time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


tools, executor = from_tools(calculate, get_current_time)

print("[from_tools]")
for t in tools:
    print(f"  {t.name}: {t.description}")
print()


# ---------------------------------------------------------------------------
# 3. Using the ToolKit with run_agent (sync)
# ---------------------------------------------------------------------------


def run_sync() -> None:
    client = LLMClient(
        ClaudeLLMStrategy(client=create_claude_client(), model="claude-opus-4-6")
    )

    response = client.run_agent(
        messages=[
            ChatMessage(
                role="user",
                content="What is (123 * 456) + (789 / 3)? Also, what time is it right now?",
            )
        ],
        tools=tools,
        tool_executor=executor,
    )
    print("[sync run_agent]", response.content)


# ---------------------------------------------------------------------------
# 4. Using the ToolKit with arun_agent (async)
# ---------------------------------------------------------------------------


async def run_async() -> None:
    client = AsyncLLMClient(
        AsyncClaudeLLMStrategy(
            client=create_async_claude_client(), model="claude-opus-4-6"
        )
    )

    response = await client.arun_agent(
        messages=[
            ChatMessage(
                role="user",
                content="What is 2 ** 10? And what time is it now?",
            )
        ],
        tools=tools,
        tool_executor=executor,
    )
    print("[async arun_agent]", response.content)


if __name__ == "__main__":
    run_sync()
    asyncio.run(run_async())
