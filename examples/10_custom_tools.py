"""
Custom tool calling — agentic loop using the library.

LLMClient.run_agent / AsyncLLMClient.arun_agent handle the full agentic loop:
  1. Send messages + tools to the model.
  2. On tool_use: execute tools, append results, repeat.
  3. On end_turn (or any non-tool stop): return the final LLMResponse.

Define tools with ToolDefinition (name, description, JSON schema input_schema).
Implement a tool_executor callable: (tool_name, tool_input_dict) -> str.

    python examples/10_custom_tools.py

Requires: ANTHROPIC_API_KEY
"""

import ast
import asyncio
import operator
from datetime import UTC, datetime
from typing import Any

from reckonsys_llm_core import (
    ChatMessage,
    ToolDefinition,
    create_async_llm,
    create_llm,
)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    ToolDefinition(
        name="calculate",
        description=(
            "Evaluate a safe arithmetic expression and return the numeric result. "
            "Supports +, -, *, /, ** and parentheses."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python arithmetic expression, e.g. '(3 + 4) * 2'",
                },
            },
            "required": ["expression"],
        },
    ),
    ToolDefinition(
        name="get_current_time",
        description="Return the current UTC date and time as an ISO-8601 string.",
        input_schema={"type": "object", "properties": {}},
    ),
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    return _eval(ast.parse(expr, mode="eval"))


def tool_executor(name: str, inputs: dict[str, Any]) -> str:
    if name == "calculate":
        try:
            return str(_safe_eval(inputs["expression"]))
        except Exception as e:
            return f"Error: {e}"
    if name == "get_current_time":
        return datetime.now(UTC).isoformat()
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Sync agent
# ---------------------------------------------------------------------------


def run_sync() -> None:
    client = create_llm("claude", "claude-opus-4-6")

    response = client.run_agent(
        messages=[
            ChatMessage(
                role="user",
                content="What is (123 * 456) + (789 / 3)? Also, what time is it right now?",
            )
        ],
        tools=TOOLS,
        tool_executor=tool_executor,
    )
    print("[sync agent]", response.content)


# ---------------------------------------------------------------------------
# Async agent
# ---------------------------------------------------------------------------


async def run_async() -> None:
    client = create_async_llm("claude", "claude-opus-4-6")

    response = await client.arun_agent(
        messages=[
            ChatMessage(
                role="user",
                content="If I invest $10,000 at 7% annual return, what will it be worth after 10 years?",
            )
        ],
        tools=TOOLS,
        tool_executor=tool_executor,  # sync executor works fine with arun_agent
    )
    print("[async agent]", response.content)


if __name__ == "__main__":
    run_sync()
    asyncio.run(run_async())
