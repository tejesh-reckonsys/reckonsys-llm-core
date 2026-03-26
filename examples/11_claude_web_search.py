"""
Claude's built-in web search tool.

Pass WEB_SEARCH_TOOL in the tools list and Claude searches the web automatically.
Anthropic executes the search server-side — no tool_executor needed.

Web search is GA as of Feb 17, 2026 — no beta header required.

    python examples/11_claude_web_search.py

Requires: ANTHROPIC_API_KEY
          pip install "reckonsys-llm-core[claude]"
          An Anthropic plan that includes web search.
"""

import asyncio

from reckonsys_llm_core import ChatMessage, create_async_llm, create_llm
from reckonsys_llm_core.strategies.claude import WEB_SEARCH_TOOL

MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# Sync — single question
# ---------------------------------------------------------------------------

sync_client = create_llm("claude", MODEL)

response = sync_client.query(
    messages=[
        ChatMessage(role="user", content="What is the latest stable release of Python?")
    ],
    tools=[WEB_SEARCH_TOOL],
)
print("[web_search sync]", response.content)

# ---------------------------------------------------------------------------
# Async — multi-turn with search context preserved
# ---------------------------------------------------------------------------

async_client = create_async_llm("claude", MODEL)


async def multi_turn_search() -> None:
    messages: list[ChatMessage] = []

    questions = [
        "What are the top 3 most downloaded Python packages right now?",
        "Which of those three is the oldest, and when was it first released?",
    ]

    for question in questions:
        messages.append(ChatMessage(role="user", content=question))

        response = await async_client.query(
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
        )

        # Preserve full context (including search results) for the next turn
        messages.append(ChatMessage(role="assistant", content=response.content))

        print(f"Q: {question}")
        print(f"A: {response.content.strip()}\n")


asyncio.run(multi_turn_search())
