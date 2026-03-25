"""
Multi-turn / iterative conversation.

The library's clients are stateless — you own the message history.
Append each assistant reply back to the list and pass it all on the next call.

Two patterns are shown:
  A) Scripted back-and-forth (automated pipeline)
  B) Interactive REPL (type your own messages)

    python examples/09_multi_turn.py          # scripted
    python examples/09_multi_turn.py --repl   # interactive

Requires: ANTHROPIC_API_KEY
"""

import sys

from reckonsys_llm_core import ChatMessage, LLMClient
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

SYSTEM = "You are a helpful assistant. Be concise."

client = LLMClient(
    ClaudeLLMStrategy(client=create_claude_client(), model="claude-opus-4-6")
)


# ---------------------------------------------------------------------------
# Pattern A — scripted pipeline
# ---------------------------------------------------------------------------


def scripted() -> None:
    """
    Demonstrate a multi-step reasoning pipeline where each turn builds on the last.
    Useful for: decomposing a complex task, critique-and-refine loops, chain-of-thought.
    """
    messages: list[ChatMessage] = []

    turns = [
        "I want to build a REST API. What technology should I pick?",
        "Let's go with FastAPI. What's the minimal project structure I need?",
        "Can you show me a minimal main.py for that structure?",
    ]

    for user_text in turns:
        messages.append(ChatMessage(role="user", content=user_text))

        response = client.query(messages=messages, system=SYSTEM)

        # Important: append the assistant reply so the next turn has full context.
        messages.append(ChatMessage(role="assistant", content=response.content))

        print(f"User:      {user_text}")
        print(f"Assistant: {response.content.strip()[:300]}")
        print()


# ---------------------------------------------------------------------------
# Pattern B — interactive REPL
# ---------------------------------------------------------------------------


def repl() -> None:
    """Simple chat loop. Type 'exit' or Ctrl-C to quit."""
    messages: list[ChatMessage] = []
    print("Chat started (type 'exit' to quit)\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        messages.append(ChatMessage(role="user", content=user_text))
        response = client.query(messages=messages, system=SYSTEM)
        messages.append(ChatMessage(role="assistant", content=response.content))

        print(f"Assistant: {response.content.strip()}\n")


if __name__ == "__main__":
    if "--repl" in sys.argv:
        repl()
    else:
        scripted()
