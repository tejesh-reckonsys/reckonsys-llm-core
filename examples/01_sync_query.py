"""
Synchronous plain text query.

    python examples/01_sync_query.py

Requires: ANTHROPIC_API_KEY
"""

from reckonsys_llm_core import ChatMessage, LLMClient, ThinkingConfig
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

strategy = ClaudeLLMStrategy(
    client=create_claude_client(),
    model="claude-opus-4-6",
)
client = LLMClient(strategy)

# Basic query
response = client.query(
    messages=[ChatMessage(role="user", content="What is the capital of France?")],
    system="Answer concisely.",
)
print("Answer:", response.content)
print("Tokens:", response.usage.total_tokens)
print("Stop reason:", response.stop_reason)

# With extended thinking
response = client.query(
    messages=[
        ChatMessage(
            role="user",
            content="A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        )
    ],
    thinking=ThinkingConfig(enabled=True, effort="medium"),
)
print("\n[Extended thinking]")
if response.thinking:
    print("<thinking>", response.thinking[:300], "...</thinking>")
print("Answer:", response.content)
