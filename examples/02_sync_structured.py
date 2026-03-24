"""
Synchronous structured output — tool-use and strict (native JSON schema) modes.

    python examples/02_sync_structured.py

Requires: ANTHROPIC_API_KEY
"""

from pydantic import BaseModel

from reckonsys_llm_core import ChatMessage, LLMClient
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client


class Person(BaseModel):
    name: str
    age: int


class Sentiment(BaseModel):
    label: str         # positive / negative / neutral
    confidence: float  # 0.0 – 1.0


client = LLMClient(
    ClaudeLLMStrategy(client=create_claude_client(), model="claude-opus-4-6")
)

# Tool-use (default) — works with any provider
response = client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Alice is 30 years old.")],
    response_models=[Person],
)
print("[tool-use]", response.content)

# Strict / native JSON schema — single model uses output_config (no tools)
strict_client = LLMClient(
    ClaudeLLMStrategy(
        client=create_claude_client(),
        model="claude-sonnet-4-6",
        strict=True,
    )
)
response = strict_client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Bob is 42 years old.")],
    response_models=[Person],
)
print("[strict]", response.content)

# Multi-model — LLM picks the right tool (routing / classification)
response = client.query_structured(
    messages=[ChatMessage(role="user", content="I love this product!")],
    response_models=[Person, Sentiment],
    system="Classify the input using the most appropriate schema.",
)
print("[multi-model]", response.content)
