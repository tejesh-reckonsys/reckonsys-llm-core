"""
Ollama — sync and async, plain text and structured output.

    python examples/06_ollama.py

Requires: a running Ollama server (default: http://localhost:11434)
    ollama pull llama3.2

Install: pip install "reckonsys-llm-core[ollama]"
"""

import asyncio

from pydantic import BaseModel

from reckonsys_llm_core import AsyncLLMClient, ChatMessage, LLMClient
from reckonsys_llm_core.strategies.ollama import (
    AsyncOllamaLLMStrategy,
    OllamaLLMStrategy,
)


class Person(BaseModel):
    name: str
    age: int


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

sync_client = LLMClient(OllamaLLMStrategy(model="llama3.2"))

response = sync_client.query(
    messages=[ChatMessage(role="user", content="Hello!")],
    system="You are a helpful assistant. Be concise.",
)
print("[ollama_sync]", response.content)

response = sync_client.query_structured(
    messages=[ChatMessage(role="user", content="Extract: Alice is 30 years old.")],
    response_models=[Person],
)
print("[ollama_sync_structured]", response.content)

# Custom host
# client = LLMClient(OllamaLLMStrategy(model="llama3.2", host="http://my-server:11434"))

# Extended thinking (deepseek-r1, qwen3, etc.)
# response = sync_client.query(
#     messages=[ChatMessage(role="user", content="Solve this step by step...")],
#     thinking=ThinkingConfig(enabled=True),
# )
# print(response.thinking)
# print(response.content)


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

async_client = AsyncLLMClient(AsyncOllamaLLMStrategy(model="llama3.2"))


async def main() -> None:
    response = await async_client.query(
        messages=[ChatMessage(role="user", content="Hello from async!")],
        system="You are a helpful assistant. Be concise.",
    )
    print("[ollama_async]", response.content)

    response = await async_client.query_structured(
        messages=[ChatMessage(role="user", content="Extract: Bob is 42 years old.")],
        response_models=[Person],
    )
    print("[ollama_async_structured]", response.content)


asyncio.run(main())
