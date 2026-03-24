"""
AWS Bedrock — sync and async.

    python examples/07_bedrock.py

Requires: AWS_IAM_ACCESS_KEY and AWS_IAM_SECRET_KEY env vars
Install:  pip install "reckonsys-llm-core[claude]"
"""

import asyncio

from reckonsys_llm_core import AsyncLLMClient, ChatMessage, LLMClient
from reckonsys_llm_core.strategies.claude import (
    AsyncClaudeLLMStrategy,
    ClaudeLLMStrategy,
    create_async_bedrock_client,
    create_bedrock_client,
)

MODEL = "anthropic.claude-opus-4-6-20251101-v1:0"

# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

sync_client = LLMClient(
    ClaudeLLMStrategy(
        client=create_bedrock_client(region="us-west-2"),
        model=MODEL,
    )
)

response = sync_client.query(
    messages=[ChatMessage(role="user", content="What is the capital of France?")],
    system="Answer concisely.",
)
print("[bedrock_sync]", response.content)


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

async_client = AsyncLLMClient(
    AsyncClaudeLLMStrategy(
        client=create_async_bedrock_client(region="us-west-2"),
        model=MODEL,
    )
)


async def main() -> None:
    response = await async_client.query(
        messages=[ChatMessage(role="user", content="What is the capital of Germany?")],
        system="Answer concisely.",
    )
    print("[bedrock_async]", response.content)


asyncio.run(main())
