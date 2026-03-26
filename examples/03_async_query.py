"""
Async plain text query and structured output with retry.

    python examples/03_async_query.py

Requires: ANTHROPIC_API_KEY
"""

import asyncio

from pydantic import BaseModel

from reckonsys_llm_core import ChatMessage, RetryContext, create_async_llm


class Person(BaseModel):
    name: str
    age: int


client = create_async_llm("claude", "claude-opus-4-6")


async def plain_query() -> None:
    response = await client.query(
        messages=[ChatMessage(role="user", content="What is the capital of Germany?")],
        system="Answer concisely.",
    )
    print("[async_query]", response.content)


async def structured_with_retry() -> None:
    def on_retry(ctx: RetryContext) -> None:
        # Hook this up to OTel, Prometheus, or your eval dataset
        print(f"  [retry] attempt={ctx.attempt} error={ctx.error[:80]}")

    retry_client = create_async_llm(
        "claude", "claude-opus-4-6", max_retries=2, on_retry=on_retry
    )
    response = await retry_client.query_structured(
        messages=[ChatMessage(role="user", content="Extract: Carol is 25 years old.")],
        response_models=[Person],
    )
    print("[async_structured]", response.content, f"(attempts={response.attempts})")


async def concurrent_queries() -> None:
    questions = [
        "What is 2 + 2?",
        "What is the capital of Japan?",
        "Name one planet in our solar system.",
    ]
    responses = await asyncio.gather(
        *[
            client.query(messages=[ChatMessage(role="user", content=q)])
            for q in questions
        ]
    )
    print("[concurrent_queries]")
    for q, r in zip(questions, responses):
        print(f"  {q!r}  →  {r.content.strip()!r}")


async def main() -> None:
    await plain_query()
    await structured_with_retry()
    await concurrent_queries()


asyncio.run(main())
