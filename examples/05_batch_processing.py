"""
Batch processing via Claude's Message Batches API.

Up to 50% cheaper and higher rate limits than standard requests.
Processes up to 100k requests per batch. Batches expire after 24 hours.

    python examples/05_batch_processing.py

Requires: ANTHROPIC_API_KEY

The submit step runs immediately. Polling/retrieval is shown as commented-out
code — in production you'd do that in a cron job or background task after
persisting the batch_id.
"""

import asyncio

from reckonsys_llm_core import (
    BatchRequest,
    ChatMessage,
    LLMParams,
    create_async_batch_llm,
    create_batch_llm,
)

REQUESTS = [
    BatchRequest(
        custom_id=f"item-{i}",
        params=LLMParams(
            messages=[
                ChatMessage(
                    role="user",
                    content=f"What is {i} squared? Answer with just the number.",
                )
            ],
        ),
    )
    for i in range(1, 6)
]


# ---------------------------------------------------------------------------
# Synchronous batch
# ---------------------------------------------------------------------------


def sync_batch() -> None:
    client = create_batch_llm("claude", "claude-opus-4-6")

    batch = client.submit(REQUESTS)
    print(f"[sync_batch] submitted  batch_id={batch.batch_id!r}")
    print(f"  status={batch.status}  counts={batch.counts}")

    # Persist batch_id to your DB here, then poll later:
    #
    # batch = client.status(batch_id)
    # if batch.status == BatchStatus.ENDED:
    #     for result in client.results(batch_id):
    #         if result.response:
    #             print(result.custom_id, result.response.content)
    #         else:
    #             print(result.custom_id, "ERROR:", result.error)


# ---------------------------------------------------------------------------
# Async batch
# ---------------------------------------------------------------------------


async def async_batch() -> None:
    client = create_async_batch_llm("claude", "claude-opus-4-6")

    batch = await client.submit(REQUESTS)
    print(f"[async_batch] submitted  batch_id={batch.batch_id!r}")
    print(f"  status={batch.status}  counts={batch.counts}")

    # Persist batch_id to your DB here, then poll later:
    #
    # batch = await client.status(batch_id)
    # if batch.status == BatchStatus.ENDED:
    #     async for result in client.results(batch_id):
    #         if result.response:
    #             print(result.custom_id, result.response.content)
    #         else:
    #             print(result.custom_id, "ERROR:", result.error)

    # Cancel a batch if needed:
    # batch = await client.cancel(batch_id)


if __name__ == "__main__":
    sync_batch()
    asyncio.run(async_batch())
