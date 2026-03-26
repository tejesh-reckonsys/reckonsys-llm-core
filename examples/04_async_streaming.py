"""
Async streaming — tokens are forwarded as they arrive.

    python examples/04_async_streaming.py

Requires: ANTHROPIC_API_KEY

Note: streaming is not retried on failure — tokens are already forwarded to the
caller before the response is complete.
"""

import asyncio

from reckonsys_llm_core import ChatMessage, StreamDone, StreamToken, create_async_llm

client = create_async_llm("claude", "claude-opus-4-6")


async def main() -> None:
    print("Streaming response: ", end="", flush=True)

    async for event in client.stream_query(
        messages=[
            ChatMessage(role="user", content="Count from 1 to 5, one number per line.")
        ],
    ):
        if isinstance(event, StreamToken):
            print(event.token, end="", flush=True)
        elif isinstance(event, StreamDone):
            print()  # newline after stream ends
            print(f"stop_reason={event.stop_reason}")
            print(
                f"tokens used: input={event.usage.input_tokens} output={event.usage.output_tokens}"
            )


asyncio.run(main())
