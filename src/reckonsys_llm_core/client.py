from dataclasses import replace
from typing import AsyncIterator, Callable, Iterator

from pydantic import BaseModel

from reckonsys_llm_core.strategy import AsyncBatchLLMStrategy, AsyncLLMStrategy, BatchLLMStrategy, LLMStrategy
from reckonsys_llm_core.types import (
    Batch,
    BatchRequest,
    BatchResult,
    ChatMessage,
    LLMParams,
    LLMResponse,
    LLMStructuredParams,
    LLMStructuredResponse,
    RetryContext,
    StopReason,
    StreamEvent,
    ThinkingConfig,
)

_CORRECTION_TEMPLATE = """\
Your previous response could not be parsed.

Output received:
{raw_content}

Validation error:
{error}

Please respond again with valid JSON that exactly matches the required schema.\
"""


class LLMClient:
    """Synchronous client — for scripts, CLI tools, and eval pipelines."""

    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy

    def query(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> LLMResponse:
        params = LLMParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
        )
        return self.strategy.send_query(params)

    def query_structured(
        self,
        messages: list[ChatMessage],
        response_models: list[type[BaseModel]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> LLMStructuredResponse:
        params = LLMStructuredParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
            response_models=response_models,
        )
        return self.strategy.send_structured_query(params)


class AsyncLLMClient:
    """
    Asynchronous client with retry-on-failure for structured queries.

    Retry strategy: on validation failure, the LLM's raw output and the
    exact error are appended to the conversation so the model can self-correct.
    This is far more effective than blind retries — the model sees exactly
    what went wrong.

    Args:
        strategy:    An AsyncLLMStrategy implementation (Claude, Ollama, …).
        max_retries: Number of correction attempts after the first failure.
                     Total calls = max_retries + 1. Default: 2.
        on_retry:    Optional callback fired on each failed attempt. Use this
                     to add OTel span events, increment Prometheus counters, or
                     log to your eval dataset — without coupling the library to
                     any observability framework.

    Example::

        def _on_retry(ctx: RetryContext) -> None:
            span.add_event("llm.retry", {"attempt": ctx.attempt, "error": ctx.error})
            retry_counter.labels(model=strategy.model).inc()

        client = AsyncLLMClient(strategy, max_retries=2, on_retry=_on_retry)
    """

    def __init__(
        self,
        strategy: AsyncLLMStrategy,
        max_retries: int = 2,
        on_retry: Callable[[RetryContext], None] | None = None,
    ):
        self.strategy = strategy
        self.max_retries = max_retries
        self.on_retry = on_retry

    async def query(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> LLMResponse:
        params = LLMParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
        )
        return await self.strategy.send_query(params)

    def stream_query(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Returns an async iterator — use with `async for`.

        No retry: streaming is incompatible with retry since tokens are already
        forwarded to the caller before the response is complete.
        """
        params = LLMParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
        )
        return self.strategy.stream_query(params)

    async def query_structured(
        self,
        messages: list[ChatMessage],
        response_models: list[type[BaseModel]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> LLMStructuredResponse:
        """
        Structured query with automatic error-feedback retry.

        On each validation failure:
        1. The LLM's raw output is appended as an assistant message.
        2. A correction user message is appended with the exact error.
        3. The strategy is called again with this extended conversation.

        The original messages list is never mutated.
        """
        params = LLMStructuredParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
            response_models=response_models,
        )

        current_messages = list(params.messages)
        result: LLMStructuredResponse | None = None

        for attempt in range(1, self.max_retries + 2):  # +2 → first attempt + N retries
            current_params = replace(params, messages=current_messages)
            result = await self.strategy.send_structured_query(current_params)
            result.attempts = attempt

            if result.stop_reason != StopReason.ERROR:
                return result

            if attempt > self.max_retries:
                break

            if self.on_retry:
                self.on_retry(
                    RetryContext(
                        attempt=attempt,
                        error=result.error or "unknown validation error",
                        raw_content=result.raw_content,
                        params=current_params,
                    )
                )

            correction = _CORRECTION_TEMPLATE.format(
                raw_content=result.raw_content or "(empty)",
                error=result.error or "unknown validation error",
            )
            current_messages = current_messages + [
                ChatMessage(role="assistant", content=result.raw_content or ""),
                ChatMessage(role="user", content=correction),
            ]

        return result  # type: ignore[return-value]  # always set after first iteration


class BatchLLMClient:
    """
    Synchronous batch client — for scripts, CLI tools, and eval pipelines.

    The caller is responsible for persisting the batch_id returned by submit(),
    deciding when and how to poll via status(), and acting on results().

    Example::

        batch = client.submit(requests)
        # store batch.batch_id in your DB here

        # later, in a cron job or background task:
        b = client.status(batch_id)
        if b.status == BatchStatus.ENDED:
            for result in client.results(batch_id):
                if result.response:
                    process(result.custom_id, result.response)
                else:
                    log_error(result.custom_id, result.error)
    """

    def __init__(self, strategy: BatchLLMStrategy) -> None:
        self.strategy = strategy

    def submit(self, requests: list[BatchRequest]) -> Batch:
        return self.strategy.create_batch(requests)

    def status(self, batch_id: str) -> Batch:
        return self.strategy.get_batch(batch_id)

    def cancel(self, batch_id: str) -> Batch:
        return self.strategy.cancel_batch(batch_id)

    def results(self, batch_id: str) -> Iterator[BatchResult]:
        """Stream results one by one — avoids loading all results into memory."""
        return self.strategy.get_results(batch_id)


class AsyncBatchLLMClient:
    """
    Asynchronous batch client — for FastAPI and async pipelines.

    The caller is responsible for persisting the batch_id returned by submit(),
    deciding when and how to poll via status(), and acting on results().

    Example::

        batch = await client.submit(requests)
        # store batch.batch_id in your DB here

        # later, in a cron job or background task:
        b = await client.status(batch_id)
        if b.status == BatchStatus.ENDED:
            async for result in client.results(batch_id):
                if result.response:
                    process(result.custom_id, result.response)
                else:
                    log_error(result.custom_id, result.error)
    """

    def __init__(self, strategy: AsyncBatchLLMStrategy) -> None:
        self.strategy = strategy

    async def submit(self, requests: list[BatchRequest]) -> Batch:
        return await self.strategy.create_batch(requests)

    async def status(self, batch_id: str) -> Batch:
        return await self.strategy.get_batch(batch_id)

    async def cancel(self, batch_id: str) -> Batch:
        return await self.strategy.cancel_batch(batch_id)

    def results(self, batch_id: str) -> AsyncIterator[BatchResult]:
        """Stream results one by one — avoids loading all results into memory."""
        return self.strategy.get_results(batch_id)
