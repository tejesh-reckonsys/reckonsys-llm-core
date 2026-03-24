import inspect
from dataclasses import replace
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator

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
    TextContent,
    ThinkingConfig,
    ToolChoice,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
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
    """Synchronous client — for scripts, CLI tools, and eval pipelines.

    Args:
        strategy:    An LLMStrategy implementation (Claude, Ollama, …).
        max_retries: Number of correction attempts after the first structured
                     query failure. Total calls = max_retries + 1. Default: 2.
        on_retry:    Optional callback fired on each failed attempt.
    """

    def __init__(
        self,
        strategy: LLMStrategy,
        max_retries: int = 2,
        on_retry: Callable[[RetryContext], None] | None = None,
    ):
        self.strategy = strategy
        self.max_retries = max_retries
        self.on_retry = on_retry

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
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponse:
        params = LLMParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
            tools=tools or [],
            tool_choice=tool_choice,
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
        """Structured query with automatic error-feedback retry.

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
            result = self.strategy.send_structured_query(current_params)
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
    ) -> Iterator[StreamEvent]:
        """Returns a generator — use with a regular for loop.

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

    def run_agent(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        tool_executor: Callable[[str, dict[str, Any]], str],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
        max_iterations: int = 10,
    ) -> LLMResponse:
        """
        Run a tool-calling agent loop until the model produces a final text reply
        or max_iterations is reached.

        On each iteration:
        1. Call the model with the current messages and tool definitions.
        2. If stop_reason is tool_use: execute each requested tool, append the
           assistant message and tool results, and call the model again.
        3. If stop_reason is anything else: return the response.

        If tool_executor raises, the exception is captured and sent back to the
        model as an error result so it can recover gracefully.

        Args:
            messages:       Initial conversation messages.
            tools:          Tool definitions available to the model.
            tool_executor:  Called with (tool_name, tool_input) for each tool call.
                            Must return the result as a string.
            max_iterations: Safety limit on the number of model calls.
        """
        current_messages = list(messages)

        for _ in range(max_iterations):
            response = self.query(
                messages=current_messages,
                tools=tools,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                thinking=thinking,
            )

            if response.stop_reason != StopReason.TOOL_USE:
                return response

            # Reconstruct the full assistant message (text + tool_use blocks)
            assistant_content: list = []
            if response.content:
                assistant_content.append(TextContent(text=response.content))
            for tc in response.tool_calls:
                assistant_content.append(ToolUseContent(id=tc.id, name=tc.name, input=tc.input))
            current_messages.append(ChatMessage(role="assistant", content=assistant_content))

            # Execute tools and collect results
            tool_results: list[ToolResultContent] = []
            for tc in response.tool_calls:
                try:
                    result = tool_executor(tc.name, tc.input)
                    tool_results.append(ToolResultContent(tool_use_id=tc.id, content=result))
                except Exception as exc:
                    tool_results.append(
                        ToolResultContent(tool_use_id=tc.id, content=str(exc), is_error=True)
                    )
            current_messages.append(ChatMessage(role="user", content=tool_results))  # type: ignore[arg-type]

        raise RuntimeError(
            f"Agent did not reach a final response within {max_iterations} iterations"
        )


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
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> LLMResponse:
        params = LLMParams(
            messages=messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            thinking=thinking,
            tools=tools or [],
            tool_choice=tool_choice,
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

    async def arun_agent(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
        tool_executor: Callable[[str, dict[str, Any]], str | Awaitable[str]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        thinking: ThinkingConfig | None = None,
        max_iterations: int = 10,
    ) -> LLMResponse:
        """
        Async version of run_agent. tool_executor may be sync or async.

        See LLMClient.run_agent for full documentation.
        """
        current_messages = list(messages)

        for _ in range(max_iterations):
            response = await self.query(
                messages=current_messages,
                tools=tools,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                thinking=thinking,
            )

            if response.stop_reason != StopReason.TOOL_USE:
                return response

            assistant_content: list = []
            if response.content:
                assistant_content.append(TextContent(text=response.content))
            for tc in response.tool_calls:
                assistant_content.append(ToolUseContent(id=tc.id, name=tc.name, input=tc.input))
            current_messages.append(ChatMessage(role="assistant", content=assistant_content))

            tool_results: list[ToolResultContent] = []
            for tc in response.tool_calls:
                try:
                    maybe_coro = tool_executor(tc.name, tc.input)
                    result = await maybe_coro if inspect.isawaitable(maybe_coro) else maybe_coro
                    tool_results.append(ToolResultContent(tool_use_id=tc.id, content=str(result)))
                except Exception as exc:
                    tool_results.append(
                        ToolResultContent(tool_use_id=tc.id, content=str(exc), is_error=True)
                    )
            current_messages.append(ChatMessage(role="user", content=tool_results))  # type: ignore[arg-type]

        raise RuntimeError(
            f"Agent did not reach a final response within {max_iterations} iterations"
        )


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
