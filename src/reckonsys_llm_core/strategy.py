from typing import AsyncIterator, Iterator, Protocol

from reckonsys_llm_core.types import (
    Batch,
    BatchRequest,
    BatchResult,
    LLMParams,
    LLMResponse,
    LLMStructuredParams,
    LLMStructuredResponse,
    StreamEvent,
)


class LLMStrategy(Protocol):
    """Synchronous LLM strategy — for scripts, CLI tools, and eval pipelines."""

    def send_query(self, params: LLMParams) -> LLMResponse: ...

    def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse: ...

    def stream_query(self, params: LLMParams) -> Iterator[StreamEvent]: ...


class AsyncLLMStrategy(Protocol):
    """
    Asynchronous LLM strategy — for FastAPI, async pipelines, and streaming.

    stream_query is a regular method (not async) that returns an AsyncIterator.
    Call it directly and iterate with `async for`.
    """

    async def send_query(self, params: LLMParams) -> LLMResponse: ...

    async def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse: ...

    def stream_query(self, params: LLMParams) -> AsyncIterator[StreamEvent]: ...


class BatchLLMStrategy(Protocol):
    """Synchronous batch strategy — for scripts and eval pipelines.

    The underlying batch processing is always async on the provider's side.
    This is just the Python calling convention (no async/await required).
    """

    def create_batch(self, requests: list[BatchRequest]) -> Batch: ...
    def get_batch(self, batch_id: str) -> Batch: ...
    def cancel_batch(self, batch_id: str) -> Batch: ...
    def get_results(self, batch_id: str) -> Iterator[BatchResult]: ...


class AsyncBatchLLMStrategy(Protocol):
    """
    Strategy for providers that support native batch APIs.

    Not all providers implement this — Ollama has no batch API.
    The caller is responsible for storing batch_id, polling, and acting on results.

    get_results is a regular method (not async) that returns an AsyncIterator.
    """

    async def create_batch(self, requests: list[BatchRequest]) -> Batch: ...
    async def get_batch(self, batch_id: str) -> Batch: ...
    async def cancel_batch(self, batch_id: str) -> Batch: ...
    def get_results(self, batch_id: str) -> AsyncIterator[BatchResult]: ...
