"""
One-call helpers for creating LLM and batch clients.

Instead of wiring up a strategy and wrapping it in a client manually, use these
helpers to get a ready-to-use client in a single call::

    from reckonsys_llm_core import create_llm
    client = create_llm("claude", "claude-sonnet-4-6")

Provider-specific parameter references
---------------------------------------
- **claude** / **claude_bedrock** — docs/providers/claude.md
- **openai**                      — docs/providers/openai.md
- **ollama**                      — docs/providers/ollama.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from reckonsys_llm_core.client import (
    AsyncBatchLLMClient,
    AsyncLLMClient,
    BatchLLMClient,
    LLMClient,
)
from reckonsys_llm_core.strategy import (
    AsyncBatchLLMStrategy,
    AsyncLLMStrategy,
    BatchLLMStrategy,
    LLMStrategy,
)
from reckonsys_llm_core.types import RetryContext

Provider = Literal["claude", "claude_bedrock", "openai", "ollama"]
BatchProvider = Literal["claude", "claude_bedrock", "openai"]


# ---------------------------------------------------------------------------
# Sync LLM
# ---------------------------------------------------------------------------


def create_llm(
    provider: Provider,
    model: str,
    *,
    # --- credentials / connection ---
    api_key: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-west-2",
    host: str = "http://localhost:11434",
    # --- generation defaults ---
    default_max_tokens: int | None = None,
    strict: bool = False,
    # --- retry ---
    max_retries: int = 2,
    on_retry: Callable[[RetryContext], None] | None = None,
) -> LLMClient:
    """Create a synchronous :class:`LLMClient` for the given provider.

    Args:
        provider: One of ``"claude"``, ``"claude_bedrock"``, ``"openai"``,
            or ``"ollama"``.
        model: The model identifier to use (e.g. ``"claude-sonnet-4-6"``).

        api_key: API key for Claude or OpenAI.  Reads from the environment
            (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``) when ``None``.
            *claude, openai only*

        access_key: AWS IAM access key ID.  Reads ``AWS_IAM_ACCESS_KEY`` when
            ``None``. *claude_bedrock only*
        secret_key: AWS IAM secret access key.  Reads ``AWS_IAM_SECRET_KEY``
            when ``None``. *claude_bedrock only*
        region: AWS region (default ``"us-west-2"``). *claude_bedrock only*

        host: Ollama server URL (default ``"http://localhost:11434"``).
            *ollama only*

        default_max_tokens: Token limit used when ``max_tokens`` is not
            specified on a per-request basis.  Provider defaults:
            ``8000`` (claude / ollama), ``4096`` (openai).

        strict: Enable strict JSON-schema mode for structured output.
            *claude, openai only* — see provider docs for model support.

        max_retries: Number of correction attempts after the first structured
            query failure.  Total calls = ``max_retries + 1``.
        on_retry: Optional callback fired on each failed structured-query
            attempt.  Useful for logging, metrics, or eval datasets.

    Returns:
        A configured :class:`LLMClient`.

    Provider docs:
        - Claude / Bedrock — ``docs/providers/claude.md``
        - OpenAI           — ``docs/providers/openai.md``
        - Ollama           — ``docs/providers/ollama.md``

    Example::

        from reckonsys_llm_core import create_llm, ChatMessage

        client = create_llm("openai", "gpt-5.4-mini")
        response = client.query(
            messages=[ChatMessage(role="user", content="Hello!")],
        )
        print(response.content)
    """
    strategy = _make_sync_llm_strategy(
        provider,
        model,
        api_key,
        access_key,
        secret_key,
        region,
        host,
        default_max_tokens,
        strict,
    )
    return LLMClient(strategy, max_retries=max_retries, on_retry=on_retry)


# ---------------------------------------------------------------------------
# Async LLM
# ---------------------------------------------------------------------------


def create_async_llm(
    provider: Provider,
    model: str,
    *,
    api_key: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-west-2",
    host: str = "http://localhost:11434",
    default_max_tokens: int | None = None,
    strict: bool = False,
    max_retries: int = 2,
    on_retry: Callable[[RetryContext], None] | None = None,
) -> AsyncLLMClient:
    """Create an asynchronous :class:`AsyncLLMClient` for the given provider.

    Args:
        provider: One of ``"claude"``, ``"claude_bedrock"``, ``"openai"``,
            or ``"ollama"``.
        model: The model identifier to use.

        api_key: API key for Claude or OpenAI.  Reads from the environment
            when ``None``. *claude, openai only*

        access_key: AWS IAM access key ID. *claude_bedrock only*
        secret_key: AWS IAM secret access key. *claude_bedrock only*
        region: AWS region (default ``"us-west-2"``). *claude_bedrock only*

        host: Ollama server URL (default ``"http://localhost:11434"``).
            *ollama only*

        default_max_tokens: Token limit used when ``max_tokens`` is not
            specified per request.  Provider defaults: ``8000`` (claude /
            ollama), ``4096`` (openai).

        strict: Enable strict JSON-schema mode for structured output.
            *claude, openai only*

        max_retries: Correction attempts after the first structured query
            failure.
        on_retry: Optional callback fired on each failed attempt.

    Returns:
        A configured :class:`AsyncLLMClient`.

    Provider docs:
        - Claude / Bedrock — ``docs/providers/claude.md``
        - OpenAI           — ``docs/providers/openai.md``
        - Ollama           — ``docs/providers/ollama.md``

    Example::

        import asyncio
        from reckonsys_llm_core import create_async_llm, ChatMessage

        async def main():
            client = create_async_llm("claude", "claude-sonnet-4-6")
            response = await client.query(
                messages=[ChatMessage(role="user", content="Hello!")],
            )
            print(response.content)

        asyncio.run(main())
    """
    strategy = _make_async_llm_strategy(
        provider,
        model,
        api_key,
        access_key,
        secret_key,
        region,
        host,
        default_max_tokens,
        strict,
    )
    return AsyncLLMClient(strategy, max_retries=max_retries, on_retry=on_retry)


# ---------------------------------------------------------------------------
# Sync batch
# ---------------------------------------------------------------------------


def create_batch_llm(
    provider: BatchProvider,
    model: str,
    *,
    api_key: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-west-2",
    default_max_tokens: int | None = None,
) -> BatchLLMClient:
    """Create a synchronous :class:`BatchLLMClient` for the given provider.

    Ollama has no batch API and is not supported here.

    Args:
        provider: One of ``"claude"``, ``"claude_bedrock"``, or ``"openai"``.
        model: The model identifier to use.

        api_key: API key for Claude or OpenAI.  Reads from the environment
            when ``None``. *claude, openai only*

        access_key: AWS IAM access key ID. *claude_bedrock only*
        secret_key: AWS IAM secret access key. *claude_bedrock only*
        region: AWS region (default ``"us-west-2"``). *claude_bedrock only*

        default_max_tokens: Token limit used for each batch request.
            Provider defaults: ``8000`` (claude), ``4096`` (openai).

    Returns:
        A configured :class:`BatchLLMClient`.

    Provider docs:
        - Claude / Bedrock — ``docs/providers/claude.md``
        - OpenAI           — ``docs/providers/openai.md``
        - Batch guide      — ``docs/guides/batch-processing.md``

    Example::

        from reckonsys_llm_core import create_batch_llm, BatchRequest, LLMParams, ChatMessage

        client = create_batch_llm("openai", "gpt-5.4-mini")
        batch = client.submit([
            BatchRequest("req-1", LLMParams(
                messages=[ChatMessage(role="user", content="Hello")],
            )),
        ])
        print(batch.batch_id)
    """
    strategy = _make_sync_batch_strategy(
        provider,
        model,
        api_key,
        access_key,
        secret_key,
        region,
        default_max_tokens,
    )
    return BatchLLMClient(strategy)


# ---------------------------------------------------------------------------
# Async batch
# ---------------------------------------------------------------------------


def create_async_batch_llm(
    provider: BatchProvider,
    model: str,
    *,
    api_key: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-west-2",
    default_max_tokens: int | None = None,
) -> AsyncBatchLLMClient:
    """Create an asynchronous :class:`AsyncBatchLLMClient` for the given provider.

    Ollama has no batch API and is not supported here.

    Args:
        provider: One of ``"claude"``, ``"claude_bedrock"``, or ``"openai"``.
        model: The model identifier to use.

        api_key: API key for Claude or OpenAI.  Reads from the environment
            when ``None``. *claude, openai only*

        access_key: AWS IAM access key ID. *claude_bedrock only*
        secret_key: AWS IAM secret access key. *claude_bedrock only*
        region: AWS region (default ``"us-west-2"``). *claude_bedrock only*

        default_max_tokens: Token limit used for each batch request.
            Provider defaults: ``8000`` (claude), ``4096`` (openai).

    Returns:
        A configured :class:`AsyncBatchLLMClient`.

    Provider docs:
        - Claude / Bedrock — ``docs/providers/claude.md``
        - OpenAI           — ``docs/providers/openai.md``
        - Batch guide      — ``docs/guides/batch-processing.md``

    Example::

        import asyncio
        from reckonsys_llm_core import create_async_batch_llm, BatchRequest, LLMParams, ChatMessage

        async def main():
            client = create_async_batch_llm("claude", "claude-sonnet-4-6")
            batch = await client.submit([
                BatchRequest("req-1", LLMParams(
                    messages=[ChatMessage(role="user", content="Hello")],
                )),
            ])
            print(batch.batch_id)

        asyncio.run(main())
    """
    strategy = _make_async_batch_strategy(
        provider,
        model,
        api_key,
        access_key,
        secret_key,
        region,
        default_max_tokens,
    )
    return AsyncBatchLLMClient(strategy)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _make_sync_llm_strategy(
    provider: str,
    model: str,
    api_key: str | None,
    access_key: str | None,
    secret_key: str | None,
    region: str,
    host: str,
    default_max_tokens: int | None,
    strict: bool,
) -> LLMStrategy:
    if provider == "claude":
        from reckonsys_llm_core.strategies.claude import (
            ClaudeLLMStrategy,
            create_claude_client,
        )

        kw: dict = {} if api_key is None else {"api_key": api_key}
        client = create_claude_client(**kw)
        return ClaudeLLMStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
            strict=strict,
        )

    if provider == "claude_bedrock":
        from reckonsys_llm_core.strategies.claude import (
            ClaudeLLMStrategy,
            create_bedrock_client,
        )

        client = create_bedrock_client(
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )
        return ClaudeLLMStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
            strict=strict,
        )

    if provider == "openai":
        from reckonsys_llm_core.strategies.openai import (
            OpenAILLMStrategy,
            create_openai_client,
        )

        kw = {} if api_key is None else {"api_key": api_key}
        client = create_openai_client(**kw)
        return OpenAILLMStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
            strict=strict,
        )

    if provider == "ollama":
        from reckonsys_llm_core.strategies.ollama import OllamaLLMStrategy

        return OllamaLLMStrategy(
            model,
            host=host,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
        )

    raise ValueError(
        f"Unknown provider {provider!r}. "
        "Choose from 'claude', 'claude_bedrock', 'openai', 'ollama'."
    )


def _make_async_llm_strategy(
    provider: str,
    model: str,
    api_key: str | None,
    access_key: str | None,
    secret_key: str | None,
    region: str,
    host: str,
    default_max_tokens: int | None,
    strict: bool,
) -> AsyncLLMStrategy:
    if provider == "claude":
        from reckonsys_llm_core.strategies.claude import (
            AsyncClaudeLLMStrategy,
            create_async_claude_client,
        )

        kw: dict = {} if api_key is None else {"api_key": api_key}
        client = create_async_claude_client(**kw)
        return AsyncClaudeLLMStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
            strict=strict,
        )

    if provider == "claude_bedrock":
        from reckonsys_llm_core.strategies.claude import (
            AsyncClaudeLLMStrategy,
            create_async_bedrock_client,
        )

        client = create_async_bedrock_client(
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )
        return AsyncClaudeLLMStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
            strict=strict,
        )

    if provider == "openai":
        from reckonsys_llm_core.strategies.openai import (
            AsyncOpenAILLMStrategy,
            create_async_openai_client,
        )

        kw = {} if api_key is None else {"api_key": api_key}
        client = create_async_openai_client(**kw)
        return AsyncOpenAILLMStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
            strict=strict,
        )

    if provider == "ollama":
        from reckonsys_llm_core.strategies.ollama import AsyncOllamaLLMStrategy

        return AsyncOllamaLLMStrategy(
            model,
            host=host,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
        )

    raise ValueError(
        f"Unknown provider {provider!r}. "
        "Choose from 'claude', 'claude_bedrock', 'openai', 'ollama'."
    )


def _make_sync_batch_strategy(
    provider: str,
    model: str,
    api_key: str | None,
    access_key: str | None,
    secret_key: str | None,
    region: str,
    default_max_tokens: int | None,
) -> BatchLLMStrategy:
    if provider == "claude":
        from reckonsys_llm_core.strategies.claude import (
            ClaudeBatchStrategy,
            create_claude_client,
        )

        kw: dict = {} if api_key is None else {"api_key": api_key}
        client = create_claude_client(**kw)
        return ClaudeBatchStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
        )

    if provider == "claude_bedrock":
        from reckonsys_llm_core.strategies.claude import (
            ClaudeBatchStrategy,
            create_bedrock_client,
        )

        client = create_bedrock_client(
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )
        return ClaudeBatchStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
        )

    if provider == "openai":
        from reckonsys_llm_core.strategies.openai import (
            OpenAIBatchStrategy,
            create_openai_client,
        )

        kw = {} if api_key is None else {"api_key": api_key}
        client = create_openai_client(**kw)
        return OpenAIBatchStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
        )

    raise ValueError(
        f"Unknown batch provider {provider!r}. "
        "Choose from 'claude', 'claude_bedrock', 'openai'. "
        "Ollama does not support batch processing."
    )


def _make_async_batch_strategy(
    provider: str,
    model: str,
    api_key: str | None,
    access_key: str | None,
    secret_key: str | None,
    region: str,
    default_max_tokens: int | None,
) -> AsyncBatchLLMStrategy:
    if provider == "claude":
        from reckonsys_llm_core.strategies.claude import (
            AsyncClaudeBatchStrategy,
            create_async_claude_client,
        )

        kw: dict = {} if api_key is None else {"api_key": api_key}
        client = create_async_claude_client(**kw)
        return AsyncClaudeBatchStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
        )

    if provider == "claude_bedrock":
        from reckonsys_llm_core.strategies.claude import (
            AsyncClaudeBatchStrategy,
            create_async_bedrock_client,
        )

        client = create_async_bedrock_client(
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        )
        return AsyncClaudeBatchStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
        )

    if provider == "openai":
        from reckonsys_llm_core.strategies.openai import (
            AsyncOpenAIBatchStrategy,
            create_async_openai_client,
        )

        kw = {} if api_key is None else {"api_key": api_key}
        client = create_async_openai_client(**kw)
        return AsyncOpenAIBatchStrategy(
            client,
            model,
            **(
                {}
                if default_max_tokens is None
                else {"default_max_tokens": default_max_tokens}
            ),
        )

    raise ValueError(
        f"Unknown batch provider {provider!r}. "
        "Choose from 'claude', 'claude_bedrock', 'openai'. "
        "Ollama does not support batch processing."
    )
