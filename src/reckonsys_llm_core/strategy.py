from typing import Protocol

from reckonsys_llm_core.types import (
    LLMParams,
    LLMResponse,
    LLMStructuredParams,
    LLMStructuredResponse,
)


class LLMStrategy(Protocol):
    def send_query(self, params: LLMParams) -> LLMResponse: ...

    def send_structured_query(
        self, params: LLMStructuredParams
    ) -> LLMStructuredResponse: ...
