# API Changes & Library Gaps

> Last reviewed: March 2026

Tracks significant upstream API changes from Anthropic and OpenAI, and notes where the library needs to be updated to support them.

---

## Anthropic

### Breaking changes

| Change | Date | Library impact |
|---|---|---|
| `output_format` renamed to `output_config.format` | Feb 5, 2026 | ✅ Library already uses `output_config` — no action needed |
| `cache_control` must be on parent block, not nested inside `tool_result.content` or `document.source.content` | May 1, 2025 | ✅ Library constructs these correctly |
| `top_p` default changed from 0.999 → 0.99 across all models | May 22, 2025 | ✅ No impact — library doesn't set a default `top_p` |

### New features — not yet in library

| Feature | Date | Notes |
|---|---|---|
| `effort` parameter for thinking (replaces `budget_tokens` on Opus 4.6+) | Nov 2025 → GA Feb 2026 | ✅ `ThinkingConfig.effort` field added — takes priority over `budget_tokens` when set |
| `thinking.display: "omitted"` | Mar 2026 | Strip thinking content for faster streaming while preserving multi-turn signature |
| Automatic prompt caching (single `cache_control` on request body) | Feb 2026 | Library uses manual breakpoints only |
| Web search GA — beta header no longer required | Feb 17, 2026 | ✅ Beta header injection removed |
| Interleaved thinking (`interleaved-thinking-2025-05-14` beta) | May 2025 | Think between tool calls — not exposed |
| `model_context_window_exceeded` stop reason | Sep 2025 | ✅ Added to `STOP_REASON_MAP` → maps to `StopReason.MAX_TOKENS` |
| Web fetch tool (`web_fetch_20250305`) | Sep 2025 → GA Feb 2026 | ✅ `WEB_FETCH_TOOL` constant added to `claude.py` |
| Code execution tool v2 | Sep 2025 | Use via `raw_config` |
| Memory tool | Sep 2025 → GA Feb 2026 | ✅ `MEMORY_TOOL` constant added to `claude.py` |
| Tool search tool | Nov 2025 → GA Feb 2026 | Use via `raw_config` |
| MCP connector | May 2025 | Use via `raw_config` |
| `GET /v1/models` capability fields (`max_input_tokens`, `max_tokens`, `capabilities`) | Mar 2026 | Not used by library — informational |

### Models to add to docs / examples

- `claude-haiku-4-5-20251001` — current Haiku, replaces Haiku 3.5
- 1M context window for Opus 4.6 and Sonnet 4.6 is GA (no beta header)

### Upcoming deprecation

- **Claude Haiku 3** (`claude-3-haiku-20240307`) retires **April 19, 2026**

---

## OpenAI

### Breaking changes

| Change | Date | Library impact |
|---|---|---|
| `gpt-4o` API endpoint returning 404 | Feb 16, 2026 | Update any hardcoded `gpt-4o` references in examples |
| Assistants API deprecated | Aug 2025 | Sunset Aug 26, 2026 — not used by this library |

### New features — not yet in library

| Feature | Date | Notes |
|---|---|---|
| `reasoning_effort: "minimal"` | Aug 2025 (GPT-5) | ✅ Added to `ThinkingConfig.reasoning_effort` type |
| `reasoning_effort: "xhigh"` | Dec 2025 (GPT-5.2) | ✅ Added to `ThinkingConfig.reasoning_effort` type |
| Code interpreter tool (`"type": "code_interpreter"`) | May 2025 | ✅ `OPENAI_CODE_INTERPRETER_TOOL` constant added |
| File search tool (`"type": "file_search"`) | Mar 2025 | ✅ `OPENAI_FILE_SEARCH_TOOL` constant added |
| Tool search tool (`"type": "tool_search"`) | Mar 2026 | ✅ `OPENAI_TOOL_SEARCH_TOOL` constant added |
| Image generation tool (`"type": "image_generation"`) | Dec 2025 | Use via `raw_config` |
| MCP tool (`"type": "mcp"`) | May 2025 | Use via `raw_config` |
| `web_search_2025_08_26` tool type string | Aug 2025 | `OPENAI_WEB_SEARCH_TOOL` uses `web_search` — both work |
| Reusable prompts (`prompt` parameter with `id`, `version`, `variables`) | Jun 2025 | Pass via `_kwargs` override if needed |
| `phase` parameter (`"commentary"` / `"final_answer"`) | Feb 2026 | Not exposed |
| Conversations API (stateful multi-turn) | Aug 2025 | Not used — library is stateless by design |
| Background mode (`background: true`) | Mar 2025 | Async server-side execution — not exposed |
| Webhooks support | Jun 2025 | Not exposed |

### Models to update in docs / examples

- Recommended default: `gpt-5.4-mini` (replaces `gpt-4o`)
- Reasoning: `o4-mini` or `o3` (replaces earlier o-series)
- `gpt-4.1` family for cost-effective 1M context

---

## Workaround for missing features

Until the library is updated, any API feature not yet exposed can be used by passing `raw_config` on a `ToolDefinition` (for built-in tools), or by creating a client directly and calling the API yourself alongside the library's type system.

```python
# Example: use the Claude web fetch tool via raw_config
from reckonsys_llm_core import ToolDefinition

WEB_FETCH_TOOL = ToolDefinition(
    name="web_fetch",
    raw_config={"type": "web_fetch_20250305", "name": "web_fetch"},
)

# Example: use the OpenAI code interpreter via raw_config
CODE_INTERPRETER_TOOL = ToolDefinition(
    name="code_interpreter",
    raw_config={"type": "code_interpreter", "container": {"type": "auto"}},
)
```
