"""
Helpers for converting Python functions into ToolDefinitions.

    tool_from_function(fn)       → ToolDefinition
    from_tools(fn1, fn2, ...)    → ToolKit(tools, executor)

The function's name, type annotations, default values, and docstring are used
to build the ToolDefinition automatically — no manual JSON schema required.
"""

import inspect
import re
from collections.abc import Callable
from typing import Any, NamedTuple, get_type_hints

from pydantic import Field, create_model

from reckonsys_llm_core.types import ToolDefinition

_SECTION_HEADER = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Raises|Yields|Example|Examples|Note|Notes)\s*:",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """
    Extract (summary, {param_name: description}) from a docstring.

    Supports:
    - Google-style ``Args:`` / ``Parameters:`` sections
    - reStructuredText ``:param name: description`` directives
    """
    if not doc:
        return "", {}

    param_descriptions: dict[str, str] = {}

    # reST-style: :param name: description
    for m in re.finditer(r":param\s+(\w+):\s*(.+?)(?=\n|$)", doc):
        param_descriptions[m.group(1)] = m.group(2).strip()

    # Google-style: Args: / Parameters: block
    args_match = re.search(
        r"(?:Args|Arguments|Parameters)\s*:\s*\n((?:[ \t]+\S.*\n?)*)",
        doc,
        re.MULTILINE,
    )
    if args_match:
        block = args_match.group(1)
        # Each entry: "    name (optional type): description\n    continuation"
        for m in re.finditer(
            r"^[ \t]+(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+?)(?=\n[ \t]+\w|\Z)",
            block,
            re.MULTILINE | re.DOTALL,
        ):
            desc = " ".join(
                line.strip() for line in m.group(2).split("\n") if line.strip()
            )
            param_descriptions[m.group(1)] = desc

    summary_lines: list[str] = []
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped or _SECTION_HEADER.match(stripped) or stripped.startswith(":"):
            break
        summary_lines.append(stripped)

    return " ".join(summary_lines), param_descriptions


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------


def tool_from_function(fn: Callable[..., Any]) -> ToolDefinition:
    """
    Convert a Python function to a ToolDefinition.

    - Function name → tool name.
    - Docstring summary → tool description.
    - Type annotations + defaults → JSON schema (via Pydantic).
    - ``Args:`` / ``:param:`` docstring sections → per-field descriptions in the schema.
    - ``Annotated[T, Field(description="...")]`` is also respected.
    - ``*args`` and ``**kwargs`` parameters are ignored.

    Example::

        from typing import Literal

        def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
            \"\"\"Return the current weather for a city.

            Args:
                city: The city name.
                unit: Temperature unit.
            \"\"\"
            ...

        tool_def = tool_from_function(get_weather)
        # ToolDefinition(
        #     name="get_weather",
        #     description="Return the current weather for a city.",
        #     input_schema={
        #         "type": "object",
        #         "properties": {
        #             "city": {"type": "string", "description": "The city name."},
        #             "unit": {"enum": ["celsius", "fahrenheit"], "default": "celsius", ...},
        #         },
        #         "required": ["city"],
        #     },
        # )
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    doc = inspect.getdoc(fn) or ""
    description, param_descriptions = _parse_docstring(doc)

    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = hints.get(name, Any)
        param_desc = param_descriptions.get(name)

        if param.default is inspect.Parameter.empty:
            fields[name] = (
                annotation,
                Field(..., description=param_desc) if param_desc else ...,
            )
        else:
            fields[name] = (
                annotation,
                Field(param.default, description=param_desc)
                if param_desc
                else param.default,
            )

    DynamicModel = create_model(fn.__name__, **fields)
    schema = DynamicModel.model_json_schema()
    schema.pop("title", None)

    return ToolDefinition(
        name=fn.__name__,
        description=description,
        input_schema=schema,
    )


# ---------------------------------------------------------------------------
# Multi-function helper
# ---------------------------------------------------------------------------


class ToolKit(NamedTuple):
    """
    Returned by from_tools. Unpacks into (tools, executor) for run_agent / arun_agent.

        tools, executor = from_tools(get_weather, calculate)
        response = client.run_agent(messages, tools, executor)
    """

    tools: list[ToolDefinition]
    executor: Callable[[str, dict[str, Any]], str]


def from_tools(*fns: Callable[..., Any]) -> ToolKit:
    """
    Convert functions into a ToolKit (tools, executor) ready for run_agent.

    Each function is converted via tool_from_function(). The returned executor
    dispatches calls by name, passes the LLM's input dict as keyword arguments,
    and returns str(result). Exceptions from tool functions propagate so that
    run_agent / arun_agent can mark them as error results (is_error=True).

    Raises:
        ValueError: If two functions share the same name.

    Example::

        def get_weather(city: str) -> str:
            \"\"\"Return the weather for a city.\"\"\"
            return f"Sunny in {city}"

        def get_time() -> str:
            \"\"\"Return the current UTC time.\"\"\"
            from datetime import datetime, timezone
            return datetime.now(timezone.utc).isoformat()

        tools, executor = from_tools(get_weather, get_time)
        response = client.run_agent(messages, tools, executor)
    """
    fn_map: dict[str, Callable[..., Any]] = {}
    for fn in fns:
        if fn.__name__ in fn_map:
            raise ValueError(
                f"Duplicate tool name {fn.__name__!r}. "
                "Rename one of the functions or set fn.__name__ manually."
            )
        fn_map[fn.__name__] = fn

    tools = [tool_from_function(fn) for fn in fns]

    def executor(name: str, inputs: dict[str, Any]) -> str:
        if name not in fn_map:
            return f"Unknown tool: {name!r}"
        return str(fn_map[name](**inputs))

    return ToolKit(tools=tools, executor=executor)
