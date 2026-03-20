"""Internal shared utilities — not part of the public API."""
import json
import logging
from typing import Any

import pydantic
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def parse_json_response(
    raw: str,
    model: type[BaseModel],
) -> tuple[BaseModel | None, str | None]:
    """Parse a JSON string into a Pydantic model.

    Returns (content, error). When error is set, content is None.
    """
    try:
        return model.model_validate(json.loads(raw)), None
    except (pydantic.ValidationError, json.JSONDecodeError) as e:
        logger.warning("Structured response validation failed: %s", e)
        return None, str(e)


def validate_dict_response(
    data: Any,
    model: type[BaseModel],
) -> tuple[BaseModel | None, str | None]:
    """Validate a dict/object into a Pydantic model.

    Returns (content, error). When error is set, content is None.
    Used for tool-use responses where the provider already parsed the JSON.
    """
    try:
        return model.model_validate(data), None
    except pydantic.ValidationError as e:
        logger.warning("Structured response validation failed: %s", e)
        return None, str(e)
