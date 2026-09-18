"""High-level VLM call: describe a lost/found item from an image."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ai.providers.base import VLMProvider, ProviderError
from ai.providers.factory import get_vlm
from ai.schemas import ITEM_DESCRIPTION_SCHEMA, ItemDescription


_PROMPT = """You are an assistant cataloguing items for a lost-and-found service.

Look at the photo and the user's description, and produce a structured JSON
description that will be used for matching against other items.

User description:
{user_text}

Be precise and concise. List concrete colours rather than vague terms ("navy
blue" not just "dark"). If you cannot identify a brand confidently, set
"brand" to null. "confidence" reflects how sure you are this is the kind of
object you say it is, on a 0..1 scale.
"""


def describe_item(
    image_path: str,
    user_text: str,
    *,
    vlm: VLMProvider | None = None,
) -> ItemDescription:
    """Use a VLM to describe an item.

    Parameters
    ----------
    image_path : str
        Path to a JPEG or PNG file.
    user_text : str
        The user's free-text description supplied at registration time.
    vlm : VLMProvider | None
        Override the provider. Pass an instance for testing or to explicitly
        pin a provider; otherwise the env-var-selected default is used.

    Raises
    ------
    ProviderError
        If the model errors or returns an unparseable / schema-invalid response.
    """
    vlm = vlm or get_vlm()
    prompt = _PROMPT.format(user_text=user_text or "(none provided)")
    raw = vlm.describe(image_path, prompt, json_schema=ITEM_DESCRIPTION_SCHEMA)
    payload = _parse_json(raw)
    try:
        return ItemDescription.model_validate(payload)
    except ValidationError as e:
        raise ProviderError(f"VLM response failed schema validation: {e}") from e


def _parse_json(raw: str) -> dict[str, Any]:
    """Forgiving JSON parser: handles models that wrap output in fences."""
    s = raw.strip()
    # Strip common code-fence wrappers.
    if s.startswith("```"):
        # remove first line (``` or ```json) and last line (```)
        lines = s.splitlines()
        if lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        lines = lines[1:]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise ProviderError(f"Could not parse JSON from VLM: {e}\nRaw: {raw[:300]!r}")
    if not isinstance(obj, dict):
        raise ProviderError(f"Expected JSON object, got {type(obj).__name__}")
    return obj
