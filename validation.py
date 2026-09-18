"""Input validation: images + text fields.

Rules (per TOPIC.md):
- only JPEG/PNG
- size <= settings.max_file_size_bytes
- file must be a real image Pillow can open
- user_description must be non-empty (API/CLI enforce), capped at 2000 chars
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_TEXT_LEN = 2000


class ValidationError(ValueError):
    pass


def validate_image_file(path: str | Path, max_bytes: int) -> Path:
    p = Path(path)
    if not p.is_file():
        raise ValidationError(f"Image not found: {p}")
    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValidationError(
            f"Unsupported extension {p.suffix!r}. Only JPEG/PNG allowed."
        )
    size = p.stat().st_size
    if size == 0:
        raise ValidationError("Image file is empty.")
    if size > max_bytes:
        raise ValidationError(
            f"Image too large: {size} bytes > {max_bytes} bytes limit."
        )
    try:
        with Image.open(p) as img:
            img.verify()  # checks file integrity without decoding fully
        with Image.open(p) as img:
            if (img.format or "").upper() not in ALLOWED_FORMATS:
                raise ValidationError(
                    f"Invalid image format {img.format!r}. Only JPEG/PNG allowed."
                )
    except (UnidentifiedImageError, OSError) as e:
        raise ValidationError(f"File is not a valid JPEG/PNG image: {e}") from e
    return p


def validate_image_bytes(data: bytes, filename: str, max_bytes: int) -> str:
    """Validate raw upload bytes. Returns normalized suffix (.jpg/.png)."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValidationError(
            f"Unsupported extension {suffix!r}. Only JPEG/PNG allowed."
        )
    if not data:
        raise ValidationError("Image file is empty.")
    if len(data) > max_bytes:
        raise ValidationError(
            f"Image too large: {len(data)} bytes > {max_bytes} bytes limit."
        )
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        with Image.open(BytesIO(data)) as img:
            if (img.format or "").upper() not in ALLOWED_FORMATS:
                raise ValidationError(
                    f"Invalid image format {img.format!r}. Only JPEG/PNG allowed."
                )
            return ".jpg" if (img.format or "").upper() == "JPEG" else ".png"
    except (UnidentifiedImageError, OSError) as e:
        raise ValidationError(f"File is not a valid JPEG/PNG image: {e}") from e


def validate_user_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        raise ValidationError("user_description is required and must be non-empty.")
    if len(t) > MAX_TEXT_LEN:
        raise ValidationError(
            f"user_description too long ({len(t)} > {MAX_TEXT_LEN} chars)."
        )
    return t
