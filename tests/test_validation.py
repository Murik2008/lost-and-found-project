"""Tests for src.validation (your module)."""

from pathlib import Path

import pytest

from validation import (
    ValidationError,
    validate_image_bytes,
    validate_image_file,
    validate_user_text,
)

DATA_LOST = Path(__file__).parent.parent / "data" / "lost"
MAX_BYTES = 5 * 1024 * 1024


def test_validate_user_text_ok_strips():
    assert validate_user_text("  black umbrella  ") == "black umbrella"


def test_validate_user_text_empty_raises():
    with pytest.raises(ValidationError):
        validate_user_text("   ")
    with pytest.raises(ValidationError):
        validate_user_text("")


def test_validate_user_text_too_long_raises():
    with pytest.raises(ValidationError):
        validate_user_text("x" * 2001)


def test_validate_image_file_real_png():
    # Uses repo sample data — must be real JPEG/PNG.
    candidates = list(DATA_LOST.glob("*.png"))
    assert candidates, "data/lost/*.png samples missing"
    p = validate_image_file(candidates[0], MAX_BYTES)
    assert p.is_file()


def test_validate_image_file_bad_extension(tmp_path):
    f = tmp_path / "evil.gif"
    f.write_bytes(b"GIF89a")
    with pytest.raises(ValidationError):
        validate_image_file(f, MAX_BYTES)


def test_validate_image_bytes_empty():
    with pytest.raises(ValidationError):
        validate_image_bytes(b"", "upload.png", MAX_BYTES)


def test_validate_image_bytes_bad_suffix():
    with pytest.raises(ValidationError):
        validate_image_bytes(b"not-an-image", "upload.txt", MAX_BYTES)


def test_validate_image_bytes_real_file():
    candidates = list(DATA_LOST.glob("*.png"))
    data = candidates[0].read_bytes()
    suffix = validate_image_bytes(data, candidates[0].name, MAX_BYTES)
    assert suffix == ".png"


def test_validate_image_bytes_oversize():
    candidates = list(DATA_LOST.glob("*.png"))
    data = candidates[0].read_bytes()
    with pytest.raises(ValidationError):
        validate_image_bytes(data, candidates[0].name, max_bytes=10)
