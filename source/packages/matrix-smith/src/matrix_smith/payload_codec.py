"""Unambiguous canonical-JSON payload encoding for sectioned XYZ files."""

from __future__ import annotations

import base64
import binascii
from typing import Sequence


BASE64_CANONICAL_JSON_ENCODING = "BASE64_CANONICAL_JSON_UTF8"
LEGACY_CANONICAL_JSON_ENCODING = "CANONICAL_JSON"


def encode_canonical_json_lines(text: str, *, width: int = 120) -> tuple[str, ...]:
    """Encode JSON so no data line can be mistaken for a section marker."""

    if width < 1:
        raise ValueError("payload chunk width must be positive")
    encoded = base64.b64encode(str(text).encode("utf-8")).decode("ascii")
    return tuple(encoded[start : start + width] for start in range(0, len(encoded), width)) or ("",)


def decode_canonical_json_lines(lines: Sequence[str], *, encoding: str) -> str:
    """Decode current base64 payloads while retaining legacy JSON support."""

    normalized = str(encoding).strip().upper()
    joined = "".join(str(line).strip() for line in lines)
    if normalized == LEGACY_CANONICAL_JSON_ENCODING:
        return joined
    if normalized != BASE64_CANONICAL_JSON_ENCODING:
        raise ValueError(f"unsupported canonical JSON payload encoding: {encoding}")
    try:
        return base64.b64decode(joined, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("invalid base64 canonical JSON payload") from exc


def is_payload_subsection_header(line: str) -> bool:
    """Recognize MATRIX payload markers without classifying JSON arrays."""

    text = str(line).strip()
    if len(text) < 3 or not (text.startswith("[") and text.endswith("]")):
        return False
    name = text[1:-1]
    return bool(name) and name[0].isupper() and all(
        character.isupper() or character.isdigit() or character == "_"
        for character in name
    )


__all__ = [
    "BASE64_CANONICAL_JSON_ENCODING",
    "LEGACY_CANONICAL_JSON_ENCODING",
    "decode_canonical_json_lines",
    "encode_canonical_json_lines",
    "is_payload_subsection_header",
]
