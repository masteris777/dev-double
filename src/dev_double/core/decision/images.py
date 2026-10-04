"""Images in a request: base64 strings, checked and typed by their magic bytes.

Only the formats vision models take everywhere are accepted: PNG, JPEG, WebP.
"""

from __future__ import annotations

import base64
from typing import Optional

HEAD_CHARS = 16  # base64 characters (12 bytes), enough to cover every format's signature


def mime_of(head: bytes) -> Optional[str]:
    """The MIME type of the image that starts with ``head``, or None if it isn't PNG, JPEG, or WebP."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_mime(image: str) -> Optional[str]:
    """The MIME type of a base64 image, read from its first bytes only."""
    try:
        return mime_of(base64.b64decode(image[:HEAD_CHARS]))
    except ValueError:  # binascii.Error is one
        return None


def check_image(image: str, where: str = "image") -> str:
    """The MIME type of ``image``; raises ``ValueError`` (naming ``where``) unless it is
    plain base64 of a PNG, JPEG, or WebP. URLs and data URLs are refused."""
    if image.startswith("data:"):
        raise ValueError(f"`{where}` must be plain base64, not a data URL.")
    if "://" in image:
        raise ValueError(f"`{where}` must be base64 image data, not a URL.")
    try:
        data = base64.b64decode(image, validate=True)
    except ValueError:
        raise ValueError(f"`{where}` is not valid base64.") from None
    mime = mime_of(data[:12])
    if mime is None:
        raise ValueError(f"`{where}` is not a PNG, JPEG, or WebP image.")
    return mime
