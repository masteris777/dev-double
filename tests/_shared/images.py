"""Tiny images for tests, built with the standard library."""

from __future__ import annotations

import base64
import struct
import zlib


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def solid_png(rgb: tuple[int, int, int], size: int = 64) -> bytes:
    """A ``size`` x ``size`` PNG of one colour."""
    row = b"\x00" + bytes(rgb) * size  # filter type 0, then the pixels
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(row * size))
        + _chunk(b"IEND", b"")
    )


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


PNG = b64(solid_png((255, 0, 0), size=4))
# Only the signatures: enough to be recognised by their magic bytes.
JPEG = b64(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
WEBP = b64(b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16)
RED_PNG = b64(solid_png((255, 0, 0)))
