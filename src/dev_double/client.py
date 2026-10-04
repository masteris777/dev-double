"""Re-export of the sync HTTP client, kept for ``from dev_double.client import DevDouble``."""

from .apps.client.http_client import DevDouble, image_base64

__all__ = ["DevDouble", "image_base64"]
