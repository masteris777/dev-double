"""Re-export of the sync HTTP client, kept for ``from dev_double.client import DevDouble``."""

from .apps.client.http_client import DevDouble

__all__ = ["DevDouble"]
