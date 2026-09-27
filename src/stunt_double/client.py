"""Re-export of the sync HTTP client, kept for ``from stunt_double.client import StuntDouble``."""

from .apps.client.http_client import StuntDouble

__all__ = ["StuntDouble"]
