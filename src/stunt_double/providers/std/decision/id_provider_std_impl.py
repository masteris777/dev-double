"""Random UUID4 ids."""

from __future__ import annotations

import uuid

from stunt_double.core.decision.i_id_provider import IIdProvider


class IdProviderStdImpl(IIdProvider):
    def new_id(self) -> str:
        return str(uuid.uuid4())
