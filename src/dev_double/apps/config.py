"""Settings from environment variables (CLI flags override them)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

ENV_PREFIX = "DEV_DOUBLE_"
ENGINES = ("openai", "mock", "systemone", "needle")
TRUE_WORDS = ("1", "true", "yes", "on")


@dataclass
class Settings:
    engine: str = "openai"  # one of ENGINES
    base_url: str = "http://localhost:11434/v1"  # OpenAI-compatible server (engine=openai)
    model: str = "qwen2.5:7b"
    vision_model: Optional[str] = None  # answers questions about images (engine=openai), e.g. qwen2.5vl:7b
    api_key: Optional[str] = None
    max_concurrency: int = 4
    timeout: float = 120.0
    record: Optional[str] = None
    systemone_url: str = "http://localhost:8000"  # Kev / Laya server (engine=systemone)
    honor_request_model: bool = False  # answer with the request's `model`, not the configured one

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        env = os.environ
        s.engine = env.get(ENV_PREFIX + "ENGINE", s.engine)
        s.base_url = env.get(ENV_PREFIX + "BASE_URL", s.base_url)
        s.model = env.get(ENV_PREFIX + "MODEL", s.model)
        s.vision_model = env.get(ENV_PREFIX + "VISION_MODEL", s.vision_model)
        s.api_key = env.get(ENV_PREFIX + "API_KEY", s.api_key)
        s.max_concurrency = int(env.get(ENV_PREFIX + "MAX_CONCURRENCY", s.max_concurrency))
        s.timeout = float(env.get(ENV_PREFIX + "TIMEOUT", s.timeout))
        s.record = env.get(ENV_PREFIX + "RECORD", s.record)
        s.systemone_url = env.get(ENV_PREFIX + "SYSTEMONE_URL", s.systemone_url)
        s.honor_request_model = (
            env.get(ENV_PREFIX + "HONOR_REQUEST_MODEL", str(s.honor_request_model)).strip().lower() in TRUE_WORDS
        )
        return s
