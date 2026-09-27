"""Settings from environment variables (CLI flags override them)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .engines import Engine, FakeEngine, OpenAICompatEngine

ENV_PREFIX = "STUNT_DOUBLE_"


@dataclass
class Settings:
    engine: str = "openai"
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b"
    api_key: str | None = None
    max_concurrency: int = 4
    timeout: float = 120.0
    record: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        env = os.environ
        s.engine = env.get(ENV_PREFIX + "ENGINE", s.engine)
        s.base_url = env.get(ENV_PREFIX + "BASE_URL", s.base_url)
        s.model = env.get(ENV_PREFIX + "MODEL", s.model)
        s.api_key = env.get(ENV_PREFIX + "API_KEY", s.api_key)
        s.max_concurrency = int(env.get(ENV_PREFIX + "MAX_CONCURRENCY", s.max_concurrency))
        s.timeout = float(env.get(ENV_PREFIX + "TIMEOUT", s.timeout))
        s.record = env.get(ENV_PREFIX + "RECORD", s.record)
        return s

    def build_engine(self) -> Engine:
        if self.engine == "fake":
            return FakeEngine()
        if self.engine == "openai":
            return OpenAICompatEngine(
                base_url=self.base_url, model=self.model, api_key=self.api_key, timeout=self.timeout
            )
        raise ValueError(f"Unknown engine {self.engine!r}; expected 'openai' or 'fake'.")
