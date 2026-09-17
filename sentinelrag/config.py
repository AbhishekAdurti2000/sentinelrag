"""Central configuration, loaded from environment variables / .env.

Kept deliberately tiny and dependency-light so tests can import it without
needing real credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    github_token: str | None = field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))
    repo: str = field(default_factory=lambda: os.getenv("SENTINELRAG_REPO", "langchain-ai/langgraph"))
    model: str = field(default_factory=lambda: os.getenv("SENTINELRAG_MODEL", "claude-sonnet-4-5"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SENTINELRAG_DATA_DIR", "./data")))

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "cache").mkdir(parents=True, exist_ok=True)


settings = Settings()
