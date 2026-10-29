"""Environment-driven API settings with safe local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("./data/genesis.sqlite3")
    event_log_path: Path = Path("./data/events.jsonl")
    checkpoint_dir: Path = Path("./checkpoints")
    research_log_path: Path = Path("./docs/research_log.md")
    default_seed: int = 7
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    environment: str = "development"

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.environ.get("GENESIS_CORS_ORIGINS", "http://localhost:5173")
        return cls(
            db_path=Path(os.environ.get("GENESIS_DB_PATH", "./data/genesis.sqlite3")),
            event_log_path=Path(
                os.environ.get("GENESIS_EVENT_LOG_PATH", "./data/events.jsonl")
            ),
            checkpoint_dir=Path(
                os.environ.get("GENESIS_CHECKPOINT_DIR", "./checkpoints")
            ),
            research_log_path=Path(
                os.environ.get("GENESIS_RESEARCH_LOG_PATH", "./docs/research_log.md")
            ),
            default_seed=int(os.environ.get("GENESIS_DEFAULT_SEED", "7")),
            cors_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
            environment=os.environ.get("GENESIS_ENV", "development"),
        )

    def prepare_directories(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
