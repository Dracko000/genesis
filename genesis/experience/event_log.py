"""Append-only JSONL event log for reproducible experiment history.

The event log is intentionally separate from the SQLite repository. It is the
durable, human-readable audit trail: every meaningful state transition can be
appended as a single JSON line and replayed later. The log never rewrites or
deletes earlier lines.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Event(BaseModel):
    """A single append-only experiment event."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=_utc_now)
    event_type: str = Field(min_length=1, max_length=120)
    payload: dict[str, Any] = Field(default_factory=dict)


class EventLog:
    """Thread-safe JSONL writer/reader.

    Each instance owns one file. Appends are flushed and fsynced so that a
    crash cannot leave a partially buffered event silently lost.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _last_sequence(self) -> int:
        last = -1
        if not self.path.exists():
            return last
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last = max(last, int(event.get("sequence", -1)))
        return last

    def append(self, event_type: str, payload: dict[str, Any] | None = None) -> Event:
        """Append one event and return it with its assigned sequence number."""

        with self._lock:
            event = Event(
                sequence=self._last_sequence() + 1,
                event_type=event_type,
                payload=payload or {},
            )
            line = event.model_dump_json()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def read_all(self) -> list[Event]:
        """Read and validate every event in file order."""

        if not self.path.exists():
            return []
        events: list[Event] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(Event.model_validate_json(line))
        return events

    def iter_events(self, event_type: str | None = None) -> Iterator[Event]:
        for event in self.read_all():
            if event_type is None or event.event_type == event_type:
                yield event

    def tail(self, limit: int = 100) -> list[Event]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        at_most = self.read_all()
        return at_most[-limit:] if limit else []
