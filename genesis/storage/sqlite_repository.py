"""SQLite implementation of :class:`genesis.storage.repository.Repository`.

SQLite is the initial prototype backend. Connections use ``check_same_thread``
plus an application-level lock so that background experiment threads and API
requests can share one repository safely. Experiment history is never deleted.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from genesis.experience.schema import Observation
from genesis.representation.checkpoint import CheckpointMetadata
from genesis.storage.repository import (
    CheckpointRecord,
    ExperimentRecord,
    LogRecord,
    MetricRecord,
    Repository,
    _utc_now,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    config TEXT NOT NULL,
    seed INTEGER,
    result TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_logs_created ON logs(created_at);
"""


def _to_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _from_json(value: str) -> Any:
    return json.loads(value)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLiteRepository(Repository):
    """Thread-safe SQLite repository."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(_SCHEMA)
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # -- experiences -----------------------------------------------------
    def add_experience(self, observation: Observation) -> Observation:
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO experiences (id, payload, created_at) VALUES (?, ?, ?)",
                    (
                        observation.id,
                        _to_json(observation.model_dump(mode="json")),
                        _utc_now().isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"experience {observation.id!r} already exists") from exc
            self._connection.commit()
        return observation

    def get_experience(self, observation_id: str) -> Observation | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM experiences WHERE id = ?", (observation_id,)
            ).fetchone()
        return None if row is None else Observation.model_validate(_from_json(row["payload"]))

    def list_experiences(self, *, limit: int = 100, offset: int = 0) -> list[Observation]:
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be non-negative")
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM experiences ORDER BY rowid DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [Observation.model_validate(_from_json(row["payload"])) for row in rows]

    # -- experiments -----------------------------------------------------
    def create_experiment(
        self,
        *,
        experiment_id: str,
        name: str,
        status: str,
        config: dict[str, Any],
        seed: int | None = None,
    ) -> ExperimentRecord:
        now = _utc_now()
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO experiments "
                    "(id, name, status, config, seed, result, error, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, '{}', NULL, ?, ?)",
                    (experiment_id, name, status, _to_json(config), seed, now.isoformat(), now.isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"experiment {experiment_id!r} already exists") from exc
            self._connection.commit()
        return ExperimentRecord(
            id=experiment_id,
            name=name,
            status=status,
            config=config,
            seed=seed,
            created_at=now,
            updated_at=now,
        )

    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> ExperimentRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"experiment {experiment_id!r} does not exist")
            updates: list[str] = []
            params: list[Any] = []
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            if result is not None:
                updates.append("result = ?")
                params.append(_to_json(result))
            if error is not None:
                updates.append("error = ?")
                params.append(error)
            updates.append("updated_at = ?")
            params.append(_utc_now().isoformat())
            params.append(experiment_id)
            self._connection.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?", params
            )
            self._connection.commit()
            row = self._connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return self._experiment_row(row)

    @staticmethod
    def _experiment_row(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            id=row["id"],
            name=row["name"],
            status=row["status"],
            config=_from_json(row["config"]),
            seed=row["seed"],
            result=_from_json(row["result"]),
            error=row["error"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return None if row is None else self._experiment_row(row)

    def list_experiments(self, *, limit: int = 100) -> list[ExperimentRecord]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._experiment_row(row) for row in rows]

    # -- metrics ---------------------------------------------------------
    def add_metric(
        self,
        *,
        experiment_id: str,
        name: str,
        value: float,
        step: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> MetricRecord:
        now = _utc_now()
        with self._lock:
            if self._connection.execute(
                "SELECT 1 FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone() is None:
                raise KeyError(f"experiment {experiment_id!r} does not exist")
            cursor = self._connection.execute(
                "INSERT INTO metrics (experiment_id, name, value, step, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    name,
                    float(value),
                    step,
                    _to_json(payload or {}),
                    now.isoformat(),
                ),
            )
            self._connection.commit()
            metric_id = int(cursor.lastrowid)
        return MetricRecord(
            id=metric_id,
            experiment_id=experiment_id,
            name=name,
            value=float(value),
            step=step,
            payload=payload or {},
            created_at=now,
        )

    def list_metrics(self, experiment_id: str) -> list[MetricRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM metrics WHERE experiment_id = ? ORDER BY id", (experiment_id,)
            ).fetchall()
        return [
            MetricRecord(
                id=row["id"],
                experiment_id=row["experiment_id"],
                name=row["name"],
                value=row["value"],
                step=row["step"],
                payload=_from_json(row["payload"]),
                created_at=_dt(row["created_at"]),
            )
            for row in rows
        ]

    # -- checkpoints -----------------------------------------------------
    def add_checkpoint(self, *, path: str, metadata: CheckpointMetadata) -> CheckpointRecord:
        now = _utc_now()
        payload = metadata.model_dump(mode="json")
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO checkpoints (checkpoint_id, path, metadata, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (metadata.checkpoint_id, path, _to_json(payload), now.isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"checkpoint {metadata.checkpoint_id!r} already exists; refusing to overwrite"
                ) from exc
            self._connection.commit()
        return CheckpointRecord(
            checkpoint_id=metadata.checkpoint_id,
            path=path,
            metadata=payload,
            created_at=now,
        )

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            ).fetchone()
        return None if row is None else self._checkpoint_row(row)

    def list_checkpoints(self, *, limit: int = 100) -> list[CheckpointRecord]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM checkpoints ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._checkpoint_row(row) for row in rows]

    @staticmethod
    def _checkpoint_row(row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=row["checkpoint_id"],
            path=row["path"],
            metadata=_from_json(row["metadata"]),
            created_at=_dt(row["created_at"]),
        )

    # -- revisions and logs ---------------------------------------------
    def add_revision(self, *, experiment_id: str | None, payload: dict[str, Any]) -> int:
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO revisions (experiment_id, payload, created_at) VALUES (?, ?, ?)",
                (experiment_id, _to_json(payload), _utc_now().isoformat()),
            )
            self._connection.commit()
            return int(cursor.lastrowid)

    def list_revisions(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if experiment_id is None:
                rows = self._connection.execute(
                    "SELECT * FROM revisions ORDER BY id"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT * FROM revisions WHERE experiment_id = ? ORDER BY id",
                    (experiment_id,),
                ).fetchall()
        return [
            {
                "id": row["id"],
                "experiment_id": row["experiment_id"],
                "payload": _from_json(row["payload"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def add_log(
        self,
        *,
        level: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> LogRecord:
        now = _utc_now()
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO logs (level, message, payload, created_at) VALUES (?, ?, ?, ?)",
                (level, message, _to_json(payload or {}), now.isoformat()),
            )
            self._connection.commit()
            log_id = int(cursor.lastrowid)
        return LogRecord(
            id=log_id,
            level=level,
            message=message,
            payload=payload or {},
            created_at=now,
        )

    def list_logs(self, *, limit: int = 200) -> list[LogRecord]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            LogRecord(
                id=row["id"],
                level=row["level"],
                message=row["message"],
                payload=_from_json(row["payload"]),
                created_at=_dt(row["created_at"]),
            )
            for row in rows
        ]
