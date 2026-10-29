"""Reproducible checkpoint persistence.

A checkpoint embeds the parameter state, the encoder contract, model
configuration and metadata in one ``.pt`` file, plus a human-readable ``.json``
sidecar. Loading validates the format/schema version before returning anything.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from genesis.representation.versioning import (
    CHECKPOINT_FORMAT,
    SCHEMA_VERSION,
    ensure_supported_checkpoint,
)
from genesis.version import __version__


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CheckpointMetadata(BaseModel):
    """Inspectable metadata stored next to parameter tensors."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=_utc_now)
    genesis_version: str = __version__
    format: str = CHECKPOINT_FORMAT
    schema_version: int = SCHEMA_VERSION
    seed: int
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    feature_names: list[str]
    include_action: bool = False
    encoder_state: dict[str, Any] = Field(default_factory=dict)
    target_mean: float = 0.0
    target_std: float = 1.0
    training_summary: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


@dataclass
class LoadedCheckpoint:
    """A checkpoint payload returned by :func:`load_checkpoint`."""

    metadata: CheckpointMetadata
    state_dict: dict[str, Any]
    path: Path
    extra: dict[str, Any] = field(default_factory=dict)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json") if path.suffix else path.with_suffix(".json")


def save_checkpoint(
    path: str | Path,
    *,
    state_dict: dict[str, Any],
    metadata: CheckpointMetadata,
) -> Path:
    """Atomically persist parameters and metadata.

    Existing checkpoints are never overwritten silently: callers must choose a
    new path or pass an explicit checkpoint id. An accidental collision raises
    ``FileExistsError``.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {target}")

    payload = {
        "metadata": metadata.model_dump(mode="json"),
        "state_dict": state_dict,
    }
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)

    sidecar = _sidecar_path(target)
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_tmp.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    sidecar_tmp.replace(sidecar)
    return target


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> LoadedCheckpoint:
    """Load and validate a checkpoint."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "metadata" not in payload or "state_dict" not in payload:
        raise ValueError("invalid checkpoint payload")
    metadata_raw = payload["metadata"]
    if not isinstance(metadata_raw, dict):
        raise ValueError("invalid checkpoint metadata")
    ensure_supported_checkpoint(metadata_raw)
    metadata = CheckpointMetadata.model_validate(metadata_raw)
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("invalid checkpoint state_dict")
    extra = {key: value for key, value in payload.items() if key not in {"metadata", "state_dict"}}
    return LoadedCheckpoint(metadata=metadata, state_dict=state_dict, path=source, extra=extra)


def checkpoints_equivalent(
    left: LoadedCheckpoint,
    right: LoadedCheckpoint,
    *,
    atol: float = 1e-6,
) -> bool:
    """Check parameter equivalence within an explicit tolerance."""

    if left.state_dict.keys() != right.state_dict.keys():
        return False
    for key in left.state_dict:
        lhs = left.state_dict[key]
        rhs = right.state_dict[key]
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            if lhs.shape != rhs.shape or not torch.allclose(
                lhs.float(), rhs.float(), atol=atol, rtol=0.0
            ):
                return False
        elif lhs != rhs:
            return False
    return True
