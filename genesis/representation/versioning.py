"""Version identifiers and compatibility checks for persisted state."""

from __future__ import annotations

SCHEMA_VERSION = 1
CHECKPOINT_FORMAT = "genesis.checkpoint"
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
DEFAULT_PREDICTION_TOLERANCE = 1e-6


class VersionCompatibilityError(RuntimeError):
    """Raised when persisted state was written by an unsupported version."""


def ensure_supported_checkpoint(metadata: dict[str, object]) -> None:
    fmt = metadata.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise VersionCompatibilityError(
            f"unsupported checkpoint format {fmt!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    version = metadata.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise VersionCompatibilityError(
            f"unsupported checkpoint schema version {version!r}; "
            f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
