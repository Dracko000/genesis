"""Genesis: an experimental experience-driven representation learning system.

The package is intentionally organised around explicit boundaries:

- ``genesis.experience``: validated observations and append-only event logs.
- ``genesis.language``: inspectable, non-pretrained text primitives.
- ``genesis.environments``: deterministic synthetic worlds with hidden rules.
- ``genesis.representation``: learned state, versioning and checkpoints.
- ``genesis.learning``: neural baseline, prediction and controlled revision.
- ``genesis.evaluation``: datasets, metrics, benchmarks and regression checks.
- ``genesis.storage``: repository interfaces and the SQLite implementation.
"""

from genesis.version import __version__

__all__ = ["__version__"]
