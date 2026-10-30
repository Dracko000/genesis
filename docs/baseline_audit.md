# Pre-upgrade baseline audit

This audit freezes the behaviour of the v0.1 code before the retention and
transfer upgrades. Existing result files were **not** overwritten. The
reproduction was written to new files so it can be diffed against the original
artefacts.

## Environment

| Item | Value |
| --- | --- |
| Commit | `45547d46cdc0f096b8b7ff6e098e4db123071f1e` |
| Python | 3.14.5 |
| torch | 2.14.1+cpu |
| numpy | 2.5.1 |
| pydantic | 2.13.4 |
| fastapi | 0.141.1 |
| Platform | win32 |
| Seeds | 0, 1, 2 |

## Reproduction commands

```bash
py scripts/run_experiment.py E002 \
  --output experiments/E002_structure_reuse/baseline_pre_upgrade.json

py scripts/run_experiment.py E003 \
  --output experiments/E003_representation_revision/baseline_pre_upgrade.json
```

## Verification

`summary` and `criterion` for both new files are **exactly equal** to the
committed `results_seeds_0-1-2.json` files:

```text
E002_structure_reuse        summary_equal: True  criterion_equal: True
E003_representation_revision summary_equal: True  criterion_equal: True
```

## E002 baseline (structure reuse)

| Condition | MSE mean |
| --- | --- |
| `transfer_small_budget` | 0.010987 |
| `fresh_small_budget` | 0.003650 |
| `fresh_full_budget` | 0.001997 |
| `unrelated_prior_small_budget` | 0.002547 |
| `source_only` | 0.000017 |
| `transfer_source_retention` | 0.632762 |

Criterion `transfer <= 0.9 * fresh_small` was **not met**
(`relative_improvement_vs_fresh = -2.010`). The source task was almost perfectly
retained before transfer and collapsed during it. The result is a documented
negative, not a bug to be hidden.

## E003 baseline (representation revision)

| Measure | MSE mean |
| --- | --- |
| New task before update | 0.535701 |
| New task after update | 0.002354 |
| Old task before update | 0.000326 |
| Old task after update | 0.547721 |
| Replay: new task | 0.342718 |
| Replay: old task | 0.038171 |

Accepted update rate was 1.0; isolated-anomaly acceptance was 0.0. New-task
improvement was 99.6%, but old-task retention collapsed without replay. The
legacy criterion only gated new-task improvement, so it reports `met: true`
while retention is catastrophic.

## Implications for the upgrade

- E002 must be re-run with matched controls and related/unrelated rule pairs
  before claiming transfer; the legacy pair (`linear` → `interaction`) is
  effectively unrelated.
- E003 needs a dual gate (new-task improvement **and** bounded old-task
  deterioration) plus evidence filtering and replay, reported as a separate
  experiment. The legacy `run_e003` and its result file remain unchanged.
