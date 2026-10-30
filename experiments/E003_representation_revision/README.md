# E003 — Representation revision

Research question: can the learner adapt after a rule change while preserving
earlier performance?

Protocol: `docs/experiment_protocols.md#e003--representation-revision`.

Run:

```bash
py scripts/run_experiment.py E003
```

Result: `results_seeds_0-1-2.json` (generated 2026-10-29). New-task MSE improved
99.6% and isolated anomalies were rejected, but old-task retention collapsed
without replay. Replay improved retention at the cost of slower adaptation.
Retention is reported, not gated.

## Dual-gated retention (v2)

```bash
py scripts/run_experiment.py E003-RETENTION
```

Result: `retention_seeds_0-1-2.json` (generated 2026-10-30). At 50 epochs:

| Condition | Accepted | New MSE | Old MSE |
| --- | --- | --- | --- |
| Legacy gate | 3/3 | 0.00235 | 0.54772 |
| Dual gate | 0/3 | 0.53570 (restored) | 0.00033 (restored) |
| Dual gate + replay | 3/3 | 0.34272 | 0.03817 |

The dual gate refuses updates that exceed the predeclared retention budget;
replay is what adapts while staying inside it. Isolated anomalies 0/3 accepted;
the injected outlier was filtered in 3/3 contaminated batches. Retention is now
gated, and the criterion is conditional on the declared budget.

## Baseline freeze

`baseline_pre_upgrade.json` exactly reproduces the committed 2026-10-29 summary
and criterion. See `docs/baseline_audit.md`.
