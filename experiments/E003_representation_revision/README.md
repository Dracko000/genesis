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
