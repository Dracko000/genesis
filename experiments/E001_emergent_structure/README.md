# E001 — Emergent structure

Research question: can a learner improve prediction on unseen feature
combinations without being given high-level concepts?

Protocol: `docs/experiment_protocols.md#e001--emergent-structure`.

Run:

```bash
py scripts/run_experiment.py E001
```

Result: `results_seeds_0-1-2.json` (generated 2026-10-29). Neural held-out MSE
0.0405 ± 0.0091 vs nearest-neighbour 0.2210 ± 0.0245; declared criterion met.
Improved prediction is not evidence of concept formation.
