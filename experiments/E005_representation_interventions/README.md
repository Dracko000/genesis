# E005 — Representation interventions

Research question: are the hidden units most active on the source task causally
influential, and what happens to that influence after transfer to a related
target task?

Protocol: `docs/experiment_protocols.md#e005--representation-interventions`.

Run:

```bash
py scripts/run_experiment.py E005
```

Result: `results_seeds_0-1-2.json` (generated 2026-10-30). Source-top-k ablation
changed source MSE by 2.86× the matched-random magnitude at layer 1 (1.75× at
layer 0), meeting the declared selectivity threshold. CKA between source and
target activations was near zero (0.010 and 0.007), and ablation lowered source
MSE while raising target MSE. The units are more influential than random units,
but the probe does not show positive requirement or semantic reuse.
