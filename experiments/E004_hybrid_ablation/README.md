# E004 — Hybrid ablation (exploratory)

Compare neural-only, symbolic-basis-only and a validation-selected hybrid.

Protocol: `docs/experiment_protocols.md#hybrid-ablation`.

Run:

```bash
py scripts/run_experiment.py HYBRID
```

Result: `results_seeds_0-1-2.json` (generated 2026-10-29). The symbolic basis
contains the true terms for the `interaction` rule and reaches machine-precision
MSE, so the hybrid selects symbolic in all seeds. The criterion was met but the
comparison is confounded by the basis prior. This is not evidence of general
hybrid superiority.
