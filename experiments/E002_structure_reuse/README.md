# E002 — Structure reuse

Research question: does prior experience reduce the learning needed for a
related task?

Protocol: `docs/experiment_protocols.md#e002--structure-reuse`.

Run:

```bash
py scripts/run_experiment.py E002
```

Result: `results_seeds_0-1-2.json` (generated 2026-10-29). **Negative result.**
Transfer target MSE 0.010987 ± 0.006544 was worse than equal-budget fresh
0.003650 ± 0.000890, and source retention collapsed. The hypothesis was not
supported for this rule pair. Do not treat this as success.
