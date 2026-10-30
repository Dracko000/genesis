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

## Fair diagnostics (v2)

```bash
py scripts/run_experiment.py E002-DIAG
```

Result: `diagnostics_seeds_0-1-2.json` (generated 2026-10-30). Matched 50-epoch
target budget, same architecture seed, related and unrelated rule pairs.

| Pair | Transfer MSE | Fresh MSE | Improvement |
| --- | --- | --- | --- |
| `linear` → `linear_alt` | 0.000055 | 0.000462 | +87.5% |
| `interaction` → `interaction_alt` | 0.001904 | 0.002026 | +4.6% |
| `linear` → `interaction` | 0.017494 | 0.002394 | −701% |
| `linear` → `quadratic` | 0.004579 | 0.004085 | −18.1% |

The pooled related-pair criterion is met (46.1%), driven almost entirely by the
additive pair. The secondary check `secondary_all_related_pairs_met` is `false`.
Source retention collapsed for every pair. This is overwrite, not preserved
knowledge, and the pooled number hides an interaction-pair failure.

## Baseline freeze

`baseline_pre_upgrade.json` exactly reproduces the committed 2026-10-29 summary
and criterion. See `docs/baseline_audit.md`.
