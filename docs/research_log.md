# Genesis Research Log

## 2026-10-29 — Genesis v0.1 initial build and first runs

### Environment

- Windows, CPython 3.14.5, NumPy 2.5.1, PyTorch 2.14.1+cpu, Pydantic 2.13.4,
  FastAPI 0.141.1.
- Commands run from the repository root with the `py` launcher.
- Full test suite: `py -m pytest` → 56 passed, 1 dependency deprecation warning
  (Starlette/hypers, not a Genesis failure).

### Implemented

Experience schema/provenance, append-only JSONL log, character tokenizer and
vocabulary, deterministic synthetic world with five rules, numeric encoder,
PyTorch MLP learner, checkpoints, validation-gated revision, replay buffer,
SQLite repository, FastAPI service, Vite/React dashboard, E001/E002/E003 and the
hybrid ablation.

### E001 — emergent structure (confirmatory)

Command: `py scripts/run_experiment.py E001`

Held-out sign combination `(+,+)`, `interaction` rule, seeds 0/1/2, 400 train,
200 test.

| Method | MSE (mean ± std) | R² |
| --- | --- | --- |
| Neural | 0.040467 ± 0.009052 | 0.7877 |
| Nearest neighbour | 0.221033 ± 0.024486 | -0.1654 |
| Ridge | 1.128036 ± 0.146789 | -4.9398 |
| Mean | 0.346543 ± 0.018219 | -0.8331 |
| Oracle | 0.000000 | 1.0000 |

Declared criterion met: 81.7% lower MSE than 1-NN, R² 0.788. Feature permutation
raised neural MSE to 0.103093 ± 0.010817. Sample-efficiency neural MSE at
50/100/200 examples: 0.0497 / 0.0469 / 0.0570. Noise sensitivity: at σ=0.1
neural 0.0828; at σ=0.3 neural 0.5862, worse than 1-NN 0.3572.

Interpretation: useful generalisation to held-out combinations, but robustness
to heavy label noise is weak, and the result does not demonstrate concept
formation.

### E002 — structure reuse (confirmatory) — NEGATIVE RESULT

Command: `py scripts/run_experiment.py E002`

Source `linear` → target `interaction`, 25 target epochs for small-budget
conditions.

| Condition | Target MSE |
| --- | --- |
| Transfer | 0.010987 ± 0.006544 |
| Fresh, equal budget | 0.003650 ± 0.000890 |
| Fresh, full budget | 0.001997 ± 0.000587 |
| Unrelated prior | 0.002547 ± 0.000615 |

Transfer was 201% worse than equal-budget fresh. Source retention MSE after
adaptation rose from 0.000017 to 0.632762. The declared criterion was not met.
The hypothesis is not supported for this pair. Plausible causes: target-
statistic anchoring, output-layer commitment, or genuinely unhelpful source
features. This result is not reinterpreted as success.

### E003 — representation revision (confirmatory + exploratory replay)

Command: `py scripts/run_experiment.py E003`

Rule change `interaction` → `changed`, seeds 0/1/2.

- New-task MSE: 0.535701 ± 0.010653 → 0.002354 ± 0.000447 (99.6% improvement).
- Old-task MSE: 0.000326 ± 0.000102 → 0.547721 ± 0.029716 (catastrophic forgetting).
- Validation-gated updates accepted in 3/3 seeds.
- Single isolated anomaly rejected in 3/3 seeds.
- Replay consolidation (exploratory): old-task MSE 0.038171 ± 0.008553,
  new-task MSE 0.342718 ± 0.025429.

Interpretation: controlled revision adapts successfully and protects against a
single outlier, but retention fails without replay, and simple replay trades
adaptation speed for retention. The adaptation/retention trade-off is unresolved.

### Hybrid ablation (exploratory)

Command: `py scripts/run_experiment.py HYBRID`

The symbolic basis contains the true terms for the `interaction` rule, so it
reaches machine-precision MSE (neural-only 0.000649 ± 0.000185). The hybrid
selected symbolic in 3/3 seeds. The declared criterion was met, but the result
is confounded by the basis prior and is **not** evidence of general hybrid
superiority.

### Test status

`py -m pytest` → 56 passed. Tests cover data validation, deterministic environment
generation, train/eval separation, target-leakage rejection, learner training,
checkpoint save/restore equivalence, revision acceptance/rollback, event-log
persistence, SQLite round-trips, API validation, background experiment
execution, and experiment reproducibility.

### Next experiment

E002 diagnostic follow-up (encoder-only transfer, target-statistic reset,
warm-up schedule sweep) to distinguish an optimisation artefact from a genuine
representation effect. Secondary: replay scheduling for E003 retention without
the observed adaptation slowdown.
