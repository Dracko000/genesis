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

---

## 2026-10-30 — Retention and transfer upgrade

### Baseline freeze

Before changing revision logic, the legacy E002 and E003 runs were reproduced
with the current code into new files
(`baseline_pre_upgrade.json`). `summary` and `criterion` matched the committed
results exactly. Details are in `docs/baseline_audit.md`.

### Implemented

- `EvidenceValidator` (median/MAD residual filter, minimum batch size).
- `RetentionRevisionController` with a dual new/old gate, replay, rollback and
  structured revision history.
- E002-v2 transfer matrix with related/unrelated rule pairs and matched target
  budgets (`E002-DIAG`).
- E003-v2 retention comparison (`E003-RETENTION`).
- Hidden-activation, ablation and linear-CKA helpers, plus E005.
- API runners and `/experiments/{id}/revisions`; dashboard revision history.
- New rules: `linear_alt`, `interaction_alt`, `partial_interaction`.

Full test suite: `py -m pytest` → 72 passed, 1 dependency deprecation warning.

### E003-v2 — dual-gated retention (confirmatory)

Command: `py scripts/run_experiment.py E003-RETENTION`

| Condition (50 epochs) | Accepted | New MSE | Old MSE |
| --- | --- | --- | --- |
| Legacy gate | 3/3 | 0.00235 | 0.54772 |
| Dual gate | 0/3 | 0.53570 (restored) | 0.00033 (restored) |
| Dual gate + replay | 3/3 | 0.34272 | 0.03817 |

Retention budget was ~0.051. Isolated-anomaly acceptance 0/3; contaminated
outlier filtered 3/3. The declared criterion was met. Interpretation: the dual
gate prevents catastrophic forgetting by refusing unpreserved updates; replay is
what actually adapts while staying inside the budget.

### E002-v2 — fair transfer diagnostics (mixed)

Command: `py scripts/run_experiment.py E002-DIAG`

| Pair | Transfer MSE | Fresh MSE | Relative improvement |
| --- | --- | --- | --- |
| `linear` → `linear_alt` | 0.000055 | 0.000462 | +87.5% |
| `interaction` → `interaction_alt` | 0.001904 | 0.002026 | +4.6% |
| `linear` → `interaction` (legacy) | 0.017494 | 0.002394 | −701% |
| `linear` → `quadratic` | 0.004579 | 0.004085 | −18.1% |

The pooled related-pair criterion was met (46.1%), but only the additive pair
cleared 10%; the interaction pair was a near miss, so the secondary
`secondary_all_related_pairs_met` check is `false`. Source retention collapsed
for every pair, so these are overwrite-style gains, not preserved prior
knowledge. This heterogeneity is the main finding.

### E005 — representation interventions (probe met, interpretation limited)

Command: `py scripts/run_experiment.py E005`

Selectivity ratio was 1.75 at layer 0 and 2.86 at layer 1, meeting the declared
threshold. CKA between source and target activations was near zero (0.010 and
0.007), so the transferred network did not reuse linearly similar
representations. Ablation lowered source MSE and raised target MSE; the selected
units are more influential than random units, but not "required" in the
predicted direction.

### Next experiment

- Per-pair transfer criteria; do not pool a strong additive result with a weak
  interaction result.
- Encoder-only transfer and target-statistic reset to test whether forgetting is
  caused by output-layer commitment.
- Bounded replay buffers and replay-schedule sweeps; compare against full replay.
- Repeat the intervention probe with multiple random draws and shuffled-label
  controls.
