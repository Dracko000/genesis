# Research Hypotheses

## Primary hypothesis

> A learning system with minimal predefined conceptual structure can develop
> increasingly useful internal representations from structured experiences,
> prediction errors, and feedback, without requiring a predefined ontology of
> high-level concepts.

The hypothesis is falsifiable. If the implemented mechanisms do not outperform
the relevant baselines, the result is documented and the hypothesis is revised.

## Operational definitions

A representation is judged only by measurable consequences:

1. **Generalisation**: held-out prediction error on examples not seen in
   training, including unseen feature combinations.
2. **Prediction**: error between predicted and observed outcomes.
3. **Transfer**: target-task error at a fixed target-task budget after prior
   experience, compared with fresh initialisation.
4. **Adaptation**: new-task error before and after a controlled update when the
   hidden rule changes.
5. **Retention**: error on an earlier task after adapting to the new one.
6. **Sample efficiency**: held-out error as a function of training-set size.
7. **Robustness**: sensitivity to label noise, renamed/permuted features and
   isolated anomalous observations.

Predictive improvement alone is not evidence of human-like concepts. Any
"concept" language requires additional interventions and behavioural tests that
this prototype does not yet implement.

## Sub-hypotheses

### H1 — Emergent structure (E001)

A small neural learner on raw neutral features predicts held-out sign
combinations better than a nearest-example baseline.

- Falsified if neural held-out MSE is not at least 10% lower than
  nearest-neighbour, or if mean R² < 0.5.
- Status (2026-10-29): **supported** for the `interaction` rule and the declared
  held-out quadrant (81.7% lower MSE, R² 0.788).
- Caveat: this is a narrow operationalisation; it does not establish concept
  formation.

### H2 — Structure reuse (E002)

Warm-start transfer from a related task lowers target-task error at an
equal target-task budget relative to fresh initialisation.

- Falsified if transfer target MSE is not at least 10% lower than the
  equal-budget fresh initialisation.
- Status (2026-10-29): **falsified** for linear → interaction under 25 target
  epochs. Transfer was 201% worse than fresh; source retention collapsed.
- This is retained as a negative result. Possible causes: target-statistic
  anchoring, output-layer commitment, or genuinely unhelpful source features.
- Status (2026-10-30, E002-v2): **mixed**. Under matched 50-epoch budgets the
  additive-related pair improved 87.5%, the interaction-related pair improved
  only 4.6%, and unrelated pairs were negative. The pooled criterion was met,
  but pooling hides the failure. Source retention collapsed in every pair, so
  the gains are overwrite-style rather than preserved reuse.

### H3 — Representation revision (E003)

A validation-gated update reduces new-task error after a rule change while
rejecting an isolated anomalous observation.

- Falsified if new-task MSE improves by less than 10%, or if an isolated
  anomalous observation is accepted on its own.
- Status (2026-10-29): **supported** for the `interaction` → `changed` pair.
  New-task MSE fell 99.6%; the isolated anomaly was rejected in all seeds.
- Retention is **not** supported: old-task MSE rose from 0.000326 to 0.547721.
  Replay improved retention but slowed adaptation. This is a trade-off to be
  studied, not a solved problem.
- Status (2026-10-30, E003-v2): **supported under the declared budget**. The
  dual gate rejected all three forgetting updates and restored state; dual gate
  plus replay accepted 3/3, improved new-task MSE 36.0%, and kept old-task MSE
  (0.0382) inside the ~0.051 budget. Isolated and embedded outliers were
  rejected. This is conditional on the predeclared budget, not a general
  retention guarantee.

### H5 — Source-specialised units are causally influential (E005)

Ablating the source-most-active hidden units changes source MSE more than a
matched random ablation.

- Criterion: maximum per-layer selectivity ratio ≥ 1.5.
- Status (2026-10-30): **supported as an influence claim only**. Ratios were
  1.75 and 2.86, but ablation lowered source MSE while raising target MSE and
  CKA was near zero. The units are influential, not positively required, and
  the probe does not establish semantic reuse.

### H4 — Hybrid advantage

A validation-selected hybrid of the neural baseline and a symbolic basis is no
worse than the better single method on held-out data.

- Criterion: hybrid MSE at most 5% worse than the better single method.
- Status (2026-10-29): **criterion met but confounded**. The symbolic basis
  explicitly contains the true terms (`f1*f2`, `f3`), giving it an oracle-like
  prior. The hybrid selected symbolic in all seeds. This is not evidence that
  hybrid learning is generally superior.

## Baseline policy

Every experiment compares against the relevant simple methods:

- training-set mean (sanity floor),
- 1-nearest-neighbour on standardised features (memorisation),
- closed-form ridge (simple parametric control),
- ground-truth rule oracle (upper reference only, never given to the learner).

The oracle is allowed to construct predictions from the true rule because it is
an evaluation reference. It is never an input to a learner.

## Falsification and reporting policy

- Hypothesis, independent variables, controls, splits, baselines, metrics, seeds,
  expected outcomes, failure conditions and success thresholds are declared in
  protocol files before runs.
- Thresholds are not chosen after seeing results.
- Results report variability across seeds (mean ± sample standard deviation).
- Negative results are kept in the research log with the same prominence as
  positive ones.
- Exploratory results are labelled as exploratory; they do not retroactively
  change a confirmatory criterion.
- No experiment is reported as successful unless it was executed and its output
  inspected.

## Claims policy

The following claims are **not** supported by this prototype:

- that Genesis has general intelligence or human-like understanding,
- that tokenization or segmentation provides semantic understanding,
- that improved prediction implies human-interpretable concepts,
- that a hybrid is superior in general,
- that transfer works across related tasks in general (one additive pair helped,
  one interaction pair did not, and source retention collapsed),
- that adaptation preserves earlier knowledge without an explicit retention
  gate and replay (it did not),
- that ablation-selected units are positively required for a task (E005 found
  influence and near-zero CKA, not reuse).

## Open questions

1. Is E002's negative transfer an optimisation artefact? Test encoder-only
   transfer, target-statistic reset and warm-up schedules.
2. Can replay or regularisation achieve retention without the observed
   adaptation slowdown?
3. Do the E001 gains survive in higher dimensions, with non-stationary rules, or
   under structured (non-IID) experience streams?
4. Which intervention could provide evidence for reusable internal structure
   beyond prediction accuracy (e.g. probe transfer, ablation of latent units)?
