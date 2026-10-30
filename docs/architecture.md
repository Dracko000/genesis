# Genesis Architecture (v0.1)

## Pipeline

```
Experience Input
  -> Input Encoding
  -> Representation Update
  -> Prediction
  -> Prediction Error
  -> Evaluation
  -> Optional Revision
  -> Checkpoint and Experiment Log
```

The pipeline is split across packages with one responsibility per boundary. The
learning algorithm never imports the API, and the API never contains learning
logic.

## Module map

| Package | Responsibility |
| --- | --- |
| `genesis.experience` | `Observation`, `Provenance`, JSONL event log |
| `genesis.language` | Character tokenizer, vocabulary, segmentation |
| `genesis.environments` | `Dataset`, `Environment`, `SyntheticWorld` and rules |
| `genesis.representation` | Feature encoder, `LatentState`, checkpoints, versioning |
| `genesis.learning` | Neural predictor, learner, revision/retention, interventions, consolidation, symbolic/hybrid |
| `genesis.evaluation` | Splits, metrics, baselines, regression checks, E001–E005 runners |
| `genesis.storage` | `Repository` contract and `SQLiteRepository` |
| `services.api` | FastAPI schemas, endpoints, background experiment workers |
| `apps.research-ui` | Vite + React + TypeScript dashboard |

## Data contracts

### Observation

- `features`: named finite floats. Targets must not be embedded here.
- `outcome`: optional scalar target. `None` means unlabeled.
- `action`, `action_label`, `context`, `text`: optional surface data.
- `feedback`: optional scalar teacher feedback, kept separate from `outcome`.
- `provenance`: source, generator, seed and the `inferred` flag.
- `kind`: `observation` or `inference`.

Invariant: `kind=inference` requires `provenance.inferred=True`, and
`kind=observation` rejects `provenance.inferred=True`. A hypothesis can never
be stored as a verified observation.

### Dataset

A `Dataset` is a feature matrix plus a separate target vector and neutral
feature names (`f1`, `f2`, ...). The ground-truth rule lives in the environment
and evaluation layer and is never placed in the learner's input.

## Learner

`Learner` binds:

- `NumericFeatureEncoder`: standardises only the declared feature names (plus an
  optional action) using training statistics frozen at fit time.
- `NeuralPredictor`: an MLP regressor with configurable hidden sizes,
  activation, learning rate, batch size, weight decay, epochs and seed.
  Targets are standardised using training statistics only.

`NeuralPredictor.fit` supports `reset_parameters=False` for warm-start transfer
and a `epochs` override for budget-controlled experiments. Initialisation uses
PyTorch's default `Linear` initialisation; the seed is explicit and stored in
checkpoint metadata.

The encoder is the target-leakage boundary. It reads only named features; an
extra key such as `leaked_target` is ignored unless it is explicitly listed in
the feature contract. `assert_no_target_in_features` additionally rejects the
common bug of appending the target column to the matrix.

## Prediction and revision

A prediction record stores the prediction, actual outcome (if known), error,
checkpoint id, model version, features and timestamp.

Revision is validation-gated:

1. Snapshot predictor parameters and encoder state.
2. Measure validation MSE before the candidate update.
3. Warm-start training on new (optionally replayed) experience for a bounded
   number of epochs.
4. Measure validation MSE after.
5. Accept only if it improves by `min_improvement`; otherwise restore the
   snapshot.

A single prediction error is never treated as proof that the representation is
wrong. `ReplayBuffer` implements deterministic reservoir sampling for optional
consolidation; replay is exploratory in the legacy E003 and does not affect its
gated criterion.

The v2 retention path adds two layers without changing the legacy controller:

1. `EvidenceValidator` computes residual median/MAD z-scores with the current
   learner and rejects inconsistent records or batches that are too small to
   evaluate.
2. `RetentionRevisionController` gates on new-task improvement **and** an
   old-task retention budget (`max(before * ratio, floor_fraction * variance)`).
   Rejected candidates are rolled back and every attempt is appended to a
   structured `history` payload.

E005 additionally exposes per-hidden-layer activations, temporary unit ablation
via a forward hook, and linear CKA. These are measurement tools, not a claim
that units are interpretable symbols.

## Checkpoints

A checkpoint is one `.pt` payload plus a `.json` sidecar:

- `metadata`: schema version, format, Genesis version, checkpoint id, timestamp,
  seed, full learner config, feature names, encoder state, target statistics,
  training summary, metrics and tags.
- `state_dict`: parameter tensors.

Save is atomic (temporary file then replace) and refuses to overwrite an
existing path. Load validates `format` and `schema_version` before use.
`checkpoints_equivalent` and the default restore test use an explicit `1e-6`
absolute tolerance.

## Storage

`Repository` is storage-agnostic and has no delete operations for experiments or
checkpoints. `SQLiteRepository` uses a single connection plus an `RLock`, so
API request threads and background experiment workers can share it safely.
Tables: `experiences`, `experiments`, `metrics`, `checkpoints`, `revisions`,
`logs`. JSON columns preserve the full payload. A PostgreSQL implementation can
satisfy the same contract later without changing the learning algorithm.

The JSONL event log is separate and append-only. Each append is flushed and
fsynced; sequence numbers are monotonic.

## API

`create_app(settings)` is side-effect-free at import. `services.api.app:app` is
the ASGI entry point that builds the default app. Long experiments run on daemon
threads and persist `running` -> `completed`/`failed` transitions, including the
error text. Checkpoint ids are unique; restoration loads and validates before
replacing the active learner. `GET /experiments/{id}/revisions` and
`GET /revisions` expose the structured revision payloads written by E003 and
E003-RETENTION.

## Reproducibility contract

- Configs are Pydantic models with `extra="forbid"`.
- Seeds are explicit fields and are stored in results/checkpoints.
- Environment draws use `numpy.random.default_rng(seed)`.
- Neural training uses `torch.manual_seed` plus a local CPU generator for batch
  shuffling.
- Splits are deterministic index permutations.
- Experiment runners are pure functions returning JSON-serialisable dicts.
- Success criteria are embedded in results before inspection.

Cross-platform bitwise equality is not claimed; within the recorded environment
and dependency versions, repeated runs are identical and checkpoint restore is
verified numerically.

## Extension boundaries

- **PostgreSQL**: implement `Repository`; no learner change.
- **Rust runtime**: the learner/environment interfaces are explicit data in /
  data out; a future Rust implementation must match the checkpoint schema.
- **Symbolic discovery**: the current `SymbolicRuleLearner` is a fixed-basis
  baseline. A future discovery engine needs its own protocol and must not
  receive a high-level concept catalogue.
- **Text**: the tokenizer is inspectable and non-pretrained. No semantic claims
  follow from tokenization or segmentation.
