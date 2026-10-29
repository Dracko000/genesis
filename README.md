# Genesis

Genesis is an experimental research prototype for studying how internal
representations and generalisation can develop from structured experience. It
is **not** a chatbot, and it does **not** claim human-like understanding. Every
claim in this repository must be backed by a reproducible run whose results are
inspected and logged.

Current status: **v0.1 research prototype**. Phases 1–6 are implemented at a
small scale, including neural baselines, SQLite persistence, a FastAPI service,
a real Vite/React dashboard, E001/E002/E003, and a symbolic/hybrid ablation.

## What is implemented

- Deterministic synthetic worlds with hidden rules (`linear`, `interaction`,
  `trigonometric`, `quadratic`, `changed`), controlled label noise, IID and
  compositional splits, and renamed-feature probes.
- Validated experience records that keep observations distinct from inferred
  explanations and keep targets out of input features.
- A minimal character tokenizer, vocabulary and segmentation layer. No
  pretrained language model and no pretrained semantic embeddings.
- A configurable PyTorch MLP baseline with explicit seeds, saveable parameters,
  checkpoint restore, validation-gated revision and replay consolidation.
- A symbolic-basis baseline and a validation-selected hybrid, explicitly
  documented as having a fixed low-level operator prior.
- Baselines: training-mean, 1-nearest-neighbour, closed-form ridge, and a
  ground-truth oracle used only as an upper reference.
- SQLite repository, append-only JSONL event log, FastAPI endpoints, and a
  research dashboard that renders actual API results.

## Requirements

- Python 3.12+ (verified on CPython 3.14.5)
- NumPy, PyTorch (CPU), Pydantic v2, FastAPI, pytest
- Node.js 20+ and npm for the dashboard

Dependencies are bounded in `pyproject.toml`; no version is unpinned.

## Setup

```bash
# from the repository root
py -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:  source .venv/bin/activate
py -m pip install -e ".[dev]"
cp .env.example .env   # optional; defaults work without it
```

On Windows this project uses the `py` launcher (the `python`/`python3` Store
stubs do not run). Run commands with `py`.

## Tests

```bash
py -m pytest                 # fast default suite: 56 passed (2026-10-29)
py -m pytest -m slow         # reserved for longer research runs (none tagged yet)
```

The default suite covers schema validation, deterministic environments, learner
training/prediction, checkpoint round-trips, target-leakage guards, revision and
rollback, SQLite persistence, API validation, background experiment execution,
and experiment reproducibility.

## Run the API

```bash
py -m uvicorn services.api.app:app --reload --port 8000
```

Configuration is read from environment variables (see `.env.example`). The
OpenAPI docs are available at `http://127.0.0.1:8000/docs`.

Core endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/version` | Liveness and version |
| POST/GET | `/experiences` | Store and list experiences |
| POST | `/learning/step` | Train on stored labeled experiences, save a checkpoint |
| POST | `/predict` | Predict with the active learner or a checkpoint |
| POST | `/experiments` | Start E001/E002/E003 in a background worker |
| GET | `/experiments`, `/experiments/{id}` | Status and full result |
| GET | `/experiments/{id}/metrics` | Persisted metric rows |
| GET | `/checkpoints` | Checkpoint history |
| POST | `/checkpoints/{id}/restore` | Restore a checkpoint as active |
| GET | `/research-log` | `docs/research_log.md` plus runtime logs |

Checkpoints are never overwritten silently: a duplicate id is rejected with
`409`. There are no delete endpoints for experiments or checkpoints.

## Run the dashboard

```bash
cd apps/research-ui
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` to `http://127.0.0.1:8000`.
Set `VITE_API_BASE` to point at a different API origin. The dashboard shows the
experience form/history, real learning-step metrics, predictions, experiment
status/results, checkpoints, and the research log. No metrics are mocked.

Build check: `npm run build`.

## Run experiments

```bash
py scripts/run_experiment.py E001
py scripts/run_experiment.py E002
py scripts/run_experiment.py E003
py scripts/run_experiment.py HYBRID
```

Optional flags: `--seeds 0,1,2`, `--epochs 300`, `--config configs/E001.json`,
`--output path.json`. Results are written under
`experiments/<name>/results_seeds_<seeds>.json`. The same config and software
produce the same result dictionary.

Fast end-to-end smoke test:

```bash
py scripts/train_demo.py
```

## Actual measured results (2026-10-29)

Full JSON files are in `experiments/`. All numbers are held-out, three seeds
(0,1,2), mean ± standard deviation.

**E001 — emergent structure** (`interaction` rule, held-out sign combination
`(+,+)`, 400 train / 200 held-out test):

| Method | MSE | R² |
| --- | --- | --- |
| Neural MLP | **0.0405 ± 0.0091** | **0.788** |
| Nearest neighbour (memorisation) | 0.2210 ± 0.0245 | -0.165 |
| Ridge (linear) | 1.1280 ± 0.1468 | -4.940 |
| Training mean | 0.3465 ± 0.0182 | -0.833 |
| Oracle (true rule) | 0.0000 | 1.000 |

The pre-declared criterion (≥10% lower MSE than nearest-neighbour and R² ≥ 0.5)
was met: 81.7% relative improvement. Sample efficiency was flat-to-improving
between 50 and 200 training examples (MSE 0.0497 → 0.0469 → 0.0570). Under
label noise σ=0.1 neural MSE was 0.0828 vs oracle noise floor 0.010; at σ=0.3
neural MSE 0.5862 was **worse** than nearest-neighbour 0.3572. Improved
prediction is not evidence of human-like concept formation.

**E002 — structure reuse: NEGATIVE RESULT.** Transfer from a `linear` source to
an `interaction` target made target performance worse than an equal-budget
fresh initialisation:

| Condition | Target MSE |
| --- | --- |
| Transfer, 25 target epochs | 0.010987 ± 0.006544 |
| Fresh, 25 target epochs | **0.003650 ± 0.000890** |
| Fresh, 250 target epochs | 0.001997 ± 0.000587 |
| Unrelated prior, 25 target epochs | 0.002547 ± 0.000615 |

Source retention after transfer collapsed from MSE 0.000017 to 0.632762. The
pre-declared transfer criterion was **not met**. This is logged as a negative
result, not hidden or reinterpreted.

**E003 — representation revision.** After changing the rule, the validation-
gated update reduced new-task MSE from 0.535701 ± 0.010653 to **0.002354 ±
0.000447** (99.6% relative improvement) and rejected an isolated anomalous
observation in all seeds. Retention was poor: old-task MSE rose from 0.000326
to 0.547721 (catastrophic forgetting). An exploratory replay variant improved
old-task MSE to 0.038171 ± 0.008553 while leaving new-task MSE at 0.342718 ±
0.025429 — a clear adaptation/retention trade-off, not a free win.

**Hybrid ablation.** The symbolic basis reached machine-precision MSE because
its declared basis contains the true terms (`f1*f2`, `f3`); the neural-only MSE
was 0.000649 ± 0.000185. The validation-selected hybrid chose symbolic in all
seeds and matched it. This **does not** show hybrid superiority: the symbolic
baseline has an oracle-like prior for this rule. The criterion was met, but the
result is confounded and is recorded as such.

## Repository layout

```
genesis/                # learning engine package
  language/             # character tokenizer, vocabulary, segmentation
  experience/           # schemas, provenance, append-only event log
  environments/         # synthetic world and hidden rules
  representation/       # feature encoder, latent state, checkpoints
  learning/             # neural baseline, revision, consolidation, symbolic
  evaluation/           # datasets, metrics, baselines, experiment runners
  storage/              # repository interface + SQLite implementation
services/api/           # FastAPI app, schemas, background experiment workers
apps/research-ui/       # Vite + React + TypeScript dashboard
experiments/            # protocols and generated result JSON
tests/                  # fast automated suite
scripts/                # experiment CLI and demo
docs/                   # architecture, hypotheses, protocols, research log
configs/                # reusable experiment configuration overrides
```

## Reproducibility and scientific rules

- Every experiment is a pure function of a validated config plus seeds.
- Inputs are generated from `numpy.random.default_rng(seed)`; neural training
  uses explicit PyTorch seeds and generators.
- Encoder normalisation is fitted on training data only. Targets are passed as
  a separate vector and are never read by the encoder.
- Checkpoints embed parameters, encoder state, model config, seed and metadata
  in one file plus a JSON sidecar; restore is verified within `1e-6`.
- Success criteria are declared in the protocol and in the result payload
  before results are inspected. Negative results stay in the research log.
- No pretrained LLM, vector database, knowledge graph, microservice, or Rust
  subsystem has been added.

## Known limitations

- Low-dimensional synthetic worlds only; no real sensory or language data.
- Compositional generalisation is operationalised as held-out sign quadrants.
- E002's negative transfer is one rule pair and one budget schedule; it may be
  an optimisation artefact as much as a representation result.
- E003 retention is poor without replay; replay is exploratory and not tuned.
- The hybrid's symbolic basis is a fixed prior and therefore not a fair test of
  emergent symbols.
- Numerical determinism is guaranteed for the recorded environment (CPU,
  same dependency versions); cross-platform bitwise equality is not claimed.
- The API has no authentication and is intended for local research use only.

## Unresolved questions

- Is the E002 negative transfer caused by target-statistic mismatch, output-
  layer anchoring, or genuinely unhelpful features? An encoder-only transfer
  ablation and a warm-up schedule sweep are the next tests.
- Can replay be scheduled to retain old tasks without slowing adaptation
  (e.g. gradient projection or elastic weight consolidation)?
- Do held-out-quadrant gains survive on higher-dimensional or non-stationary
  environments?

## Recommended next experiment

Run an E002 follow-up with three diagnostics under the same seeds and budgets:
(a) encoder-only transfer with a reinitialised predictor, (b) transfer with
source target-statistics reset, and (c) a warm-up schedule sweep. The current
evidence does not support the structure-reuse hypothesis, and the next run
should isolate whether that is an optimisation effect or a representation
effect.
