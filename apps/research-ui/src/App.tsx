import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type {
  CheckpointRecord,
  ExperimentRecord,
  Health,
  LearningStepResponse,
  MetricRecord,
  Observation,
  PredictResponse,
  ResearchLog,
  RevisionRecord,
} from "./types";

type Tab = "experiences" | "learning" | "predict" | "experiments" | "checkpoints" | "log";

const TABS: { id: Tab; label: string }[] = [
  { id: "experiences", label: "Experiences" },
  { id: "learning", label: "Learning step" },
  { id: "predict", label: "Predict" },
  { id: "experiments", label: "Experiments" },
  { id: "checkpoints", label: "Checkpoints" },
  { id: "log", label: "Research log" },
];

const DEFAULT_CONFIGS: Record<string, Record<string, unknown>> = {
  E001: {
    seeds: [0, 1, 2],
    n_train: 400,
    n_test: 200,
    sample_sizes: [50, 100, 200],
    noise_levels: [0, 0.1, 0.3],
    run_noise_study: true,
    neural: { epochs: 300, hidden_sizes: [32, 32], seed: 0 },
  },
  E002: {
    seeds: [0, 1, 2],
    n_source: 500,
    n_target: 120,
    n_target_test: 300,
    warm_epochs: 25,
    full_epochs: 250,
    neural: { epochs: 250, hidden_sizes: [32, 32], seed: 0 },
  },
  E003: {
    seeds: [0, 1, 2],
    n_train: 400,
    n_new: 120,
    n_new_val: 100,
    n_test: 300,
    adaptation_epochs: [5, 10, 25, 50],
    neural: { epochs: 300, hidden_sizes: [32, 32], seed: 0 },
  },
  "E002-DIAG": {
    pairs: [
      {
        label: "additive_related",
        source_rule: "linear",
        target_rule: "linear_alt",
        related: true,
      },
      {
        label: "interaction_related",
        source_rule: "interaction",
        target_rule: "interaction_alt",
        related: true,
      },
      {
        label: "legacy_linear_to_interaction",
        source_rule: "linear",
        target_rule: "interaction",
        related: false,
      },
      {
        label: "unrelated_linear_to_quadratic",
        source_rule: "linear",
        target_rule: "quadratic",
        related: false,
      },
    ],
    seeds: [0, 1, 2],
    n_source: 500,
    n_target: 120,
    n_target_test: 300,
    source_epochs: 250,
    target_epochs: 50,
    full_target_epochs: 250,
    neural: { epochs: 250, hidden_sizes: [32, 32], seed: 0 },
  },
  "E003-RETENTION": {
    seeds: [0, 1, 2],
    n_train: 400,
    n_new: 120,
    n_new_val: 100,
    n_test: 300,
    adaptation_epochs: [5, 10, 25, 50],
    neural: { epochs: 300, hidden_sizes: [32, 32], seed: 0 },
    min_new_task_improvement: 1e-6,
    old_task_max_ratio: 4,
    old_task_mse_floor_fraction: 0.2,
    min_accepted_fraction: 0.5,
    max_residual_z: 6,
    min_batch_size: 3,
    success_new_task_improvement: 0.1,
  },
  E005: {
    seeds: [0, 1, 2],
    source_rule: "linear",
    target_rule: "linear_alt",
    n_source: 400,
    n_target: 120,
    n_target_test: 200,
    source_epochs: 200,
    target_epochs: 50,
    ablation_fraction: 0.2,
    neural: { epochs: 300, hidden_sizes: [32, 32], seed: 0 },
    success_selectivity_ratio: 1.5,
  },
};

function parseFeatures(raw: string): Record<string, number> {
  const parsed: unknown = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("features must be a JSON object");
  }
  const features: Record<string, number> = {};
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`feature ${key} must be a finite number`);
    }
    features[key] = value;
  }
  return features;
}

function Json({ value }: { value: unknown }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}

function numberOrNull(raw: string): number | null {
  if (!raw.trim()) return null;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error("value must be a finite number");
  return value;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("experiences");
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const [experiences, setExperiences] = useState<Observation[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([]);
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);

  const [featureJson, setFeatureJson] = useState('{"f1": 0.1, "f2": -0.2, "f3": 0.3}');
  const [outcome, setOutcome] = useState("1.0");
  const [action, setAction] = useState("");

  const [featureNames, setFeatureNames] = useState("f1,f2,f3");
  const [epochs, setEpochs] = useState("300");
  const [hidden, setHidden] = useState("32,32");
  const [learningRate, setLearningRate] = useState("0.01");
  const [validationFraction, setValidationFraction] = useState("0.2");
  const [stepResult, setStepResult] = useState<LearningStepResponse | null>(null);

  const [predictJson, setPredictJson] = useState('{"f1": 0.1, "f2": -0.2, "f3": 0.3}');
  const [predictActual, setPredictActual] = useState("");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  const [experimentKind, setExperimentKind] = useState("E001");
  const [experimentConfig, setExperimentConfig] = useState(
    JSON.stringify(DEFAULT_CONFIGS.E001, null, 2),
  );
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<MetricRecord[]>([]);
  const [selectedRevisions, setSelectedRevisions] = useState<RevisionRecord[]>([]);

  const [researchLog, setResearchLog] = useState<ResearchLog | null>(null);
  const [compareA, setCompareA] = useState("");
  const [compareB, setCompareB] = useState("");
  const [busy, setBusy] = useState(false);

  const selected = useMemo(
    () => experiments.find((record) => record.id === selectedExperiment) ?? null,
    [experiments, selectedExperiment],
  );

  function begin() {
    setBusy(true);
    setError(null);
    setMessage(null);
  }

  function fail(err: unknown) {
    setError(err instanceof Error ? err.message : String(err));
  }

  async function loadExperiences() {
    try {
      setExperiences(await api.listExperiences());
    } catch (err) {
      fail(err);
    }
  }

  async function loadCheckpoints() {
    try {
      setCheckpoints(await api.listCheckpoints());
    } catch (err) {
      fail(err);
    }
  }

  async function loadExperiments() {
    try {
      setExperiments(await api.listExperiments());
    } catch (err) {
      fail(err);
    }
  }

  async function loadResearchLog() {
    try {
      setResearchLog(await api.researchLog());
    } catch (err) {
      fail(err);
    }
  }

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(fail);
    void loadExperiences();
    void loadCheckpoints();
    void loadExperiments();
  }, []);

  useEffect(() => {
    if (!experiments.some((record) => record.status === "running")) return;
    const timer = setInterval(() => {
      void loadExperiments();
    }, 2000);
    return () => clearInterval(timer);
  }, [experiments]);

  useEffect(() => {
    if (!selectedExperiment) {
      setSelectedMetrics([]);
      setSelectedRevisions([]);
      return;
    }
    api
      .getMetrics(selectedExperiment)
      .then(setSelectedMetrics)
      .catch(fail);
    api
      .getRevisions(selectedExperiment)
      .then(setSelectedRevisions)
      .catch(fail);
  }, [selectedExperiment]);

  useEffect(() => {
    if (tab === "log") void loadResearchLog();
  }, [tab]);

  async function submitExperience(event: React.FormEvent) {
    event.preventDefault();
    begin();
    try {
      const created = await api.createExperience({
        features: parseFeatures(featureJson),
        outcome: numberOrNull(outcome),
        action: numberOrNull(action),
      });
      setMessage(`Stored experience ${created.id}`);
      await loadExperiences();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function runLearningStep() {
    begin();
    try {
      const neural = {
        epochs: Number(epochs),
        hidden_sizes: hidden
          .split(",")
          .map((value) => Number(value.trim()))
          .filter((value) => Number.isFinite(value)),
        learning_rate: Number(learningRate),
        seed: 7,
      };
      const result = await api.learningStep({
        experiment_name: "dashboard_learning_step",
        feature_names: featureNames
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean),
        neural,
        validation_fraction: Number(validationFraction),
      });
      setStepResult(result);
      setMessage(`Learning step completed (${result.experiment_id.slice(0, 8)})`);
      await Promise.all([loadCheckpoints(), loadExperiments()]);
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function runPrediction(event: React.FormEvent) {
    event.preventDefault();
    begin();
    try {
      const result = await api.predict({
        features: parseFeatures(predictJson),
        actual: numberOrNull(predictActual),
      });
      setPrediction(result);
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function startExperiment() {
    begin();
    try {
      const config = JSON.parse(experimentConfig) as Record<string, unknown>;
      const record = await api.startExperiment(experimentKind, config);
      setSelectedExperiment(record.id);
      setMessage(`${experimentKind} started (${record.id.slice(0, 8)})`);
      await loadExperiments();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  async function restoreCheckpoint(id: string) {
    begin();
    try {
      await api.restoreCheckpoint(id);
      setMessage(`Checkpoint ${id.slice(0, 8)} restored`);
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  }

  function selectExperimentKind(kind: string) {
    setExperimentKind(kind);
    setExperimentConfig(JSON.stringify(DEFAULT_CONFIGS[kind] ?? {}, null, 2));
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Genesis</h1>
          <p className="subtitle">Experience-driven representation research dashboard</p>
        </div>
        <div className="status">
          {health ? (
            <>
              <span className="badge ok">API {health.status}</span>
              <span className="badge">v{health.version}</span>
              <span className="badge">{health.environment}</span>
            </>
          ) : (
            <span className="badge bad">API unreachable</span>
          )}
        </div>
      </header>

      {error && (
        <div className="banner error">
          <strong>Error:</strong> {error}
          <button onClick={() => setError(null)}>dismiss</button>
        </div>
      )}
      {message && (
        <div className="banner info">
          {message}
          <button onClick={() => setMessage(null)}>dismiss</button>
        </div>
      )}

      <nav className="tabs">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            className={tab === entry.id ? "active" : ""}
            onClick={() => setTab(entry.id)}
          >
            {entry.label}
          </button>
        ))}
      </nav>

      {tab === "experiences" && (
        <div className="grid two">
          <section className="card">
            <h2>Add experience</h2>
            <form onSubmit={submitExperience}>
              <label>
                Features (JSON)
                <textarea
                  rows={4}
                  value={featureJson}
                  onChange={(event) => setFeatureJson(event.target.value)}
                />
              </label>
              <label>
                Outcome (optional)
                <input value={outcome} onChange={(event) => setOutcome(event.target.value)} />
              </label>
              <label>
                Action (optional)
                <input value={action} onChange={(event) => setAction(event.target.value)} />
              </label>
              <button type="submit" disabled={busy}>
                Store experience
              </button>
            </form>
            <p className="hint">
              Targets are stored separately from features. Inferred records are rejected unless
              provenance marks them as inferred.
            </p>
          </section>

          <section className="card">
            <h2>Recent experiences ({experiences.length})</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Features</th>
                    <th>Outcome</th>
                    <th>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {experiences.map((record) => (
                    <tr key={record.id}>
                      <td className="mono">{record.id.slice(0, 8)}</td>
                      <td className="mono">{JSON.stringify(record.features)}</td>
                      <td>{record.outcome ?? "—"}</td>
                      <td>{record.provenance.source}</td>
                    </tr>
                  ))}
                  {experiences.length === 0 && (
                    <tr>
                      <td colSpan={4} className="empty">
                        No experiences yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      )}

      {tab === "learning" && (
        <div className="grid two">
          <section className="card">
            <h2>Run a learning step</h2>
            <label>
              Feature names (comma separated)
              <input value={featureNames} onChange={(event) => setFeatureNames(event.target.value)} />
            </label>
            <div className="row">
              <label>
                Epochs
                <input value={epochs} onChange={(event) => setEpochs(event.target.value)} />
              </label>
              <label>
                Hidden sizes
                <input value={hidden} onChange={(event) => setHidden(event.target.value)} />
              </label>
              <label>
                Learning rate
                <input
                  value={learningRate}
                  onChange={(event) => setLearningRate(event.target.value)}
                />
              </label>
              <label>
                Validation fraction
                <input
                  value={validationFraction}
                  onChange={(event) => setValidationFraction(event.target.value)}
                />
              </label>
            </div>
            <button onClick={runLearningStep} disabled={busy}>
              Train on stored labeled experiences
            </button>
            <p className="hint">
              Trains the neural baseline on real stored experiences, saves a checkpoint, and
              makes the learner active for predictions.
            </p>
          </section>

          <section className="card">
            <h2>Last learning step</h2>
            {stepResult ? <Json value={stepResult} /> : <p className="empty">Not run yet.</p>}
          </section>
        </div>
      )}

      {tab === "predict" && (
        <div className="grid two">
          <section className="card">
            <h2>Request prediction</h2>
            <form onSubmit={runPrediction}>
              <label>
                Features (JSON)
                <textarea
                  rows={4}
                  value={predictJson}
                  onChange={(event) => setPredictJson(event.target.value)}
                />
              </label>
              <label>
                Observed outcome (optional)
                <input
                  value={predictActual}
                  onChange={(event) => setPredictActual(event.target.value)}
                />
              </label>
              <button type="submit" disabled={busy}>
                Predict
              </button>
            </form>
            <p className="hint">
              Uses the active learner; run a learning step first or restore a checkpoint.
            </p>
          </section>
          <section className="card">
            <h2>Prediction</h2>
            {prediction ? (
              <div className="kv">
                <div>
                  <span>Prediction</span>
                  <strong>{prediction.prediction.toFixed(6)}</strong>
                </div>
                <div>
                  <span>Actual</span>
                  <strong>{prediction.actual ?? "—"}</strong>
                </div>
                <div>
                  <span>Error</span>
                  <strong>{prediction.error ?? "—"}</strong>
                </div>
                <div>
                  <span>Checkpoint</span>
                  <strong className="mono">{prediction.checkpoint_id ?? "active"}</strong>
                </div>
              </div>
            ) : (
              <p className="empty">No prediction yet.</p>
            )}
          </section>
        </div>
      )}

      {tab === "experiments" && (
        <div className="grid two wide-right">
          <section className="card">
            <h2>Start experiment</h2>
            <label>
              Experiment
              <select
                value={experimentKind}
                onChange={(event) => selectExperimentKind(event.target.value)}
              >
                <option value="E001">E001 — Emergent structure</option>
                <option value="E002">E002 — Structure reuse</option>
                <option value="E002-DIAG">E002-v2 — Fair transfer diagnostics</option>
                <option value="E003">E003 — Representation revision</option>
                <option value="E003-RETENTION">
                  E003-v2 — Dual-gated retention
                </option>
                <option value="E005">E005 — Representation interventions</option>
              </select>
            </label>
            <label>
              Configuration (JSON)
              <textarea
                rows={16}
                value={experimentConfig}
                onChange={(event) => setExperimentConfig(event.target.value)}
              />
            </label>
            <button onClick={startExperiment} disabled={busy}>
              Run experiment
            </button>
            <p className="hint">
              Experiments run in a background worker. Results and declared success criteria are
              stored verbatim; nothing is mocked.
            </p>
          </section>

          <section className="card">
            <h2>Experiment history</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Kind</th>
                    <th>Status</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((record) => (
                    <tr
                      key={record.id}
                      className={record.id === selectedExperiment ? "selected" : ""}
                      onClick={() => setSelectedExperiment(record.id)}
                    >
                      <td className="mono">{record.id.slice(0, 8)}</td>
                      <td>{record.name}</td>
                      <td>
                        <span className={`badge ${record.status}`}>{record.status}</span>
                      </td>
                      <td>{new Date(record.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
                  {experiments.length === 0 && (
                    <tr>
                      <td colSpan={4} className="empty">
                        No experiments yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {selected && (
              <div className="detail">
                <h3>
                  {selected.name} · {selected.status}
                </h3>
                {selected.error && <div className="banner error">{selected.error}</div>}
                {selected.status === "completed" && (
                  <>
                    <h4>Summary</h4>
                    <Json value={selected.result.summary ?? selected.result} />
                    {"criterion" in selected.result && (
                      <>
                        <h4>Declared criterion</h4>
                        <Json value={selected.result.criterion} />
                      </>
                    )}
                    <h4>Metrics</h4>
                    {selectedMetrics.length ? (
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Name</th>
                              <th>Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {selectedMetrics.map((metric, index) => (
                              <tr key={`${metric.name}-${index}`}>
                                <td className="mono">{metric.name}</td>
                                <td>{metric.value.toFixed(6)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className="empty">No metric rows.</p>
                    )}
                    <h4>Revision history ({selectedRevisions.length})</h4>
                    {selectedRevisions.length ? (
                      <div className="revisions">
                        {selectedRevisions.map((revision) => {
                          const payload = revision.payload as {
                            seed?: number;
                            accepted?: boolean;
                            update_accepted?: boolean;
                          };
                          const accepted = payload.accepted ?? payload.update_accepted;
                          return (
                            <details key={revision.id}>
                              <summary>
                                revision #{revision.id} · seed {payload.seed ?? "—"} ·{" "}
                                {accepted === undefined
                                  ? "candidate"
                                  : accepted
                                    ? "accepted"
                                    : "rejected"}
                              </summary>
                              <Json value={revision.payload} />
                            </details>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="empty">No revision rows.</p>
                    )}
                  </>
                )}
              </div>
            )}
          </section>
        </div>
      )}

      {tab === "checkpoints" && (
        <section className="card">
          <h2>Checkpoints ({checkpoints.length})</h2>
          {checkpoints.length > 0 && (
            <div className="compare">
              <div className="row two-col">
                <label>
                  Compare checkpoint A
                  <select value={compareA} onChange={(event) => setCompareA(event.target.value)}>
                    <option value="">(first)</option>
                    {checkpoints.map((checkpoint) => (
                      <option key={checkpoint.checkpoint_id} value={checkpoint.checkpoint_id}>
                        {checkpoint.checkpoint_id.slice(0, 8)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Compare checkpoint B
                  <select value={compareB} onChange={(event) => setCompareB(event.target.value)}>
                    <option value="">(second)</option>
                    {checkpoints.map((checkpoint) => (
                      <option key={checkpoint.checkpoint_id} value={checkpoint.checkpoint_id}>
                        {checkpoint.checkpoint_id.slice(0, 8)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="grid two">
                <div>
                  <h4>A</h4>
                  <Json
                    value={
                      checkpoints.find((item) => item.checkpoint_id === compareA)?.metadata ??
                      checkpoints[0].metadata
                    }
                  />
                </div>
                <div>
                  <h4>B</h4>
                  <Json
                    value={
                      checkpoints.find((item) => item.checkpoint_id === compareB)?.metadata ??
                      (checkpoints[1] ?? checkpoints[0]).metadata
                    }
                  />
                </div>
              </div>
            </div>
          )}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Seed</th>
                  <th>Features</th>
                  <th>Metrics</th>
                  <th>Created</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {checkpoints.map((checkpoint) => {
                  const metadata = checkpoint.metadata as {
                    seed?: number;
                    feature_names?: string[];
                    metrics?: Record<string, number>;
                  };
                  return (
                    <tr key={checkpoint.checkpoint_id}>
                      <td className="mono">{checkpoint.checkpoint_id.slice(0, 8)}</td>
                      <td>{metadata.seed ?? "—"}</td>
                      <td className="mono">{(metadata.feature_names ?? []).join(", ")}</td>
                      <td className="mono">{JSON.stringify(metadata.metrics ?? {})}</td>
                      <td>{new Date(checkpoint.created_at).toLocaleString()}</td>
                      <td>
                        <button
                          className="small"
                          onClick={() => restoreCheckpoint(checkpoint.checkpoint_id)}
                          disabled={busy}
                        >
                          Restore
                        </button>
                      </td>
                    </tr>
                  );
                })}
                {checkpoints.length === 0 && (
                  <tr>
                    <td colSpan={6} className="empty">
                      No checkpoints yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {tab === "log" && (
        <div className="grid two">
          <section className="card">
            <h2>research_log.md</h2>
            {researchLog ? (
              <pre className="markdown">{researchLog.markdown || "No research log found."}</pre>
            ) : (
              <p className="empty">Loading…</p>
            )}
          </section>
          <section className="card">
            <h2>Runtime logs</h2>
            {researchLog?.logs.length ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Level</th>
                      <th>Message</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {researchLog.logs.map((entry) => (
                      <tr key={entry.id}>
                        <td>
                          <span className={`badge ${entry.level.toLowerCase()}`}>{entry.level}</span>
                        </td>
                        <td>{entry.message}</td>
                        <td>{new Date(entry.created_at).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="empty">No runtime logs.</p>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
