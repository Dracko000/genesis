export type Health = {
  status: string;
  version: string;
  environment: string;
  database: string;
};

export type Observation = {
  id: string;
  features: Record<string, number>;
  outcome: number | null;
  action: number | null;
  text: string | null;
  provenance: { source: string; inferred: boolean; generator?: string | null };
  kind: "observation" | "inference";
  timestamp: string;
};

export type LearningStepResponse = {
  experiment_id: string;
  status: string;
  metrics: Record<string, number>;
  checkpoint_id: string | null;
  n_train: number;
  n_validation: number;
  learner: Record<string, unknown>;
};

export type PredictResponse = {
  prediction: number;
  actual: number | null;
  error: number | null;
  checkpoint_id: string | null;
  feature_names: string[];
};

export type ExperimentRecord = {
  id: string;
  name: string;
  status: string;
  config: Record<string, unknown>;
  seed: number | null;
  result: Record<string, unknown>;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type MetricRecord = {
  name: string;
  value: number;
  step: number | null;
  payload: Record<string, unknown>;
};

export type RevisionRecord = {
  id: number;
  experiment_id: string | null;
  payload: Record<string, unknown>;
  created_at: string;
};

export type CheckpointRecord = {
  checkpoint_id: string;
  path: string;
  metadata: Record<string, unknown>;
  created_at: string;
};

export type LogRecord = {
  id: number;
  level: string;
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
};

export type ResearchLog = {
  markdown: string;
  logs: LogRecord[];
};
