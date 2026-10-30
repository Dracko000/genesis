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

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // keep status text
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  listExperiences: async () => {
    const data = await request<{ experiences: Observation[] }>("/experiences?limit=200");
    return data.experiences;
  },
  createExperience: (payload: {
    features: Record<string, number>;
    outcome?: number | null;
    action?: number | null;
  }) =>
    request<Observation>("/experiences", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  learningStep: (payload: {
    experiment_name: string;
    feature_names: string[];
    neural: Record<string, unknown>;
    validation_fraction: number;
  }) =>
    request<LearningStepResponse>("/learning/step", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  predict: (payload: {
    features: Record<string, number>;
    actual?: number | null;
    checkpoint_id?: string | null;
  }) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listExperiments: async () => {
    const data = await request<{ experiments: ExperimentRecord[] }>("/experiments?limit=100");
    return data.experiments;
  },
  getExperiment: (id: string) => request<ExperimentRecord>(`/experiments/${id}`),
  getMetrics: (id: string) => request<MetricRecord[]>(`/experiments/${id}/metrics`),
  getRevisions: async (id: string) => {
    const data = await request<{ revisions: RevisionRecord[] }>(
      `/experiments/${id}/revisions`,
    );
    return data.revisions;
  },
  startExperiment: (experiment: string, config: Record<string, unknown>) =>
    request<ExperimentRecord>("/experiments", {
      method: "POST",
      body: JSON.stringify({ experiment, config }),
    }),

  listCheckpoints: async () => {
    const data = await request<{ checkpoints: CheckpointRecord[] }>("/checkpoints?limit=100");
    return data.checkpoints;
  },
  restoreCheckpoint: (id: string) =>
    request<{ checkpoint_id: string; restored: boolean }>(`/checkpoints/${id}/restore`, {
      method: "POST",
    }),

  researchLog: () => request<ResearchLog>("/research-log"),
};
