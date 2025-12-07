export interface Sample {
  id: string;
  filepath: string;
  filename: string;
  label: string | null;
  thumbnail: string | null;
  metadata: Record<string, unknown>;
  embedding_2d?: [number, number];
  embedding_2d_hyperbolic?: [number, number];
}

export interface DatasetInfo {
  name: string;
  num_samples: number;
  labels: string[];
  label_colors: Record<string, string>;
}

export interface EmbeddingsData {
  ids: string[];
  labels: (string | null)[];
  euclidean: [number, number][];
  hyperbolic: [number, number][];
  label_colors: Record<string, string>;
}

export interface SamplesResponse {
  total: number;
  offset: number;
  limit: number;
  samples: Sample[];
}

export type ViewMode = "euclidean" | "hyperbolic";

export interface EmbeddingModelInfo {
  model_id: string;
  display_name: string;
  description: string;
}

export interface EmbeddingModelsResponse {
  models: Record<string, EmbeddingModelInfo>;
  current_model: string;
  data_type: string;
}
