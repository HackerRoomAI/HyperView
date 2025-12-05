import type { DatasetInfo, EmbeddingsData, Sample, SamplesResponse } from "@/types";

const API_BASE = process.env.NODE_ENV === "development" ? "http://127.0.0.1:5151" : "";

export async function fetchDataset(): Promise<DatasetInfo> {
  const res = await fetch(`${API_BASE}/api/dataset`);
  if (!res.ok) {
    throw new Error(`Failed to fetch dataset: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSamples(
  offset: number = 0,
  limit: number = 100,
  label?: string
): Promise<SamplesResponse> {
  const params = new URLSearchParams({
    offset: offset.toString(),
    limit: limit.toString(),
  });
  if (label) {
    params.set("label", label);
  }

  const res = await fetch(`${API_BASE}/api/samples?${params}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch samples: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchEmbeddings(): Promise<EmbeddingsData> {
  const res = await fetch(`${API_BASE}/api/embeddings`);
  if (!res.ok) {
    throw new Error(`Failed to fetch embeddings: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSample(sampleId: string): Promise<Sample> {
  const res = await fetch(`${API_BASE}/api/samples/${sampleId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sample: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSamplesBatch(sampleIds: string[]): Promise<Sample[]> {
  const res = await fetch(`${API_BASE}/api/samples/batch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sample_ids: sampleIds }),
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch samples batch: ${res.statusText}`);
  }
  const data = await res.json();
  return data.samples;
}
