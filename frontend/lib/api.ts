const apiBaseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type PredictionResponse = {
  success: true;
  prediction: {
    class: string;
    confidence: number;
    confidences: Record<string, number>;
    info: { description: string; color: string };
  };
  images: { original: string; heatmap: string; overlay: string };
};

async function request(path: string, init?: RequestInit) {
  const response = await fetch(`${apiBaseUrl}${path}`, init);
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? "The analysis service could not process this request.");
  }
  return response.json();
}

export function predictImage(file: File) {
  const body = new FormData();
  body.append("file", file);
  return request("/api/predict", { method: "POST", body }) as Promise<PredictionResponse>;
}

export function checkApiHealth() {
  return request("/health") as Promise<{ status: string }>;
}
