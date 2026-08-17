const API_BASE = "http://localhost:8000";

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}

export async function getAvailableModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error("Failed to fetch models");
  return res.json();
}

export async function scrapeArticle(url) {
  const res = await fetch(`${API_BASE}/api/scrape`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to scrape article");
  }
  return res.json();
}

export async function getRandomSample() {
  const res = await fetch(`${API_BASE}/api/samples/random`);
  if (!res.ok) throw new Error("Failed to fetch sample");
  return res.json();
}

export async function getSample(index) {
  const res = await fetch(`${API_BASE}/api/samples/${index}`);
  if (!res.ok) throw new Error("Sample not found");
  return res.json();
}

export async function getTotalSamples() {
  const res = await fetch(`${API_BASE}/api/samples/total`);
  if (!res.ok) throw new Error("Failed to fetch total");
  return res.json();
}

export async function summarize(text, reference = "") {
  const res = await fetch(`${API_BASE}/api/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, reference }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Summarization failed");
  }
  return res.json();
}

export async function computeMetrics(predictions, reference) {
  const res = await fetch(`${API_BASE}/api/metrics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ predictions, reference }),
  });
  if (!res.ok) throw new Error("Metrics computation failed");
  return res.json();
}
