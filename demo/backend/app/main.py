from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from app.scraper import scrape_article
from app.samples import get_sample, get_random_sample, get_total_samples, load_data
from app.models import load_models, get_available_models, generate_all
from app.metrics import compute_all_metrics

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_data()
    loaded = load_models()
    print(f"[STARTUP] Samples loaded. Models: {loaded}")
    yield
    print("[SHUTDOWN] Server stopped.")

app = FastAPI(
    title="Vietnamese Text Summarization Demo",
    description="Compare BARTpho (FFT/LoRA) and Qwen2.5 (LoRA) on Vietnamese news articles",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScrapeRequest(BaseModel):
    url: HttpUrl

class SummarizeRequest(BaseModel):
    text: str
    reference: str = ""

class MetricsRequest(BaseModel):
    predictions: dict
    reference: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/scrape")
def scrape(req: ScrapeRequest):
    result = scrape_article(str(req.url))
    if not result.get("text"):
        raise HTTPException(status_code=400, detail="Cannot extract content from this URL.")
    return result

@app.get("/api/samples/total")
def total_samples():
    return {"total": get_total_samples()}

@app.get("/api/samples/random")
def random_sample():
    sample = get_random_sample()
    if sample is None:
        raise HTTPException(status_code=404, detail="No samples available.")
    return sample

@app.get("/api/samples/{index}")
def sample_by_index(index: int):
    result = get_sample(index)
    if result is None:
        raise HTTPException(status_code=404, detail="Index out of range.")
    return result

@app.get("/api/models")
def list_models():
    return {"models": get_available_models()}

@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    available = get_available_models()
    if not available:
        raise HTTPException(status_code=503, detail="No models loaded. Place checkpoints in models/ directory.")
    predictions = generate_all(req.text)
    metrics = {}
    if req.reference.strip():
        metrics = compute_all_metrics(predictions, req.reference)
    return {"predictions": predictions, "metrics": metrics}

@app.post("/api/metrics")
def calculate_metrics(req: MetricsRequest):
    return compute_all_metrics(req.predictions, req.reference)
