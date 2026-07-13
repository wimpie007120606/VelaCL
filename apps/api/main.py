from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from velacl.trainer import load_checkpoint, predict
from velacl.utils import atomic_json

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(os.getenv("VELACL_REGISTRY", ROOT / "experiments/registry.json"))
RUNS_PATH = Path(os.getenv("VELACL_RUNS", ROOT / "experiments/runs"))
REQUESTS = Counter("velacl_inference_requests_total", "Inference requests", ["status"])
LATENCY = Histogram("velacl_inference_latency_seconds", "Inference latency")


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)
    model_version: str = "champion"


class AnnotationRequest(BaseModel):
    event_id: str
    action: Literal["accept", "correct", "uncertain", "exclude", "train"]
    corrected_intent: str | None = None


class RateLimiter:
    def __init__(self, requests: int = 60, window_seconds: int = 60):
        self.requests = requests
        self.window = window_seconds
        self.history: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        queue = self.history[key]
        while queue and now - queue[0] > self.window:
            queue.popleft()
        if len(queue) >= self.requests:
            return False
        queue.append(now)
        return True


def detect_language(text: str) -> str:
    lowered = f" {text.lower()} "
    markers = {
        "zu": [" ngi", " yami", " sawubona", " umuntu"],
        "xh": [" ndi", " yam", " molo", " umntu"],
        "st": [" ke ", " ya ", " motho", " dumela"],
        "af": [" ek ", " rekening", " asseblief", " iemand", " het "],
    }
    scores = {
        language: sum(marker in lowered for marker in words) for language, words in markers.items()
    }
    language = max(scores, key=scores.get)
    return language if scores[language] else "en"


def risk_warnings(text: str) -> list[str]:
    lowered = text.lower()
    warnings = []
    if any(word in lowered for word in ["pin", "password", "id number", "account number"]):
        warnings.append("possible_pii")
    if any(word in lowered for word in ["ignore instructions", "system prompt", "bypass"]):
        warnings.append("prompt_injection")
    return warnings


class ModelStore:
    def __init__(self):
        self.cache: dict[str, tuple] = {}

    def registry(self) -> dict:
        if not REGISTRY_PATH.exists():
            raise HTTPException(503, "model registry unavailable; run `make experiment`")
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def load(self, version: str):
        registry = self.registry()
        method = registry.get("aliases", {}).get(version, version)
        candidate = next(
            (item for item in registry.get("models", []) if item["method"] == method), None
        )
        if not candidate:
            raise HTTPException(404, f"unknown model version: {version}")
        checkpoint = str(
            ROOT / candidate["checkpoint"]
            if not Path(candidate["checkpoint"]).is_absolute()
            else candidate["checkpoint"]
        )
        if checkpoint not in self.cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, metadata = load_checkpoint(checkpoint, device)
            self.cache[checkpoint] = (model, metadata["labels"], device, method)
        return self.cache[checkpoint]


store = ModelStore()
limiter = RateLimiter(int(os.getenv("VELACL_RATE_LIMIT", "60")))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if REGISTRY_PATH.exists():
        try:
            store.load("champion")
        except (HTTPException, FileNotFoundError):
            pass
    yield


app = FastAPI(title="VelaCL API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("VELACL_DASHBOARD_ORIGIN", "http://localhost:3000")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith("/v1/predict"):
        key = request.client.host if request.client else "unknown"
        if not limiter.allow(key):
            REQUESTS.labels("rate_limited").inc()
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "registry": REGISTRY_PATH.exists(), "cuda": torch.cuda.is_available()}


@app.get("/ready")
def ready() -> dict:
    try:
        _model, _labels, _device, method = store.load("champion")
        return {"status": "ready", "model": method}
    except HTTPException as exc:
        raise HTTPException(503, str(exc.detail)) from exc


@app.post("/v1/predict")
def inference(payload: PredictRequest) -> dict:
    started = time.perf_counter()
    try:
        model, labels, device, method = store.load(payload.model_version)
        outputs = predict(model, payload.texts, labels, device)
        for output in outputs:
            output.update(
                {
                    "language": detect_language(output["text"]),
                    "entities": [],
                    "warnings": risk_warnings(output["text"]),
                }
            )
        REQUESTS.labels("success").inc()
        return {"model_version": method, "predictions": outputs}
    except Exception:
        REQUESTS.labels("error").inc()
        raise
    finally:
        LATENCY.observe(time.perf_counter() - started)


@app.post("/v1/predict/stream")
async def stream_inference(payload: PredictRequest):
    async def generate():
        result = inference(payload)
        for prediction in result["predictions"]:
            yield f"data: {json.dumps(prediction, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/v1/experiments")
def experiments() -> dict:
    summary = RUNS_PATH / "summary.json"
    if not summary.exists():
        raise HTTPException(404, "experiment summary not found")
    methods = {}
    for path in sorted(RUNS_PATH.glob("*/metrics.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        methods[value["method"]] = value
    return {"summary": json.loads(summary.read_text(encoding="utf-8")), "experiments": methods}


@app.get("/v1/registry")
def registry() -> dict:
    return store.registry()


@app.post("/v1/annotations")
def annotate(payload: AnnotationRequest) -> dict:
    queue_path = RUNS_PATH / "active_balanced_replay" / "metrics.json"
    if not queue_path.exists():
        raise HTTPException(404, "annotation queue unavailable")
    data = json.loads(queue_path.read_text(encoding="utf-8"))
    item = next(
        (row for row in data["annotation_queue"] if row["event_id"] == payload.event_id), None
    )
    if not item:
        raise HTTPException(404, "event not found")
    item["annotation_status"] = payload.action
    if payload.corrected_intent:
        item["corrected_intent"] = payload.corrected_intent
    atomic_json(queue_path, data)
    return item


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
