"""
API Gateway — single entry point for the IR system.

Orchestrates calls to:
  Preprocessing  http://localhost:8001
  Indexing       http://localhost:8002
  Retrieval      http://localhost:8003
  Query Refine   http://localhost:8005
  Evaluation     http://localhost:8006
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service URLs
# ---------------------------------------------------------------------------

PREPROCESSING_URL = os.getenv("PREPROCESSING_URL", "http://localhost:8001")
INDEXING_URL      = os.getenv("INDEXING_URL",      "http://localhost:8002")
RETRIEVAL_URL     = os.getenv("RETRIEVAL_URL",     "http://localhost:8003")
REFINEMENT_URL    = os.getenv("QUERY_REFINEMENT_URL", "http://localhost:8005")
EVALUATION_URL    = os.getenv("EVALUATION_URL",    "http://localhost:8006")

_TIMEOUT = httpx.Timeout(60.0)

AVAILABLE_MODELS = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]
AVAILABLE_DATASETS = {
    "dataset1": "msmarco-passage",
    "dataset2": "beir/nq/train",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="IR System — API Gateway",
    description="Central entry point orchestrating all IR microservices.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "%s %s → %d  (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response

# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

async def _post(client: httpx.AsyncClient, url: str, body: dict) -> dict:
    try:
        resp = await client.post(url, json=body, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Upstream error from {url}: {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Service unreachable: {url} ({exc})",
        ) from exc


async def _get(client: httpx.AsyncClient, url: str) -> dict:
    try:
        resp = await client.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"status": "unreachable"}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    dataset: str = Field(default="dataset1", description="'dataset1' or 'dataset2'")
    model: str = Field(default="bm25", description="tfidf | bm25 | embedding | hybrid_serial | hybrid_parallel")
    top_k: int = Field(default=10, ge=1, le=100)
    use_refinement: bool = True
    user_id: str = "default"
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None


class IndexBuildRequest(BaseModel):
    dataset: str = Field(default="dataset1", description="'dataset1' or 'dataset2'")


class EvaluateRequest(BaseModel):
    model_name: str
    dataset: str = "dataset1"
    results_per_query: Dict[str, List[Dict[str, Any]]]
    qrels: List[Dict[str, Any]]
    k: int = 10

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check all services")
async def health() -> Dict[str, Any]:
    """Probe every downstream service and report their statuses."""
    async with httpx.AsyncClient() as client:
        statuses = {
            "api_gateway": "ok",
            "preprocessing": (await _get(client, f"{PREPROCESSING_URL}/health")).get("status", "unreachable"),
            "indexing":      (await _get(client, f"{INDEXING_URL}/health")).get("status", "unreachable"),
            "retrieval":     (await _get(client, f"{RETRIEVAL_URL}/health")).get("status", "unreachable"),
            "query_refinement": (await _get(client, f"{REFINEMENT_URL}/health")).get("status", "unreachable"),
            "evaluation":    (await _get(client, f"{EVALUATION_URL}/health")).get("status", "unreachable"),
        }
    overall = "ok" if all(v == "ok" for v in statuses.values()) else "degraded"
    return {"status": overall, "services": statuses}


@app.get("/datasets", summary="List available datasets")
def list_datasets() -> Dict[str, Any]:
    return {
        "datasets": [
            {"id": k, "name": v} for k, v in AVAILABLE_DATASETS.items()
        ]
    }


@app.get("/models", summary="List available retrieval models")
def list_models() -> Dict[str, List[str]]:
    return {"models": AVAILABLE_MODELS}


@app.post("/search", summary="Search documents")
async def search(req: SearchRequest) -> Dict[str, Any]:
    """Full search pipeline: optional query refinement → preprocessing → retrieval.

    Flow
    ----
    1. Optionally call Query Refinement service.
    2. Call Preprocessing service to tokenise the (refined) query.
    3. Call Retrieval service with the chosen model and dataset.
    4. Return results with metadata.
    """
    t0 = time.perf_counter()

    if req.dataset not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{req.dataset}'. Choose from {list(AVAILABLE_DATASETS)}.",
        )
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Choose from {AVAILABLE_MODELS}.",
        )

    async with httpx.AsyncClient() as client:
        # Step 1 — query refinement (optional)
        refined_query = req.query
        refinement_info = None
        if req.use_refinement:
            try:
                refine_resp = await _post(
                    client,
                    f"{REFINEMENT_URL}/refine",
                    {"query": req.query, "user_id": req.user_id},
                )
                refined_query = refine_resp.get("final_query", req.query)
                refinement_info = {
                    "original": refine_resp.get("original_query"),
                    "final": refined_query,
                    "changes": refine_resp.get("changes_made", []),
                }
                logger.info("Query refined: '%s' → '%s'", req.query, refined_query)
            except HTTPException as exc:
                logger.warning("Refinement service failed (%s); using raw query.", exc.detail)

        # Step 2 — preprocessing
        preprocess_resp = await _post(
            client,
            f"{PREPROCESSING_URL}/preprocess/query",
            {"query_text": refined_query},
        )
        query_tokens: List[str] = preprocess_resp.get("tokens", [])

        # Step 3 — retrieval
        retrieval_body: Dict[str, Any] = {
            "query_text": refined_query,
            "query_tokens": query_tokens,
            "model": req.model,
            "dataset": req.dataset,
            "top_k": req.top_k,
        }
        if req.bm25_k1 is not None:
            retrieval_body["bm25_k1"] = req.bm25_k1
        if req.bm25_b is not None:
            retrieval_body["bm25_b"] = req.bm25_b

        retrieval_resp = await _post(
            client, f"{RETRIEVAL_URL}/search", retrieval_body
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "query": req.query,
        "refined_query": refined_query,
        "refinement": refinement_info,
        "model": req.model,
        "dataset": req.dataset,
        "top_k": req.top_k,
        "time_ms": elapsed_ms,
        "results": retrieval_resp.get("results", []),
    }


@app.post("/index/build", summary="Trigger index build for a dataset")
async def build_index(req: IndexBuildRequest) -> Dict[str, Any]:
    """Ask the Indexing service to build all indexes for the given dataset."""
    if req.dataset not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{req.dataset}'.",
        )
    async with httpx.AsyncClient() as client:
        resp = await _post(
            client,
            f"{INDEXING_URL}/index/build",
            {"dataset": req.dataset},
        )
    return resp


@app.post("/evaluate", summary="Evaluate a model on a dataset")
async def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    """Forward an evaluation request to the Evaluation service."""
    async with httpx.AsyncClient() as client:
        resp = await _post(
            client,
            f"{EVALUATION_URL}/evaluate",
            req.model_dump(),
        )
    return resp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_GATEWAY_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)