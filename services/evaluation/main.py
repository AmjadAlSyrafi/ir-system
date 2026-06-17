"""
FastAPI app for the Evaluation service.

Endpoints
---------
POST /evaluate  — evaluate a single model's results
POST /compare   — compare multiple models, return DataFrame as JSON
GET  /report    — return the latest saved markdown report
GET  /health
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from metrics import IREvaluator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Evaluation Service",
    description="IR evaluation metrics: MAP, Recall, P@k, nDCG@k.",
    version="2.0.0",
)

_REPORT_PATH = os.getenv("REPORT_PATH", "/data/evaluation_report.md")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RankedResult(BaseModel):
    doc_id: str
    score: float
    rank: int


class QrelEntry(BaseModel):
    """Single qrel: one (query_id, doc_id, relevance) triple."""
    query_id: str
    doc_id: str
    relevance: int


class EvaluateRequest(BaseModel):
    model_name: str
    qrels: List[QrelEntry] = Field(..., description="Relevance judgments.")
    results_per_query: Dict[str, List[RankedResult]] = Field(
        ..., description="{query_id: [ranked results]}"
    )
    k: int = Field(default=10, ge=1)


class CompareRequest(BaseModel):
    qrels: List[QrelEntry]
    model_results: Dict[str, Dict[str, List[RankedResult]]] = Field(
        ..., description="{model_name: {query_id: [ranked results]}}"
    )
    k: int = Field(default=10, ge=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_qrels(entries: List[QrelEntry]) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    for e in entries:
        qrels.setdefault(e.query_id, {})[e.doc_id] = e.relevance
    return qrels


def _to_dicts(results_per_query: Dict[str, List[RankedResult]]) -> Dict[str, List[Dict]]:
    return {
        qid: [r.model_dump() for r in results]
        for qid, results in results_per_query.items()
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "evaluation"}


@app.post("/evaluate", summary="Evaluate a single model")
def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    """Compute MAP, Recall, P@k, and nDCG@k for one model."""
    qrels = _build_qrels(req.qrels)
    evaluator = IREvaluator(qrels)
    results_dicts = _to_dicts(req.results_per_query)

    try:
        summary = evaluator.evaluate_model(req.model_name, results_dicts, k=req.k)
    except Exception as exc:
        logger.exception("Evaluation failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return summary


@app.post("/compare", summary="Compare multiple models")
def compare(req: CompareRequest) -> Dict[str, Any]:
    """Run evaluate_model for each model and return a comparison table.

    The DataFrame is serialised as ``{columns: [...], data: [[...]...]}``.
    """
    qrels = _build_qrels(req.qrels)
    evaluator = IREvaluator(qrels)

    model_results_dicts: Dict[str, Dict[str, List[Dict]]] = {
        model_name: _to_dicts(results_per_query)
        for model_name, results_per_query in req.model_results.items()
    }

    try:
        df = evaluator.compare_models(model_results_dicts, k=req.k)
        evaluator.generate_report(df, _REPORT_PATH)
    except Exception as exc:
        logger.exception("Comparison failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "columns": list(df.columns),
        "index": list(df.index),
        "data": df.reset_index().to_dict(orient="records"),
    }


@app.get("/report", summary="Return latest evaluation report", response_class=PlainTextResponse)
def get_report() -> str:
    """Return the Markdown evaluation report written by the last /compare call."""
    if not os.path.exists(_REPORT_PATH):
        raise HTTPException(
            status_code=404,
            detail="No report found. Run /compare first.",
        )
    with open(_REPORT_PATH, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("EVALUATION_PORT", 8006))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)