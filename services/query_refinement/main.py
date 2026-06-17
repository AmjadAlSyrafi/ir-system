"""
FastAPI app for the Query Refinement service.

Endpoints
---------
POST   /refine              — refine a query (spell, synonyms, history)
GET    /history/{user_id}   — retrieve query history for a user
DELETE /history/{user_id}   — clear query history for a user
GET    /health
"""

import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
from refiner import QueryRefiner

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
    title="Query Refinement Service",
    description="Spell correction, synonym expansion, and history-based query boosting.",
    version="2.0.0",
)

_HISTORY_FILE = os.getenv("HISTORY_FILE", "query_history.json")

refiner = QueryRefiner(
    use_spellcheck=os.getenv("USE_SPELLCHECK", "true").lower() == "true",
    use_synonyms=os.getenv("USE_SYNONYMS", "true").lower() == "true",
    use_history=os.getenv("USE_HISTORY", "true").lower() == "true",
    history_file=_HISTORY_FILE,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RefineRequest(BaseModel):
    query: str = Field(..., description="Raw query string to refine.")
    user_id: str = Field(default="default", description="User scope for history.")


class RefineResponse(BaseModel):
    original_query: str
    corrected_query: str
    expanded_query: str
    final_query: str
    changes_made: list


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "service": "query_refinement"}


@app.post("/refine", response_model=RefineResponse, summary="Refine a query")
def refine(req: RefineRequest) -> RefineResponse:
    """Apply spell correction, synonym expansion, and history boosting.

    Returns a full trace of every transformation applied.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="'query' must not be empty.")
    try:
        result = refiner.refine(req.query, user_id=req.user_id)
    except Exception as exc:
        logger.exception("Error refining query.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RefineResponse(**result)


@app.get(
    "/history/{user_id}",
    summary="Get query history for a user",
    response_model=list,
)
def get_history(user_id: str) -> list:
    """Return the list of past queries for *user_id* (oldest first)."""
    return refiner.get_history(user_id)


@app.delete("/history/{user_id}", summary="Clear query history for a user")
def clear_history(user_id: str) -> dict:
    """Delete all history entries for *user_id*."""
    refiner.clear_history(user_id)
    return {"cleared": user_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("QUERY_REFINEMENT_PORT", 8005))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)