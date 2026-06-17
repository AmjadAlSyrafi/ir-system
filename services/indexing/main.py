"""
FastAPI app for the Inverted Index service.

Endpoints
---------
POST /index/build     — build index from preprocessed documents
GET  /index/stats     — return index statistics
GET  /index/postings/{term} — return postings list for a term
POST /index/save      — persist current index to disk
POST /index/load      — load index from disk
GET  /health
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Allow running directly from this directory.
sys.path.insert(0, os.path.dirname(__file__))
from indexer import InvertedIndex

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & shared index instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Indexing Service",
    description="Inverted index build, query, and persistence service.",
    version="2.0.0",
)

INDEX_NAME = os.getenv("INDEX_NAME", "main")
STORAGE_PATH = os.getenv("INDEX_STORAGE_PATH", "/data/indexes")

index = InvertedIndex(index_name=INDEX_NAME, storage_path=STORAGE_PATH)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TokenisedDocument(BaseModel):
    doc_id: str
    tokens: List[str]


class BuildRequest(BaseModel):
    documents: List[TokenisedDocument] = Field(
        ..., description="Pre-tokenised documents to index."
    )
    index_name: Optional[str] = Field(
        None, description="Override the index name for this build."
    )


class BuildResponse(BaseModel):
    indexed: int
    index_name: str
    stats: Dict[str, Any]


class SaveRequest(BaseModel):
    filename: Optional[str] = Field(
        None, description="Override the default filename (stem only or full path)."
    )


class LoadRequest(BaseModel):
    filename: Optional[str] = Field(
        None, description="Override the default filename to load from."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "indexing"}


@app.post("/index/build", response_model=BuildResponse, summary="Build index from documents")
def build_index(req: BuildRequest) -> BuildResponse:
    """Build the in-memory inverted index from a list of tokenised documents.

    Replaces any previously built index.
    """
    global index

    name = req.index_name or INDEX_NAME
    index = InvertedIndex(index_name=name, storage_path=STORAGE_PATH)

    docs = [{"doc_id": d.doc_id, "tokens": d.tokens} for d in req.documents]
    try:
        index.build_from_dataset(docs)
    except Exception as exc:
        logger.exception("Index build failed.")
        raise HTTPException(status_code=500, detail=f"Build failed: {exc}") from exc

    logger.info("Index '%s' built with %d documents.", name, index.doc_count)
    return BuildResponse(indexed=index.doc_count, index_name=name, stats=index.get_stats())


@app.get("/index/stats", summary="Return index statistics")
def get_stats() -> Dict[str, Any]:
    """Return vocab size, document count, average document length, and total terms."""
    return index.get_stats()


@app.get(
    "/index/postings/{term}",
    summary="Return postings list for a term",
    response_model=Dict[str, int],
)
def get_postings(term: str) -> Dict[str, int]:
    """Return ``{doc_id: term_frequency}`` for *term*.

    Returns an empty dict (not 404) when the term is absent from the vocabulary,
    which is the standard IR behaviour.
    """
    postings = index.get_postings(term.lower())
    logger.debug("Postings for '%s': %d documents.", term, len(postings))
    return postings


@app.post("/index/save", summary="Persist index to disk")
def save_index(req: SaveRequest = SaveRequest()) -> Dict[str, str]:
    """Serialise the current in-memory index to disk via pickle."""
    if index.doc_count == 0:
        raise HTTPException(status_code=400, detail="Index is empty — nothing to save.")
    try:
        path = index.save(req.filename)
    except OSError as exc:
        logger.exception("Failed to save index.")
        raise HTTPException(status_code=500, detail=f"Save failed: {exc}") from exc
    return {"saved_to": path}


@app.post("/index/load", summary="Load index from disk")
def load_index(req: LoadRequest = LoadRequest()) -> Dict[str, Any]:
    """Deserialise a previously saved index from disk."""
    try:
        index.load(req.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error loading index.")
        raise HTTPException(status_code=500, detail=f"Load failed: {exc}") from exc
    return {"loaded": index.index_name, "stats": index.get_stats()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("INDEXING_PORT", 8002))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)