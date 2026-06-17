"""
FastAPI service for the text preprocessing pipeline.

Endpoints
---------
POST /preprocess/document  — preprocess a single document
POST /preprocess/batch     — preprocess a list of documents in parallel
POST /preprocess/query     — preprocess a query string
GET  /health               — liveness probe
"""

import logging
import os
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from preprocessor import TextPreprocessor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Preprocessing Service",
    version="2.0.0",
    description="Text preprocessing pipeline for the IR system.",
)

_preprocessor = TextPreprocessor(
    language=os.getenv("PREPROCESS_LANGUAGE", "english"),
    do_stemming=os.getenv("PREPROCESS_STEMMING", "true").lower() == "true",
    do_lemmatization=os.getenv("PREPROCESS_LEMMATIZATION", "false").lower() == "true",
    remove_stopwords=os.getenv("PREPROCESS_STOPWORDS", "true").lower() == "true",
    min_token_length=int(os.getenv("PREPROCESS_MIN_TOKEN_LEN", "2")),
)
logger.info("TextPreprocessor initialised.")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier.")
    text: str = Field(..., description="Raw document text to preprocess.")


class DocumentResponse(BaseModel):
    doc_id: str
    text: str
    tokens: List[str]
    processed_text: str


class BatchRequest(BaseModel):
    documents: List[DocumentRequest] = Field(
        ..., min_length=1, description="List of documents to preprocess."
    )
    n_jobs: int = Field(
        default=-1,
        description="Worker processes for parallel processing. -1 = all CPUs.",
    )


class BatchResponse(BaseModel):
    documents: List[DocumentResponse]


class QueryRequest(BaseModel):
    query_text: str = Field(..., description="Raw query string to preprocess.")


class QueryResponse(BaseModel):
    original: str
    tokens: List[str]
    processed_text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "preprocessing"}


@app.post(
    "/preprocess/document",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    tags=["preprocessing"],
)
def preprocess_document(req: DocumentRequest) -> DocumentResponse:
    """Preprocess a single document.

    Returns the original document enriched with ``tokens`` and
    ``processed_text``.
    """
    logger.info("Preprocessing document doc_id='%s'", req.doc_id)
    try:
        result = _preprocessor.preprocess_document({"doc_id": req.doc_id, "text": req.text})
    except Exception as exc:
        logger.exception("Error preprocessing document doc_id='%s'", req.doc_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {exc}",
        ) from exc

    return DocumentResponse(**result)


@app.post(
    "/preprocess/batch",
    response_model=BatchResponse,
    status_code=status.HTTP_200_OK,
    tags=["preprocessing"],
)
def preprocess_batch(req: BatchRequest) -> BatchResponse:
    """Preprocess a list of documents in parallel.

    ``n_jobs=-1`` uses all available CPU cores. Parallelism kicks in
    automatically for larger batches.
    """
    logger.info(
        "Batch preprocessing %d documents (n_jobs=%d).",
        len(req.documents),
        req.n_jobs,
    )
    texts = [doc.text for doc in req.documents]
    try:
        token_lists = _preprocessor.preprocess_batch(texts, n_jobs=req.n_jobs)
    except Exception as exc:
        logger.exception("Error during batch preprocessing.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch preprocessing failed: {exc}",
        ) from exc

    processed_docs = [
        DocumentResponse(
            doc_id=doc.doc_id,
            text=doc.text,
            tokens=tokens,
            processed_text=" ".join(tokens),
        )
        for doc, tokens in zip(req.documents, token_lists)
    ]
    return BatchResponse(documents=processed_docs)


@app.post(
    "/preprocess/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["preprocessing"],
)
def preprocess_query(req: QueryRequest) -> QueryResponse:
    """Preprocess a query string.

    Returns the original query alongside its token list and joined
    processed form.
    """
    logger.info("Preprocessing query: '%s'", req.query_text[:80])
    try:
        result = _preprocessor.preprocess_query(req.query_text)
    except Exception as exc:
        logger.exception("Error preprocessing query.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query preprocessing failed: {exc}",
        ) from exc

    return QueryResponse(**result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PREPROCESSING_PORT", "8001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
