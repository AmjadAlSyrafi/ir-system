import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/search", tags=["search"])

PREPROCESSING_URL = os.getenv("PREPROCESSING_URL", "http://preprocessing:8001")
QUERY_REFINEMENT_URL = os.getenv("QUERY_REFINEMENT_URL", "http://query_refinement:8004")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval:8003")


class SearchRequest(BaseModel):
    query: str
    model: str = "hybrid"
    top_k: int = 10
    refine_query: bool = True
    preprocess_query: bool = True


class SearchResult(BaseModel):
    id: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    original_query: str
    processed_query: str
    model: str
    results: List[SearchResult]


@router.post("/", response_model=SearchResponse)
async def search(req: SearchRequest):
    query = req.query

    async with httpx.AsyncClient(timeout=30) as client:
        if req.refine_query:
            try:
                r = await client.post(f"{QUERY_REFINEMENT_URL}/refine", json={"query": query})
                r.raise_for_status()
                query = r.json().get("expanded", query)
            except httpx.HTTPError:
                pass

        if req.preprocess_query:
            try:
                r = await client.post(f"{PREPROCESSING_URL}/preprocess", json={"text": query})
                r.raise_for_status()
                query = r.json().get("processed", [query])[0]
            except httpx.HTTPError:
                pass

        try:
            r = await client.post(
                f"{RETRIEVAL_URL}/search",
                json={"query": query, "model": req.model, "top_k": req.top_k},
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Retrieval service error: {e}")

    results = [SearchResult(**item) for item in data.get("results", [])]
    return SearchResponse(
        original_query=req.query,
        processed_query=query,
        model=req.model,
        results=results,
    )
