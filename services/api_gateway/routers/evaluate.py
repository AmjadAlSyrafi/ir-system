import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(prefix="/evaluate", tags=["evaluate"])

EVALUATION_URL = os.getenv("EVALUATION_URL", "http://evaluation:8005")


class EvalRequest(BaseModel):
    retrieved: List[str]
    relevant: List[str]
    k_values: List[int] = [1, 5, 10]


class MAPRequest(BaseModel):
    queries: List[Dict[str, List[str]]]


@router.post("/")
async def evaluate(req: EvalRequest):
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.post(f"{EVALUATION_URL}/evaluate", json=req.model_dump())
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Evaluation service error: {e}")


@router.post("/map")
async def map_score(req: MAPRequest):
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.post(f"{EVALUATION_URL}/map", json=req.model_dump())
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Evaluation service error: {e}")
