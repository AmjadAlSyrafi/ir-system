import os
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/index", tags=["index"])

PREPROCESSING_URL = os.getenv("PREPROCESSING_URL", "http://preprocessing:8001")
INDEXING_URL = os.getenv("INDEXING_URL", "http://indexing:8002")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval:8003")


class Document(BaseModel):
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class IndexRequest(BaseModel):
    documents: List[Document]
    index_name: str = "default"
    preprocess: bool = True


@router.post("/")
async def index_documents(req: IndexRequest):
    docs = [doc.model_dump() for doc in req.documents]

    async with httpx.AsyncClient(timeout=60) as client:
        if req.preprocess:
            try:
                texts = [d["content"] for d in docs]
                r = await client.post(f"{PREPROCESSING_URL}/preprocess", json={"texts": texts})
                r.raise_for_status()
                processed = r.json().get("processed", texts)
                for doc, p in zip(docs, processed):
                    doc["content"] = p
            except httpx.HTTPError:
                pass

        try:
            r = await client.post(
                f"{INDEXING_URL}/index",
                json={"documents": docs, "index_name": req.index_name},
            )
            r.raise_for_status()
            index_result = r.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Indexing service error: {e}")

        try:
            await client.post(
                f"{RETRIEVAL_URL}/fit",
                json={"documents": docs, "model": "all"},
            )
        except httpx.HTTPError:
            pass

    return {"status": "indexed", "index_name": req.index_name, **index_result}


@router.get("/list")
async def list_indexes():
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{INDEXING_URL}/indexes")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Indexing service error: {e}")
