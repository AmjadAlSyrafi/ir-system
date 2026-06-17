import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from models.tfidf_model import TFIDFModel
from models.bm25_model import BM25Model
from models.embedding_model import EmbeddingModel
from models.hybrid_model import HybridModel

load_dotenv()

app = FastAPI(title="Retrieval Service", version="1.0.0")

_models: Dict[str, Any] = {
    "tfidf": TFIDFModel(),
    "bm25": BM25Model(),
    "embedding": EmbeddingModel(),
    "hybrid": HybridModel(),
}


class Document(BaseModel):
    id: str
    content: str


class FitRequest(BaseModel):
    documents: List[Document]
    model: str = "hybrid"


class SearchRequest(BaseModel):
    query: str
    model: str = "hybrid"
    top_k: int = 10


class SearchResult(BaseModel):
    id: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    model: str
    results: List[SearchResult]


@app.get("/health")
def health():
    return {"status": "ok", "service": "retrieval"}


@app.get("/models")
def list_models():
    return {"models": list(_models.keys())}


@app.post("/fit")
def fit(req: FitRequest):
    if req.model not in _models and req.model != "all":
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    docs = [doc.model_dump() for doc in req.documents]
    targets = list(_models.keys()) if req.model == "all" else [req.model]
    for name in targets:
        _models[name].fit(docs)
    return {"fitted": targets, "documents": len(docs)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if req.model not in _models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")
    raw = _models[req.model].search(req.query, top_k=req.top_k)
    results = [SearchResult(**r) for r in raw]
    return SearchResponse(query=req.query, model=req.model, results=results)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RETRIEVAL_PORT", 8003))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
