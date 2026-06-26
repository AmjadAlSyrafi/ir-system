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
import re
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

_TIMEOUT      = httpx.Timeout(60.0)
_EVAL_TIMEOUT = httpx.Timeout(300.0)

AVAILABLE_MODELS = ["tfidf", "bm25", "embedding", "hybrid_serial", "hybrid_parallel"]
AVAILABLE_DATASETS = {
    "dataset1": "beir/quora/test",
    "dataset2": "beir/hotpotqa/test",
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
    hybrid_bm25_weight: Optional[float] = None
    hybrid_embedding_weight: Optional[float] = None
    hybrid_tfidf_weight: Optional[float] = None


class IndexBuildRequest(BaseModel):
    dataset: str = Field(default="dataset1", description="'dataset1' or 'dataset2'")


class EvaluateRequest(BaseModel):
    model_name: str
    dataset: str = "dataset1"
    results_per_query: Dict[str, List[Dict[str, Any]]]
    qrels: List[Dict[str, Any]]
    k: int = 10


class RunEvalRequest(BaseModel):
    dataset: str = Field(default="dataset1", description="'dataset1' or 'dataset2'")
    models: List[str] = Field(default=["bm25"], description="List of model names to evaluate")
    max_queries: int = Field(default=20, ge=1, le=200)
    k: int = Field(default=10, ge=1, le=100)

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
        # Map hybrid_serial / hybrid_parallel → model=hybrid + mode override
        retrieval_model = req.model
        hybrid_mode: Optional[str] = None
        if req.model == "hybrid_serial":
            retrieval_model = "hybrid"
            hybrid_mode = "serial"
        elif req.model == "hybrid_parallel":
            retrieval_model = "hybrid"
            hybrid_mode = "parallel"

        retrieval_body: Dict[str, Any] = {
            "query": refined_query,
            "query_tokens": query_tokens,
            "model": retrieval_model,
            "top_k": req.top_k,
            "dataset": req.dataset,
        }
        if hybrid_mode:
            retrieval_body["mode"] = hybrid_mode
        if req.bm25_k1 is not None:
            retrieval_body["bm25_k1"] = req.bm25_k1
        if req.bm25_b is not None:
            retrieval_body["bm25_b"] = req.bm25_b
        if req.hybrid_bm25_weight is not None:
            retrieval_body["hybrid_bm25_weight"] = req.hybrid_bm25_weight
        if req.hybrid_embedding_weight is not None:
            retrieval_body["hybrid_embedding_weight"] = req.hybrid_embedding_weight
        if req.hybrid_tfidf_weight is not None:
            retrieval_body["hybrid_tfidf_weight"] = req.hybrid_tfidf_weight

        try:
            retrieval_resp = await _post(
                client, f"{RETRIEVAL_URL}/search", retrieval_body
            )
        except HTTPException as exc:
            if exc.status_code == 503 and "initializing" in exc.detail.lower():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"The '{req.model}' model is still warming up. "
                        "Please wait a few seconds and try again, "
                        "or switch to 'bm25' or 'tfidf'."
                    ),
                ) from exc
            raise

        # Enrich results with full original text from the document store.
        results: List[Dict[str, Any]] = retrieval_resp.get("results", [])
        if results:
            try:
                doc_ids = [r["doc_id"] for r in results]
                doc_resp = await client.post(
                    f"{INDEXING_URL}/documents/batch",
                    json={"doc_ids": doc_ids, "dataset": req.dataset},
                    timeout=httpx.Timeout(10.0),
                )
                if doc_resp.status_code == 200:
                    texts: Dict[str, str] = doc_resp.json()
                    for r in results:
                        r["text"] = texts.get(r["doc_id"], "")
            except Exception as exc:
                logger.warning("Could not fetch document texts: %s", exc)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "query": req.query,
        "refined_query": refined_query,
        "refinement": refinement_info,
        "model": req.model,
        "dataset": req.dataset,
        "top_k": req.top_k,
        "time_ms": elapsed_ms,
        "results": results,
    }


_SAMPLE_DOCUMENTS = [
    {"doc_id":"d001","text":"TF-IDF stands for term frequency inverse document frequency. It reflects how important a word is to a document in a collection by combining local term frequency with a global penalty for common terms."},
    {"doc_id":"d002","text":"BM25 is a probabilistic bag-of-words retrieval function that ranks documents by query term frequency, penalising very long documents and saturating repeated terms with the k1 parameter."},
    {"doc_id":"d003","text":"Dense retrieval encodes queries and documents into shared vector spaces using neural networks, enabling semantic similarity search beyond exact keyword matching."},
    {"doc_id":"d004","text":"An inverted index maps each vocabulary term to the list of documents containing it along with positional and frequency information, enabling fast postings lookup during retrieval."},
    {"doc_id":"d005","text":"Precision measures the fraction of retrieved documents that are relevant. Recall measures the fraction of all relevant documents that were retrieved. Both are key IR evaluation metrics."},
    {"doc_id":"d006","text":"NDCG, Normalised Discounted Cumulative Gain, evaluates ranking quality by awarding higher credit to relevant documents appearing near the top of the result list."},
    {"doc_id":"d007","text":"Mean Average Precision (MAP) averages the precision at each relevant document rank across all queries, giving a single number summarising ranked retrieval quality."},
    {"doc_id":"d008","text":"Query expansion adds related terms to a user query to improve retrieval recall. Common techniques include WordNet synonym lookup and pseudo-relevance feedback from top results."},
    {"doc_id":"d009","text":"The Vector Space Model represents documents and queries as vectors in a high-dimensional term space. Cosine similarity between query and document vectors determines relevance scores."},
    {"doc_id":"d010","text":"FAISS (Facebook AI Similarity Search) is a library for efficient approximate nearest-neighbour search over dense floating-point vectors, widely used in dense retrieval pipelines."},
    {"doc_id":"d011","text":"Transformer models use self-attention mechanisms to produce contextualised token representations, capturing long-range dependencies that earlier recurrent models struggled with."},
    {"doc_id":"d012","text":"BERT is a bidirectional transformer pre-trained with masked language modelling on large corpora and can be fine-tuned on tasks such as passage retrieval and question answering."},
    {"doc_id":"d013","text":"Sentence-BERT produces fixed-size sentence embeddings by adding a pooling layer to BERT and training with siamese networks on natural language inference data."},
    {"doc_id":"d014","text":"Word2Vec learns distributed word representations by predicting a word from its context window (CBOW) or predicting context from a central word (Skip-gram)."},
    {"doc_id":"d015","text":"Bi-encoder models encode queries and documents independently, making them suitable for large-scale retrieval since document embeddings can be precomputed and indexed offline."},
    {"doc_id":"d016","text":"Cross-encoder models concatenate query and document tokens and pass the pair through a transformer, producing accurate relevance scores but requiring one forward pass per candidate."},
    {"doc_id":"d017","text":"Contrastive learning trains encoders to place semantically similar pairs close together and dissimilar pairs far apart in embedding space, using in-batch negatives for efficiency."},
    {"doc_id":"d018","text":"MS-MARCO DistilBERT is a lightweight bi-encoder fine-tuned on the MS-MARCO passage ranking dataset, offering good retrieval quality at lower inference cost than full BERT."},
    {"doc_id":"d019","text":"ColBERT computes a MaxSim score between all query and document token embeddings, balancing expressiveness and retrieval efficiency with late interaction."},
    {"doc_id":"d020","text":"Knowledge distillation trains a small student model to mimic the output of a larger teacher model, reducing inference cost while retaining much of the teacher performance."},
    {"doc_id":"d021","text":"Hybrid retrieval combines sparse lexical models like BM25 with dense semantic models to leverage complementary matching signals, improving performance over either alone."},
    {"doc_id":"d022","text":"Reciprocal Rank Fusion (RRF) merges multiple ranked lists by assigning each document a score of 1 divided by k plus its rank in each list, then summing across lists."},
    {"doc_id":"d023","text":"Linear score fusion normalises each model scores to a common range and computes a weighted sum, requiring careful calibration of model-specific weights."},
    {"doc_id":"d024","text":"Multi-stage retrieval uses a fast first-stage ranker such as BM25 to produce a candidate set, then applies a more expensive neural re-ranker to the candidates only."},
    {"doc_id":"d025","text":"Re-ranking with a cross-encoder on the top-100 BM25 candidates typically improves nDCG@10 substantially while keeping end-to-end latency acceptable for interactive search."},
    {"doc_id":"d026","text":"PageRank models the web as a graph and assigns each page an importance score proportional to the number and quality of inbound hyperlinks, used as a ranking signal."},
    {"doc_id":"d027","text":"Elasticsearch is a distributed search engine built on Apache Lucene that supports full-text search, real-time indexing, and aggregations at scale."},
    {"doc_id":"d028","text":"Query understanding identifies spelling errors, expands abbreviations, classifies intent, and maps queries to structured representations before retrieval begins."},
    {"doc_id":"d029","text":"Faceted search lets users iteratively refine results by applying attribute filters such as category, price range, or date, turning a single query into a navigational experience."},
    {"doc_id":"d030","text":"Document indexing transforms raw text into an inverted index through tokenisation, normalisation, and posting list construction, enabling millisecond-latency retrieval."},
    {"doc_id":"d031","text":"Tokenisation splits raw text into individual tokens such as words, subwords, or characters. It is the first step in virtually every natural language processing pipeline."},
    {"doc_id":"d032","text":"Porter stemming removes morphological suffixes to reduce inflected words to a common stem. For example, running, runs, and runner all reduce to run."},
    {"doc_id":"d033","text":"WordNet lemmatisation converts each token to its dictionary base form using part-of-speech context, producing linguistically valid words unlike stemming."},
    {"doc_id":"d034","text":"Stop word removal discards high-frequency function words such as the, is, and at that carry little discriminative information for term-based retrieval models."},
    {"doc_id":"d035","text":"Named entity recognition identifies spans of text referring to people, organisations, locations, dates, and other entity types, useful for structured query processing."},
    {"doc_id":"d036","text":"Part-of-speech tagging assigns grammatical categories to each token in a sentence, enabling downstream tasks like lemmatisation, chunking, and syntactic parsing."},
    {"doc_id":"d037","text":"Spell correction detects and repairs typographical errors in user queries before retrieval, improving recall for misspelled terms that would otherwise match no documents."},
    {"doc_id":"d038","text":"Text normalisation converts text to a canonical form by lowercasing, expanding contractions, removing punctuation, and replacing digits and special characters."},
    {"doc_id":"d039","text":"Sub-word tokenisation algorithms such as BPE and WordPiece split rare words into smaller units, helping neural models handle out-of-vocabulary terms gracefully."},
    {"doc_id":"d040","text":"Synonym expansion uses lexical resources like WordNet or word embeddings to identify terms with similar meaning and add them to a query to improve retrieval recall."},
    {"doc_id":"d041","text":"TREC (Text REtrieval Conference) provides standardised test collections, queries, and relevance judgements that enable rigorous comparable evaluation of retrieval systems."},
    {"doc_id":"d042","text":"MS MARCO contains real user queries from Bing search logs with sparse human-annotated relevance labels, making it a challenging large-scale retrieval benchmark."},
    {"doc_id":"d043","text":"BEIR is a zero-shot retrieval benchmark covering 18 diverse datasets spanning biomedical, financial, scientific, and news domains, testing generalisation beyond training data."},
    {"doc_id":"d044","text":"Relevance judgements (qrels) provide binary or graded human assessments of document relevance for each query and form the ground truth used to compute evaluation metrics."},
    {"doc_id":"d045","text":"Learning to rank trains a supervised model on query-document feature vectors to produce relevance rankings that outperform hand-tuned weighting schemes."},
    {"doc_id":"d046","text":"Approximate nearest-neighbour algorithms like HNSW and IVF-Flat enable sub-millisecond similarity search over billions of vectors at the cost of a small recall trade-off."},
    {"doc_id":"d047","text":"Document clustering organises a corpus into thematically coherent groups using algorithms such as k-means or hierarchical clustering over TF-IDF or embedding representations."},
    {"doc_id":"d048","text":"Semantic search goes beyond keyword matching to understand the meaning and intent behind queries, returning results that are conceptually related even without shared keywords."},
    {"doc_id":"d049","text":"Open-domain question answering retrieves candidate passages from a large corpus and then applies a reader model to extract or generate a precise answer to a natural language question."},
    {"doc_id":"d050","text":"Retrieval augmented generation (RAG) combines a dense retrieval model with a generative language model, conditioning generation on retrieved context to improve factual accuracy."},
    {"doc_id":"d051","text":"Passage retrieval selects short text segments rather than whole documents, improving both relevance precision and answer extraction accuracy for question answering tasks."},
    {"doc_id":"d052","text":"Pseudo-relevance feedback assumes the top-k retrieved documents are relevant and extracts high-weight terms to expand the original query before a second retrieval pass."},
    {"doc_id":"d053","text":"Approximate nearest-neighbour search with HNSW and IVFFlat enables fast similarity search over large vector collections with tunable recall-speed trade-offs."},
    {"doc_id":"d054","text":"Entity linking connects ambiguous surface forms in text to unique knowledge base entries, enabling structured retrieval and question answering over entity-rich corpora."},
    {"doc_id":"d055","text":"The recall-precision curve shows the trade-off between recall and precision at different retrieval thresholds. Higher area under the curve indicates a better retrieval model."},
    {"doc_id":"d056","text":"Interpolated average precision fills in the precision curve so it is non-increasing, allowing fair comparison between systems retrieving different numbers of relevant documents."},
    {"doc_id":"d057","text":"Zero-shot retrieval evaluates models on domains not seen during training to assess generalisation, typically using the BEIR benchmark with a pre-trained encoder and no fine-tuning."},
    {"doc_id":"d058","text":"Index compression techniques such as variable-byte encoding and FOR-delta reduce the disk footprint of inverted indices while keeping decompression fast during query processing."},
    {"doc_id":"d059","text":"Query latency is a critical operational metric for production search systems; sub-100ms response times typically require pre-computed indexes and approximate retrieval methods."},
    {"doc_id":"d060","text":"Pseudo-relevance feedback assumes the top-k retrieved documents are relevant and extracts high-weight terms to expand the original query before a second retrieval pass."},
]


@app.post("/index/build", summary="Trigger index build for a dataset")
async def build_index(req: IndexBuildRequest) -> Dict[str, Any]:
    """Preprocess the built-in sample corpus and build the inverted index.

    Flow
    ----
    1. Send each document to the Preprocessing service.
    2. Collect tokenised documents.
    3. POST the full list to the Indexing service's /index/build.
    4. Signal the Retrieval service to reload its models.
    """
    if req.dataset not in AVAILABLE_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dataset '{req.dataset}'.",
        )

    async with httpx.AsyncClient() as client:
        # Step 1 — preprocess each document
        tokenised_docs: List[Dict[str, Any]] = []
        for doc in _SAMPLE_DOCUMENTS:
            try:
                resp = await _post(
                    client,
                    f"{PREPROCESSING_URL}/preprocess/document",
                    {"doc_id": doc["doc_id"], "text": doc["text"]},
                )
                tokenised_docs.append({
                    "doc_id": resp.get("doc_id", doc["doc_id"]),
                    "tokens": resp.get("tokens", doc["text"].lower().split()),
                    "text": doc["text"],
                })
            except HTTPException:
                tokenised_docs.append({
                    "doc_id": doc["doc_id"],
                    "tokens": doc["text"].lower().split(),
                    "text": doc["text"],
                })

        # Step 2 — build inverted index (also stores texts in document DB)
        index_resp = await _post(
            client,
            f"{INDEXING_URL}/index/build",
            {"documents": tokenised_docs, "dataset": req.dataset},
        )

        # Step 3 — tell retrieval service to reload (best-effort)
        try:
            await client.post(f"{RETRIEVAL_URL}/reload", timeout=_TIMEOUT)
        except Exception:
            pass

    return {
        "dataset": req.dataset,
        "documents_indexed": len(tokenised_docs),
        "index": index_resp,
    }


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


@app.post("/evaluate/run", summary="Server-side evaluation against real dataset qrels")
async def run_evaluation(req: RunEvalRequest) -> Dict[str, Any]:
    """Fetch real queries + qrels from the dataset, run all requested models,
    and return metrics — no client-side qrels needed.
    """
    if req.dataset not in AVAILABLE_DATASETS:
        raise HTTPException(400, f"Unknown dataset '{req.dataset}'.")
    for m in req.models:
        if m not in AVAILABLE_MODELS:
            raise HTTPException(400, f"Unknown model '{m}'.")

    dataset_id = AVAILABLE_DATASETS[req.dataset]   # e.g. "antique/test"

    async with httpx.AsyncClient() as client:
        # 1. Fetch real queries + qrels from preprocessing service
        try:
            es_resp = await client.get(
                f"{PREPROCESSING_URL}/dataset/{dataset_id}/eval-set",
                params={"max_queries": req.max_queries * 10},  # over-fetch for filtering
                timeout=_EVAL_TIMEOUT,
            )
            es_resp.raise_for_status()
        except Exception as exc:
            raise HTTPException(502, f"Could not load eval set: {exc}") from exc

        eval_set = es_resp.json()
        queries: Dict[str, str] = eval_set["queries"]
        qrels: List[Dict[str, Any]] = eval_set["qrels"]

        if not queries:
            raise HTTPException(404, "No queries with relevance judgements found for this dataset.")

        # 2. Filter to queries whose relevant docs are in the indexed corpus.
        #    Without this, sampling <1% of a large corpus gives 0.0000 metrics.
        try:
            ids_resp = await client.get(
                f"{INDEXING_URL}/documents/ids",
                params={"dataset": req.dataset},
                timeout=httpx.Timeout(30.0),
            )
            if ids_resp.status_code == 200:
                corpus_ids = set(ids_resp.json().get("doc_ids", []))
                if corpus_ids:
                    valid_qids = {
                        qr["query_id"] for qr in qrels
                        if qr["doc_id"] in corpus_ids
                    }
                    queries = {qid: t for qid, t in queries.items() if qid in valid_qids}
                    qrels   = [qr for qr in qrels if qr["query_id"] in valid_qids]
                    # Trim to max_queries after filtering
                    trim_qids = list(queries)[:req.max_queries]
                    queries = {qid: queries[qid] for qid in trim_qids}
                    qrels   = [qr for qr in qrels if qr["query_id"] in trim_qids]
        except Exception as exc:
            logger.warning("Could not fetch corpus doc_ids for filtering: %s", exc)

        if not queries:
            return {
                "dataset": req.dataset,
                "dataset_id": dataset_id,
                "num_queries": 0,
                "k": req.k,
                "models": {m: {"error": "No queries found with relevant documents in the indexed corpus. "
                               "Re-run the pipeline with a larger --sample_size."} for m in req.models},
            }

        model_metrics: Dict[str, Any] = {}

        for raw_model in req.models:
            retrieval_model = raw_model
            hybrid_mode: Optional[str] = None
            if raw_model == "hybrid_serial":
                retrieval_model = "hybrid"
                hybrid_mode = "serial"
            elif raw_model == "hybrid_parallel":
                retrieval_model = "hybrid"
                hybrid_mode = "parallel"

            results_per_query: Dict[str, List[Dict[str, Any]]] = {}
            not_ready_msg: Optional[str] = None

            for qid, query_text in queries.items():
                try:
                    # Preprocess query
                    pre = await _post(
                        client,
                        f"{PREPROCESSING_URL}/preprocess/query",
                        {"query_text": query_text},
                    )
                    query_tokens: List[str] = pre.get("tokens", [])

                    # Retrieve
                    body: Dict[str, Any] = {
                        "query": query_text,
                        "query_tokens": query_tokens,
                        "model": retrieval_model,
                        "top_k": req.k,
                        "dataset": req.dataset,
                    }
                    if hybrid_mode:
                        body["mode"] = hybrid_mode

                    search_resp = await client.post(
                        f"{RETRIEVAL_URL}/search",
                        json=body,
                        timeout=_EVAL_TIMEOUT,
                    )
                    search_resp.raise_for_status()
                    results_per_query[qid] = search_resp.json().get("results", [])
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text
                    _NOT_READY = ("not fitted", "not loaded", "faiss", "initializ", "503")
                    if any(kw in detail.lower() for kw in _NOT_READY) or exc.response.status_code in (503, 425):
                        not_ready_msg = detail[:200]
                        break
                    results_per_query[qid] = []
                except Exception:
                    results_per_query[qid] = []

            if not_ready_msg:
                model_metrics[raw_model] = {"error": f"Model not ready: {not_ready_msg}"}
                continue

            # Compute metrics
            try:
                eval_resp = await _post(
                    client,
                    f"{EVALUATION_URL}/evaluate",
                    {
                        "model_name": raw_model,
                        "dataset": req.dataset,
                        "results_per_query": results_per_query,
                        "qrels": qrels,
                        "k": req.k,
                    },
                )
                model_metrics[raw_model] = eval_resp
            except Exception as exc:
                model_metrics[raw_model] = {"error": str(exc)}

    return {
        "dataset": req.dataset,
        "dataset_id": dataset_id,
        "num_queries": len(queries),
        "k": req.k,
        "models": model_metrics,
    }


_REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def _parse_md_table(md: str) -> List[Dict[str, str]]:
    """Extract rows from the first markdown table in *md* as a list of dicts."""
    rows: List[Dict[str, str]] = []
    lines = [l.strip() for l in md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return rows
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    for line in lines[2:]:
        values = [v.strip() for v in line.split("|")[1:-1]]
        if len(values) == len(headers):
            rows.append(dict(zip(headers, values)))
    return rows


@app.get("/reports/{dataset}", summary="Return pre-computed pipeline evaluation report")
async def get_report(dataset: str) -> Dict[str, Any]:
    """Read the Markdown report generated by run_pipeline.py and return
    both structured row data (for table rendering) and the raw text.
    """
    if dataset not in AVAILABLE_DATASETS:
        raise HTTPException(400, f"Unknown dataset '{dataset}'.")
    path = os.path.join(_REPORT_DIR, f"evaluation_report_{dataset}.md")
    if not os.path.exists(path):
        raise HTTPException(
            404,
            f"No report found for '{dataset}'. Run: python run_pipeline.py",
        )
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    return {
        "dataset": dataset,
        "dataset_id": AVAILABLE_DATASETS[dataset],
        "rows": _parse_md_table(content),
        "raw": content,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_GATEWAY_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)