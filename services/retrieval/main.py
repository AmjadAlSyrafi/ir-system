import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.bm25_model import BM25Model
from models.embedding_model import EmbeddingModel
from models.hybrid_model import HybridModel
from models.tfidf_model import TFIDFModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# Add preprocessing service to sys.path so TextPreprocessor is importable.
# Use append (not insert at 0) so services/retrieval/main.py stays first on
# sys.path — inserting at 0 would cause uvicorn --reload to find
# services/preprocessing/main.py instead of this file on the next hot-reload.
_PREPROC_PATH = os.path.join(_ROOT, "services", "preprocessing")
if _PREPROC_PATH not in sys.path:
    sys.path.append(_PREPROC_PATH)

_models: Dict[str, Any] = {
    "tfidf":     TFIDFModel(),
    "bm25":      BM25Model(),
    "embedding": EmbeddingModel(),
    "hybrid":    HybridModel(),
}

# ---------------------------------------------------------------------------
# Built-in sample corpus (60 IR-domain documents — no download needed)
# ---------------------------------------------------------------------------

_SAMPLE_DOCS = [
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

# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _models_dir() -> str:
    dataset = os.getenv("DEFAULT_DATASET", "dataset1")
    return os.path.join(_ROOT, "data", "indexes", dataset, "models")


def _load_from_disk(models_dir: str) -> List[str]:
    paths = {
        "tfidf":     os.path.join(models_dir, "tfidf.joblib"),
        "bm25":      os.path.join(models_dir, "bm25.pkl"),
        "embedding": os.path.join(models_dir, "embedding"),
        "hybrid":    os.path.join(models_dir, "hybrid"),
    }
    loaded = []
    for name, path in paths.items():
        if os.path.exists(path):
            try:
                _models[name].load(path)
                loaded.append(name)
                logger.info("Loaded %-10s from disk.", name)
            except Exception as exc:
                logger.warning("Could not load %s: %s", name, exc)
    return loaded


def _get_preprocessor():
    from preprocessor import TextPreprocessor  # noqa: PLC0415
    return TextPreprocessor(do_stemming=True, remove_stopwords=True)


def _tokenise(preprocessor, docs: List[Dict]) -> List[Dict]:
    return [preprocessor.preprocess_document(dict(d)) for d in docs]


def _auto_fit_sparse(save_dir: str) -> None:
    """Fit and save TF-IDF + BM25 inline — fast, no download required."""
    logger.info("Auto-fitting TF-IDF and BM25 with %d built-in documents…", len(_SAMPLE_DOCS))
    try:
        preprocessor  = _get_preprocessor()
        tokenised     = _tokenise(preprocessor, _SAMPLE_DOCS)

        _models["tfidf"].fit(tokenised)
        _models["bm25"].fit(tokenised)
        logger.info("TF-IDF and BM25 fitted in memory.")

        os.makedirs(save_dir, exist_ok=True)
        _models["tfidf"].save(os.path.join(save_dir, "tfidf.joblib"))
        _models["bm25"].save(os.path.join(save_dir, "bm25.pkl"))
        logger.info("Sparse models saved to '%s'.", save_dir)
    except Exception as exc:
        logger.error("Auto-fit (sparse) failed: %s", exc)


def _fit_embedding_and_hybrid(save_dir: str) -> None:
    """Fit embedding + hybrid in background — HybridModel.fit() handles the transformer."""
    logger.info("[bg] Fitting embedding + hybrid (may download ~90 MB on first run)…")
    try:
        preprocessor = _get_preprocessor()
        tokenised    = _tokenise(preprocessor, _SAMPLE_DOCS)
        tokens_map   = {d["doc_id"]: d["tokens"] for d in tokenised}

        # HybridModel.fit() calls load_model() + build_index() on its own
        # internal EmbeddingModel — the transformer is fully ready after this.
        hyb = HybridModel()
        hyb.fit(_SAMPLE_DOCS, tokens_map)

        os.makedirs(save_dir, exist_ok=True)
        hyb.embedding.save(os.path.join(save_dir, "embedding"))
        hyb.save(os.path.join(save_dir, "hybrid"))

        # Share hyb's EmbeddingModel so both models use the same live transformer.
        _models["hybrid"]    = hyb
        _models["embedding"] = hyb.embedding
        logger.info("[bg] Embedding + hybrid ready (%d docs).", len(_SAMPLE_DOCS))

    except Exception as exc:
        logger.error("[bg] Embedding/hybrid fit failed: %s", exc)


def _restore_embedding_model(save_dir: str) -> None:
    """Called in background when FAISS index exists on disk but transformer isn't loaded."""
    logger.info("[bg] Restoring sentence-transformer from HuggingFace cache…")
    try:
        _models["embedding"].load_model()
        logger.info("[bg] Sentence-transformer ready (standalone embedding).")
    except Exception as exc:
        logger.error("[bg] Failed for standalone embedding: %s", exc)
    try:
        hyb = _models.get("hybrid")
        if hyb is not None and hasattr(hyb, "embedding"):
            hyb.embedding.load_model()
            logger.info("[bg] Sentence-transformer ready (HybridModel.embedding).")
    except Exception as exc:
        logger.error("[bg] Failed for HybridModel.embedding: %s", exc)


def _try_load_models() -> None:
    mdir   = _models_dir()
    loaded = _load_from_disk(mdir)

    # Ensure sparse models are always available immediately.
    if "tfidf" not in loaded or "bm25" not in loaded:
        _auto_fit_sparse(mdir)

    if "embedding" not in loaded or "hybrid" not in loaded:
        # Fit from scratch in background.
        logger.info("Embedding/hybrid missing — starting background fit thread.")
        t = threading.Thread(target=_fit_embedding_and_hybrid, args=(mdir,), daemon=True)
        t.start()
    else:
        # FAISS index loaded from disk but sentence-transformer not restored — fix in background.
        logger.info("All models loaded from disk: %s", loaded)
        t = threading.Thread(target=_restore_embedding_model, args=(mdir,), daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _try_load_models()
    yield


app = FastAPI(title="Retrieval Service", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    query_tokens: Optional[List[str]] = None
    model: str = "bm25"
    top_k: int = 10
    mode: Optional[str] = None
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None


class SearchResult(BaseModel):
    doc_id: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    model: str
    results: List[SearchResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    fitted = {name: _is_fitted(_models[name]) for name in _models}
    return {"status": "ok", "service": "retrieval", "models_fitted": fitted}


@app.post("/reload")
def reload_models():
    _try_load_models()
    fitted = {name: _is_fitted(_models[name]) for name in _models}
    return {"reloaded": True, "models_fitted": fitted}


@app.get("/models")
def list_models():
    fitted = {name: _is_fitted(_models[name]) for name in _models}
    return {"models": list(_models.keys()), "fitted": fitted}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if req.model not in _models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model}")

    tokens = req.query_tokens if req.query_tokens else req.query.split()
    model  = _models[req.model]

    try:
        if req.model == "tfidf":
            raw = model.search(tokens, top_k=req.top_k)

        elif req.model == "bm25":
            kw: Dict[str, Any] = {"top_k": req.top_k}
            if req.bm25_k1 is not None:
                kw["k1"] = req.bm25_k1
            if req.bm25_b is not None:
                kw["b"] = req.bm25_b
            raw = model.search(tokens, **kw)

        elif req.model == "embedding":
            raw = model.search(req.query, top_k=req.top_k)

        elif req.model == "hybrid":
            kw = {"top_k": req.top_k}
            if req.mode:
                kw["mode"] = req.mode
            raw = model.search(req.query, tokens, **kw)

        else:
            raw = model.search(req.query, top_k=req.top_k)

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SearchResponse(
        query=req.query,
        model=req.model,
        results=[SearchResult(**r) for r in raw],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_fitted(model: Any) -> bool:
    try:
        if hasattr(model, "_bm25"):   return model._bm25 is not None
        if hasattr(model, "_matrix"): return model._matrix is not None
        if hasattr(model, "_index") and hasattr(model, "_doc_ids"):
            return model._index is not None and len(model._doc_ids) > 0
        if hasattr(model, "bm25"):    return model.bm25._bm25 is not None
    except Exception:
        pass
    return False


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RETRIEVAL_PORT", 8003))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
