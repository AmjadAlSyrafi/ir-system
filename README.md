# IR Search Engine

A modular, production-ready Information Retrieval system built on a **Service-Oriented Architecture (SOA)**. Each capability runs as an independent FastAPI microservice communicating over HTTP, making the system easy to scale, test, and extend.

---

## Architecture

```
                    ┌──────────────────────┐
                    │     Frontend UI      │
                    │  (frontend/index.html)│
                    └──────────┬───────────┘
                               │ HTTP
                    ┌──────────▼───────────┐
                    │    API Gateway       │  :8000
                    │  (api_gateway)       │
                    └──┬──┬──┬──┬──┬──────┘
                       │  │  │  │  │
         ┌─────────────┘  │  │  │  └──────────────┐
         │         ┌──────┘  │  └──────┐           │
         ▼         ▼         ▼         ▼           ▼
  ┌───────────┐ ┌───────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
  │Preprocess │ │ Index │ │Retrieval│ │ Refine  │ │ Evaluate │
  │  :8001    │ │ :8002 │ │  :8003  │ │  :8005  │ │  :8006   │
  └───────────┘ └───────┘ └─────────┘ └─────────┘ └──────────┘
```

### Services

| Service | Port | Responsibility |
|---|---|---|
| **api_gateway** | 8000 | Single entry point; orchestrates all other services |
| **preprocessing** | 8001 | Tokenisation, stopword removal, stemming/lemmatisation |
| **indexing** | 8002 | Inverted index build, SQLite document store, postings lookup |
| **retrieval** | 8003 | TF-IDF, BM25, Embedding, and Hybrid retrieval — all datasets loaded simultaneously |
| **query_refinement** | 8005 | Spell correction, synonym expansion, history boosting |
| **evaluation** | 8006 | MAP, Recall, P@k, nDCG@k metrics and comparison reports |

### Retrieval Models

| Model | Type | Description |
|---|---|---|
| **TF-IDF** | Sparse | Vector Space Model with cosine similarity (sklearn) |
| **BM25** | Sparse | Probabilistic ranking — Okapi BM25 (rank-bm25), stored bzip2-compressed |
| **Embedding** | Dense | Semantic search with sentence-transformers + FAISS |
| **Hybrid (Serial)** | Pipeline | BM25 candidate retrieval → embedding re-rank |
| **Hybrid (Parallel)** | Fusion | RRF or weighted linear fusion of all three models |

---

## Datasets

| Tag | Dataset | Corpus Size | Domain |
|---|---|---|---|
| **dataset1** | `beir/quora/test` | ~522 k documents | Duplicate question detection |
| **dataset2** | `beir/hotpotqa/test` | ~5.2 M documents | Multi-hop factoid QA |

The pipeline samples 10 000 documents from each corpus by default. Only queries whose relevant documents appear in the sampled index are used for evaluation (corpus-aware filtering).

---

## Evaluation Results

Results from running `python run_pipeline.py --sample_size 10000`.

### Dataset 1 — Quora (`beir/quora/test`)

| Model | MAP | P@10 | nDCG@10 | Recall |
|---|---|---|---|---|
| TF-IDF | 0.7176 | 0.116 | 0.7765 | 0.7496 |
| BM25 | 0.7151 | 0.117 | 0.7748 | 0.7498 |
| **Embedding** | **0.7348** | 0.118 | **0.7911** | 0.7472 |
| Hybrid | 0.7275 | **0.119** | 0.7850 | 0.7474 |

### Dataset 2 — HotpotQA (`beir/hotpotqa/test`)

| Model | MAP | P@10 | nDCG@10 | Recall |
|---|---|---|---|---|
| TF-IDF | 0.2726 | 0.075 | 0.3645 | 0.3750 |
| **BM25** | **0.3137** | 0.084 | **0.4153** | 0.4200 |
| Embedding | 0.2119 | 0.060 | 0.2854 | 0.3000 |
| Hybrid | 0.3088 | **0.085** | 0.4113 | **0.4250** |

---

## Installation

### Prerequisites

- Python 3.10+
- (optional) Docker & Docker Compose for containerised deployment

### Install dependencies

```bash
cd ir-system
pip install -r requirements.txt

# Download NLTK data (done automatically on first run, but you can pre-fetch):
python -c "import nltk; [nltk.download(r, quiet=True) for r in ('stopwords','wordnet','punkt','omw-1.4','averaged_perceptron_tagger')]"
```

---

## Running the Full Pipeline

The pipeline script downloads and samples datasets, preprocesses documents, builds all indexes and models, stores the full document text in SQLite, and writes evaluation reports.

```bash
# Default: beir/quora/test + beir/hotpotqa/test, 10 000 docs each
python run_pipeline.py

# Custom sample size
python run_pipeline.py --sample_size 50000

# Full corpus (slow — no sample limit)
python run_pipeline.py --sample_size 0

# Skip refitting — load models from disk
python run_pipeline.py --skip_fitting
```

After the pipeline finishes it automatically pings `POST http://localhost:8003/reload` so the running retrieval service picks up the new models without a restart.

### Output layout

```
data/
├── datasets/
│   ├── dataset1/sample.json          # 10k sampled Quora docs
│   └── dataset2/sample.json          # 10k sampled HotpotQA docs
├── indexes/
│   ├── documents.db                  # SQLite full-text store (both datasets)
│   ├── dataset1/
│   │   ├── dataset1.pkl              # inverted index
│   │   └── models/
│   │       ├── tfidf.joblib
│   │       ├── bm25.pkl              # bzip2-compressed
│   │       ├── embedding/            # faiss.index + meta.pkl
│   │       └── hybrid/
│   └── dataset2/
│       └── models/ ...
├── evaluation_report_dataset1.md
└── evaluation_report_dataset2.md
```

---

## Starting All Services

### Option 1 — Bash script (local, no Docker)

```bash
bash scripts/start_all.sh
```

Starts each service with uvicorn in the background, waits for every `/health` endpoint to respond, then prints the service URLs.

To stop all services:
```bash
kill $(cat logs/*.pid)
```

### Option 2 — Docker Compose

```bash
cp .env.example .env   # configure if needed
docker-compose up --build
```

### Option 3 — Run services individually

```bash
cd services/preprocessing    && uvicorn main:app --port 8001 --reload &
cd services/indexing         && uvicorn main:app --port 8002 --reload &
cd services/retrieval        && uvicorn main:app --port 8003 --reload &
cd services/query_refinement && uvicorn main:app --port 8005 --reload &
cd services/evaluation       && uvicorn main:app --port 8006 --reload &
cd services/api_gateway      && uvicorn main:app --port 8000 --reload &
```

---

## Using the UI

1. Start all services (any option above).
2. Open `frontend/index.html` in your browser (no web server needed — it talks directly to `http://localhost:8000`).

### Search Tab

- Select a **dataset** (Quora or HotpotQA) and a **retrieval model**.
- When **BM25** is selected, two sliders appear for live `k1` / `b` tuning.
- When **Hybrid (Parallel)** is selected, three sliders appear for BM25 / Embedding / TF-IDF fusion weight tuning (values are auto-normalised to 100 %).
- Switch **Mode** to *Basic + Advanced* to enable query refinement (spell correction and synonym expansion).
- Results show the full document text with an expand/collapse button.

### Evaluate Tab

- **Live Evaluation** — select dataset and models, click *Run Evaluation*. Only queries whose relevant documents exist in the sampled index are evaluated.
- **Pipeline Evaluation Reports** — pre-computed metrics from `run_pipeline.py`, shown as a table and a grouped bar chart with the best-MAP model highlighted.

### Settings Tab

- Toggle stemming, lemmatisation, and stopword removal.
- Set the default number of results (5–50).
- Settings persist in `localStorage`.

---

## API Reference

Interactive docs available at **http://localhost:8000/docs** once the gateway is running.

### Key endpoints

```
POST /search                — search with any model
GET  /health                — health status of all services
GET  /datasets              — list available datasets
GET  /models                — list available retrieval models
POST /index/build           — trigger index build for a dataset
POST /evaluate/run          — run live evaluation metrics
GET  /reports/{dataset}     — load pre-computed pipeline evaluation report
POST /documents/batch       — batch full-text lookup by doc IDs
```

### Search example

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is information retrieval",
    "dataset": "dataset1",
    "model": "hybrid_parallel",
    "top_k": 10,
    "use_refinement": true,
    "hybrid_bm25_weight": 0.4,
    "hybrid_embedding_weight": 0.4,
    "hybrid_tfidf_weight": 0.2
  }'
```

### Evaluation example

```bash
curl -X POST http://localhost:8000/evaluate/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "dataset2",
    "models": ["bm25", "embedding", "hybrid_parallel"],
    "max_queries": 20,
    "k": 10
  }'
```

---

## Project Structure

```
ir-system/
├── services/
│   ├── preprocessing/          # TextPreprocessor + DatasetLoader (ir-datasets)
│   ├── indexing/
│   │   ├── main.py             # InvertedIndex build + /documents/batch endpoint
│   │   └── document_store.py   # SQLite full-text store with (doc_id, dataset) key
│   ├── retrieval/
│   │   ├── main.py             # Multi-dataset model router (_models_by_dataset)
│   │   └── models/
│   │       ├── tfidf_model.py
│   │       ├── bm25_model.py   # bzip2-compressed pickle save/load
│   │       ├── embedding_model.py  # sentence-transformers + FAISS
│   │       └── hybrid_model.py     # RRF / weighted fusion; per-query weight overrides
│   ├── query_refinement/       # QueryRefiner (spell, synonyms, history)
│   ├── evaluation/             # IREvaluator (MAP, Recall, P@k, nDCG@k)
│   └── api_gateway/            # Orchestration gateway; full-text enrichment
├── frontend/
│   ├── index.html              # Search / Evaluate / Settings tabs
│   ├── app.js
│   └── style.css
├── data/
│   ├── datasets/               # Sampled dataset JSON files
│   └── indexes/                # Persisted indexes, model files, SQLite store
├── scripts/
│   └── start_all.sh
├── run_pipeline.py             # End-to-end pipeline (index → fit → evaluate → reload)
├── query-test-ref.html         # Quick-reference query playground
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## SOA Design Principles

- **Loose coupling** — services communicate only via HTTP REST
- **High cohesion** — each service owns one responsibility
- **Discoverability** — every service exposes `/health` and OpenAPI docs at `/docs`
- **Scalability** — any service can be scaled independently via Docker replicas
- **Replaceability** — swap any model or preprocessing strategy without touching other services
