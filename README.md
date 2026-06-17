# IR Search Engine

A modular, production-ready Information Retrieval system built on a **Service-Oriented Architecture (SOA)**. Each capability runs as an independent microservice communicating over HTTP, making the system easy to scale, test, and extend.

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
  ┌───────────┐ ┌───────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
  │Preprocess │ │ Index │ │Retrieval│ │ Refine  │ │ Evaluate │
  │  :8001    │ │ :8002 │ │  :8003  │ │  :8005  │ │  :8006   │
  └───────────┘ └───────┘ └────────┘ └─────────┘ └──────────┘
```

### Services

| Service | Port | Responsibility |
|---|---|---|
| **api_gateway** | 8000 | Single entry point; orchestrates all other services |
| **preprocessing** | 8001 | Tokenisation, stopword removal, stemming/lemmatisation |
| **indexing** | 8002 | Inverted index build, persistence, and postings lookup |
| **retrieval** | 8003 | TF-IDF, BM25, Embedding, and Hybrid retrieval models |
| **query_refinement** | 8005 | Spell correction, synonym expansion, history boosting |
| **evaluation** | 8006 | MAP, Recall, P@k, nDCG@k metrics and comparison reports |

### Retrieval Models

| Model | Type | Description |
|---|---|---|
| **TF-IDF** | Sparse | Vector Space Model with cosine similarity (sklearn) |
| **BM25** | Sparse | Probabilistic ranking — Okapi BM25 (rank-bm25) |
| **Embedding** | Dense | Semantic search with sentence-transformers + FAISS |
| **Hybrid (Serial)** | Pipeline | BM25 candidate retrieval → embedding re-rank |
| **Hybrid (Parallel)** | Fusion | RRF or weighted fusion of all three models |

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

The pipeline script loads datasets, preprocesses documents, builds indexes, fits all models, and writes an evaluation report.

```bash
# Default: 10 000 docs per dataset
python run_pipeline.py

# Custom datasets and sample size
python run_pipeline.py --dataset1 msmarco-passage --dataset2 beir/nq/train --sample_size 50000

# Full dataset (slow — no sample limit)
python run_pipeline.py --sample_size 0

# Skip fitting (load from disk)
python run_pipeline.py --skip_fitting
```

Output is written to:
```
data/
├── datasets/dataset1/sample.json
├── datasets/dataset2/sample.json
├── indexes/dataset1/
│   ├── dataset1.pkl          # inverted index
│   └── models/               # tfidf, bm25, embedding, hybrid
├── indexes/dataset2/
└── evaluation_report_dataset1.md
```

---

## Starting All Services

### Option 1 — Bash script (local, no Docker)

```bash
bash scripts/start_all.sh
```

This starts each service with uvicorn in the background, waits for every `/health` endpoint to respond, then prints the service URLs.

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
- Select a **dataset** (MS MARCO or NQ) and a **retrieval model**.
- When **BM25** is selected, two sliders appear for live k1/b tuning.
- Check **Use Query Refinement** to apply spell correction and synonym expansion.
- Results appear as cards with rank, document ID, score, and a text snippet.

### Evaluate Tab
- Select models to compare and click **Run Evaluation**.
- Results are shown in a table and a MAP bar chart (Chart.js).

### Settings Tab
- Toggle stemming, lemmatisation, and stopword removal.
- Set the default number of results (5–50).
- Settings persist in `localStorage`.

---

## API Reference

Interactive docs available at **http://localhost:8000/docs** once the gateway is running.

### Key endpoints

```
POST /search           — search with any model
GET  /health           — health status of all services
GET  /datasets         — list available datasets
GET  /models           — list available models
POST /index/build      — trigger index build for a dataset
POST /evaluate         — run evaluation metrics
```

### Search example

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is information retrieval",
    "dataset": "dataset1",
    "model": "bm25",
    "top_k": 10,
    "use_refinement": true
  }'
```

---

## Project Structure

```
ir-system/
├── services/
│   ├── preprocessing/          # TextPreprocessor + DatasetLoader
│   ├── indexing/               # InvertedIndex
│   ├── retrieval/
│   │   └── models/             # TFIDFModel, BM25Model, EmbeddingModel, HybridModel
│   ├── query_refinement/       # QueryRefiner (spell, synonyms, history)
│   ├── evaluation/             # IREvaluator (MAP, Recall, P@k, nDCG@k)
│   └── api_gateway/            # Orchestration gateway
├── frontend/                   # HTML/CSS/JS UI
├── data/
│   ├── datasets/               # Raw + sampled dataset files
│   └── indexes/                # Persisted indexes and model files
├── scripts/
│   └── start_all.sh            # Local startup script
├── run_pipeline.py             # End-to-end pipeline runner
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
