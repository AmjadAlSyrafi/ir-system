"""
End-to-end IR pipeline runner.

Loads datasets, preprocesses documents, builds inverted indexes,
fits all four retrieval models, and produces an evaluation report.

Usage
-----
python run_pipeline.py [--dataset1 msmarco-passage] [--dataset2 beir/nq/train]
                       [--sample_size 10000] [--skip_fitting]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

# Add service directories to path so we can import directly.
_ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_ROOT, "services", "preprocessing"))
sys.path.insert(0, os.path.join(_ROOT, "services", "indexing"))
sys.path.insert(0, os.path.join(_ROOT, "services", "retrieval", "models"))

from dataset_loader import DatasetLoader          # noqa: E402
from preprocessor import TextPreprocessor         # noqa: E402
from indexer import InvertedIndex                 # noqa: E402
from tfidf_model import TFIDFModel                # noqa: E402
from bm25_model import BM25Model                  # noqa: E402
from embedding_model import EmbeddingModel        # noqa: E402
from hybrid_model import HybridModel              # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR       = os.path.join(_ROOT, "data")
INDEX_DIR      = os.path.join(DATA_DIR, "indexes")
DATASET_DIR    = os.path.join(DATA_DIR, "datasets")
EVAL_REPORT    = os.path.join(DATA_DIR, "evaluation_report.md")

for _d in (INDEX_DIR, DATASET_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_preprocess(
    dataset_name: str,
    sample_size: int,
    preprocessor: TextPreprocessor,
    loader: DatasetLoader,
) -> List[Dict]:
    """Load up to *sample_size* documents and preprocess them.

    Returns a list of ``{doc_id, text, tokens, processed_text}`` dicts.
    """
    logger.info("Loading documents from '%s' (sample_size=%d)…", dataset_name, sample_size)
    raw_docs = []
    for i, doc in enumerate(loader.load_documents(dataset_name)):
        if sample_size > 0 and i >= sample_size:
            break
        raw_docs.append(doc)

    logger.info("Preprocessing %d documents…", len(raw_docs))
    t0 = time.perf_counter()
    processed = [preprocessor.preprocess_document(d) for d in raw_docs]
    logger.info("Preprocessing done in %.1fs.", time.perf_counter() - t0)
    return processed


def _build_and_save_index(
    dataset_tag: str,
    documents: List[Dict],
) -> InvertedIndex:
    """Build inverted index and save to disk, return the index object."""
    index = InvertedIndex(
        index_name=dataset_tag,
        storage_path=INDEX_DIR,
    )
    index.build_from_dataset(documents)
    path = index.save()
    logger.info("Inverted index saved: %s", path)
    return index


def _fit_and_save_models(
    dataset_tag: str,
    documents: List[Dict],
) -> Dict:
    """Fit TF-IDF, BM25, Embedding, and Hybrid models and save to disk."""
    models_dir = os.path.join(INDEX_DIR, dataset_tag, "models")
    os.makedirs(models_dir, exist_ok=True)

    tokens_map = {d["doc_id"]: d["tokens"] for d in documents}

    # TF-IDF
    logger.info("[%s] Fitting TF-IDF…", dataset_tag)
    tfidf = TFIDFModel(max_features=50_000)
    tfidf.fit(documents)
    tfidf.save(os.path.join(models_dir, "tfidf.joblib"))

    # BM25
    logger.info("[%s] Fitting BM25…", dataset_tag)
    bm25 = BM25Model(k1=1.5, b=0.75)
    bm25.fit(documents)
    bm25.save(os.path.join(models_dir, "bm25.pkl"))

    # Embedding
    logger.info("[%s] Fitting Embedding model (this may take a while)…", dataset_tag)
    embedding = EmbeddingModel(index_type="flat")
    embedding.load_model()
    embedding.build_index(documents)
    embedding.save(os.path.join(models_dir, "embedding"))

    # Hybrid (parallel, RRF)
    logger.info("[%s] Fitting Hybrid model…", dataset_tag)
    hybrid = HybridModel(mode="parallel", fusion_method="rrf")
    hybrid.fit(documents, tokens_map)
    hybrid.save(os.path.join(models_dir, "hybrid"))

    return {"tfidf": tfidf, "bm25": bm25, "embedding": embedding, "hybrid": hybrid}


def _load_models(dataset_tag: str) -> Dict:
    """Load previously saved models from disk."""
    models_dir = os.path.join(INDEX_DIR, dataset_tag, "models")

    tfidf = TFIDFModel()
    tfidf.load(os.path.join(models_dir, "tfidf.joblib"))

    bm25 = BM25Model()
    bm25.load(os.path.join(models_dir, "bm25.pkl"))

    embedding = EmbeddingModel()
    embedding.load(os.path.join(models_dir, "embedding"))

    hybrid = HybridModel()
    hybrid.load(os.path.join(models_dir, "hybrid"))

    return {"tfidf": tfidf, "bm25": bm25, "embedding": embedding, "hybrid": hybrid}


def _run_evaluation(
    dataset_name: str,
    dataset_tag: str,
    models: Dict,
    loader: DatasetLoader,
    preprocessor: TextPreprocessor,
    k: int = 10,
    max_queries: int = 100,
) -> str:
    """Run all models against qrels, produce a Markdown report, return its path."""
    # Delayed import to avoid circular deps at module top.
    sys.path.insert(0, os.path.join(_ROOT, "services", "evaluation"))
    from metrics import IREvaluator  # noqa: E402

    logger.info("[%s] Loading qrels…", dataset_tag)
    try:
        qrels = loader.load_qrels(dataset_name)
    except ValueError as exc:
        logger.warning("No qrels available for '%s': %s", dataset_name, exc)
        return ""

    logger.info("[%s] Running evaluation on up to %d queries…", dataset_tag, max_queries)
    evaluator = IREvaluator(qrels)

    # Build results for each model over the available queries.
    query_ids = list(qrels.keys())[:max_queries]
    model_results: Dict[str, Dict] = {name: {} for name in models}

    for query_id in query_ids:
        # Load and preprocess the query text (use query_id as surrogate text
        # when actual text is unavailable — real deployments would fetch it).
        query_text = query_id  # placeholder; replace with real query lookup
        query_tokens = preprocessor.preprocess(query_text)

        for name, model in models.items():
            try:
                if name in ("tfidf",):
                    results = model.search(query_tokens, top_k=k)
                elif name == "bm25":
                    results = model.search(query_tokens, top_k=k)
                elif name == "embedding":
                    results = model.search(query_text, top_k=k)
                else:  # hybrid
                    results = model.search(query_text, query_tokens, top_k=k)
                model_results[name][query_id] = results
            except Exception as exc:
                logger.warning("[%s] %s search failed for query '%s': %s", dataset_tag, name, query_id, exc)
                model_results[name][query_id] = []

    df = evaluator.compare_models(model_results, k=k)
    print(f"\n=== Evaluation Results — {dataset_tag} ===")
    print(df.to_string())

    report_path = os.path.join(DATA_DIR, f"evaluation_report_{dataset_tag}.md")
    evaluator.generate_report(df, report_path)
    logger.info("Report saved to '%s'.", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="IR system end-to-end pipeline")
    parser.add_argument("--dataset1", default="msmarco-passage")
    parser.add_argument("--dataset2", default="beir/nq/train")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10_000,
        help="Documents to load per dataset (0 = full dataset).",
    )
    parser.add_argument(
        "--skip_fitting",
        action="store_true",
        help="Load models from disk instead of fitting from scratch.",
    )
    args = parser.parse_args()

    preprocessor = TextPreprocessor(
        language="english",
        do_stemming=True,
        do_lemmatization=False,
        remove_stopwords=True,
        min_token_length=2,
    )
    loader = DatasetLoader()

    datasets = [
        (args.dataset1, "dataset1"),
        (args.dataset2, "dataset2"),
    ]

    for dataset_name, dataset_tag in datasets:
        print(f"\n{'='*60}\nDataset: {dataset_name}  ({dataset_tag})\n{'='*60}")

        try:
            stats = loader.get_dataset_stats(dataset_name)
            print(f"  Docs:    {stats['num_docs']:>10,}")
            print(f"  Queries: {stats['num_queries']:>10,}")
            print(f"  Qrels:   {stats['num_qrels']:>10,}")
        except ValueError as exc:
            logger.error("Could not get stats for '%s': %s", dataset_name, exc)
            continue

        # Save sample
        sample_path = os.path.join(DATASET_DIR, dataset_tag, "sample.json")
        try:
            loader.save_sample(dataset_name, n=1000, output_path=sample_path)
        except ValueError as exc:
            logger.warning("Could not save sample: %s", exc)

        # Preprocess documents
        try:
            documents = _load_and_preprocess(
                dataset_name, args.sample_size, preprocessor, loader
            )
        except ValueError as exc:
            logger.error("Could not load documents: %s", exc)
            continue

        # Build inverted index
        _build_and_save_index(dataset_tag, documents)

        # Fit or load models
        if args.skip_fitting:
            logger.info("[%s] Loading models from disk…", dataset_tag)
            try:
                models = _load_models(dataset_tag)
            except FileNotFoundError as exc:
                logger.error("Model files not found: %s", exc)
                continue
        else:
            models = _fit_and_save_models(dataset_tag, documents)

        # Evaluation
        _run_evaluation(dataset_name, dataset_tag, models, loader, preprocessor)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()