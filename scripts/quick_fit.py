"""
quick_fit.py — Fit all retrieval models with a built-in sample dataset.

No external downloads required. Run this once after starting the services to
enable search without waiting for large IR dataset downloads.

Usage
-----
    python scripts/quick_fit.py                 # fits dataset1, includes embedding
    python scripts/quick_fit.py --no-embedding  # skip embedding (faster, BM25+TF-IDF only)
    python scripts/quick_fit.py --dataset dataset2
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "services", "preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "services", "retrieval"))

# ---------------------------------------------------------------------------
# Sample corpus — 60 IR-domain documents
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    # Information Retrieval fundamentals
    {"doc_id": "d001", "text": "TF-IDF stands for term frequency inverse document frequency. It reflects how important a word is to a document in a collection by combining local term frequency with a global penalty for common terms."},
    {"doc_id": "d002", "text": "BM25 is a probabilistic bag-of-words retrieval function that ranks documents by query term frequency, penalising very long documents and saturating repeated terms with the k1 parameter."},
    {"doc_id": "d003", "text": "Dense retrieval encodes queries and documents into shared vector spaces using neural networks, enabling semantic similarity search beyond exact keyword matching."},
    {"doc_id": "d004", "text": "An inverted index maps each vocabulary term to the list of documents containing it along with positional and frequency information, enabling fast postings lookup during retrieval."},
    {"doc_id": "d005", "text": "Precision measures the fraction of retrieved documents that are relevant. Recall measures the fraction of all relevant documents that were retrieved. Both are key IR evaluation metrics."},
    {"doc_id": "d006", "text": "NDCG, Normalised Discounted Cumulative Gain, evaluates ranking quality by awarding higher credit to relevant documents appearing near the top of the result list."},
    {"doc_id": "d007", "text": "Mean Average Precision (MAP) averages the precision at each relevant document rank across all queries, giving a single number summarising ranked retrieval quality."},
    {"doc_id": "d008", "text": "Query expansion adds related terms to a user query to improve retrieval recall. Common techniques include WordNet synonym lookup and pseudo-relevance feedback from top results."},
    {"doc_id": "d009", "text": "The Vector Space Model represents documents and queries as vectors in a high-dimensional term space. Cosine similarity between query and document vectors determines relevance scores."},
    {"doc_id": "d010", "text": "FAISS (Facebook AI Similarity Search) is a library for efficient approximate nearest-neighbour search over dense floating-point vectors, widely used in dense retrieval pipelines."},
    # Neural retrieval and embeddings
    {"doc_id": "d011", "text": "Transformer models use self-attention mechanisms to produce contextualised token representations, capturing long-range dependencies that earlier recurrent models struggled with."},
    {"doc_id": "d012", "text": "BERT is a bidirectional transformer pre-trained with masked language modelling on large corpora and can be fine-tuned on tasks such as passage retrieval and question answering."},
    {"doc_id": "d013", "text": "Sentence-BERT produces fixed-size sentence embeddings by adding a pooling layer to BERT and training with siamese networks on natural language inference data."},
    {"doc_id": "d014", "text": "Word2Vec learns distributed word representations by predicting a word from its context window (CBOW) or predicting context from a central word (Skip-gram)."},
    {"doc_id": "d015", "text": "Bi-encoder models encode queries and documents independently, making them suitable for large-scale retrieval since document embeddings can be precomputed and indexed offline."},
    {"doc_id": "d016", "text": "Cross-encoder models concatenate query and document tokens and pass the pair through a transformer, producing accurate relevance scores but requiring one forward pass per candidate."},
    {"doc_id": "d017", "text": "Contrastive learning trains encoders to place semantically similar pairs close together and dissimilar pairs far apart in embedding space, using in-batch negatives for efficiency."},
    {"doc_id": "d018", "text": "MS-MARCO DistilBERT is a lightweight bi-encoder fine-tuned on the MS-MARCO passage ranking dataset, offering good retrieval quality at lower inference cost than full BERT."},
    {"doc_id": "d019", "text": "ColBERT (Contextualised Late Interaction over BERT) computes a MaxSim score between all query and document token embeddings, balancing expressiveness and retrieval efficiency."},
    {"doc_id": "d020", "text": "Knowledge distillation trains a small student model to mimic the output of a larger teacher model, reducing inference cost while retaining much of the teacher's performance."},
    # Hybrid retrieval and fusion
    {"doc_id": "d021", "text": "Hybrid retrieval combines sparse lexical models like BM25 with dense semantic models to leverage complementary matching signals, improving performance over either alone."},
    {"doc_id": "d022", "text": "Reciprocal Rank Fusion (RRF) merges multiple ranked lists by assigning each document a score of 1 divided by k plus its rank in each list, then summing across lists."},
    {"doc_id": "d023", "text": "Linear score fusion normalises each model's scores to a common range and computes a weighted sum, requiring careful calibration of model-specific weights."},
    {"doc_id": "d024", "text": "Multi-stage retrieval uses a fast first-stage ranker such as BM25 to produce a candidate set, then applies a more expensive neural re-ranker to the candidates only."},
    {"doc_id": "d025", "text": "Re-ranking with a cross-encoder on the top-100 BM25 candidates typically improves nDCG@10 substantially while keeping end-to-end latency acceptable for interactive search."},
    # Search engines and infrastructure
    {"doc_id": "d026", "text": "PageRank models the web as a graph and assigns each page an importance score proportional to the number and quality of inbound hyperlinks, used as a ranking signal in web search."},
    {"doc_id": "d027", "text": "Elasticsearch is a distributed search engine built on Apache Lucene that supports full-text search, real-time indexing, and aggregations at scale."},
    {"doc_id": "d028", "text": "Query understanding identifies spelling errors, expands abbreviations, classifies intent, and maps queries to structured representations before retrieval begins."},
    {"doc_id": "d029", "text": "Faceted search lets users iteratively refine results by applying attribute filters such as category, price range, or date, turning a single query into a navigational experience."},
    {"doc_id": "d030", "text": "Document indexing transforms raw text into an inverted index through tokenisation, normalisation, and posting list construction, enabling millisecond-latency retrieval over millions of documents."},
    # NLP preprocessing
    {"doc_id": "d031", "text": "Tokenisation splits raw text into individual tokens such as words, subwords, or characters. It is the first step in virtually every natural language processing pipeline."},
    {"doc_id": "d032", "text": "Porter stemming removes morphological suffixes to reduce inflected words to a common stem. For example, running, runs, and runner all reduce to run."},
    {"doc_id": "d033", "text": "WordNet lemmatisation converts each token to its dictionary base form using part-of-speech context, producing linguistically valid words unlike stemming."},
    {"doc_id": "d034", "text": "Stop word removal discards high-frequency function words such as the, is, and at that carry little discriminative information for term-based retrieval models."},
    {"doc_id": "d035", "text": "Named entity recognition (NER) identifies spans of text referring to people, organisations, locations, dates, and other entity types, useful for structured query processing."},
    {"doc_id": "d036", "text": "Part-of-speech tagging assigns grammatical categories to each token in a sentence, enabling downstream tasks like lemmatisation, chunking, and syntactic parsing."},
    {"doc_id": "d037", "text": "Spell correction detects and repairs typographical errors in user queries before retrieval, improving recall for misspelled terms that would otherwise match no documents."},
    {"doc_id": "d038", "text": "Text normalisation converts text to a canonical form by lowercasing, expanding contractions, removing punctuation, and replacing digits and special characters."},
    {"doc_id": "d039", "text": "Sub-word tokenisation algorithms such as BPE and WordPiece split rare words into smaller units, helping neural models handle out-of-vocabulary terms gracefully."},
    {"doc_id": "d040", "text": "Synonym expansion uses lexical resources like WordNet or word embeddings to identify terms with similar meaning and add them to a query to improve retrieval recall."},
    # Evaluation benchmarks and methodology
    {"doc_id": "d041", "text": "TREC (Text REtrieval Conference) provides standardised test collections, queries, and relevance judgements that enable rigorous comparable evaluation of retrieval systems."},
    {"doc_id": "d042", "text": "MS MARCO contains real user queries from Bing search logs with sparse human-annotated relevance labels, making it a challenging large-scale retrieval benchmark."},
    {"doc_id": "d043", "text": "BEIR is a zero-shot retrieval benchmark covering 18 diverse datasets spanning biomedical, financial, scientific, and news domains, testing generalisation beyond training data."},
    {"doc_id": "d044", "text": "Relevance judgements (qrels) provide binary or graded human assessments of document relevance for each query and form the ground truth used to compute evaluation metrics."},
    {"doc_id": "d045", "text": "Pooling creates evaluation collections by merging the top results from multiple retrieval systems and presenting them to assessors, reducing annotation cost while maintaining coverage."},
    {"doc_id": "d046", "text": "Statistical significance tests such as paired t-tests and Wilcoxon signed-rank tests determine whether performance differences between retrieval systems are likely due to chance."},
    {"doc_id": "d047", "text": "Online evaluation with A/B testing exposes real users to competing retrieval systems and measures implicit feedback such as click-through rate and session abandonment."},
    {"doc_id": "d048", "text": "Diversity and novelty metrics like alpha-NDCG measure whether results cover multiple subtopics of an ambiguous query rather than returning redundant relevant documents."},
    # Applications and advanced topics
    {"doc_id": "d049", "text": "Open-domain question answering retrieves candidate passages from a large corpus and then applies a reader model to extract or generate a precise answer to a natural language question."},
    {"doc_id": "d050", "text": "Retrieval augmented generation (RAG) combines a dense retrieval model with a generative language model, conditioning generation on retrieved context to improve factual accuracy."},
    {"doc_id": "d051", "text": "Passage retrieval selects short text segments rather than whole documents, improving both relevance precision and answer extraction accuracy for question answering tasks."},
    {"doc_id": "d052", "text": "Learning to rank trains a supervised model on query-document feature vectors to produce relevance rankings that outperform hand-tuned weighting schemes."},
    {"doc_id": "d053", "text": "Approximate nearest-neighbour algorithms like HNSW and IVF-Flat enable sub-millisecond similarity search over billions of vectors at the cost of a small recall trade-off."},
    {"doc_id": "d054", "text": "Document clustering organises a corpus into thematically coherent groups using algorithms such as k-means or hierarchical clustering over TF-IDF or embedding representations."},
    {"doc_id": "d055", "text": "Entity linking connects ambiguous surface forms in text to unique knowledge base entries, enabling structured retrieval and question answering over entity-rich corpora."},
    {"doc_id": "d056", "text": "Conversational search requires tracking dialogue context across multiple turns to resolve coreferences and ellipsis before retrieving documents relevant to the current information need."},
    {"doc_id": "d057", "text": "Zero-shot retrieval evaluates models on domains not seen during training to assess generalisation, typically using the BEIR benchmark and a pre-trained encoder with no fine-tuning."},
    {"doc_id": "d058", "text": "Index compression techniques such as variable-byte encoding and FOR-delta reduce the disk footprint of inverted indices while keeping decompression fast during query processing."},
    {"doc_id": "d059", "text": "Query latency is a critical operational metric for production search systems; sub-100ms end-to-end response times typically require pre-computed indexes and approximate retrieval."},
    {"doc_id": "d060", "text": "Pseudo-relevance feedback assumes the top-k retrieved documents are relevant and extracts high-weight terms to expand the original query before a second retrieval pass."},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ok(msg: str) -> None:
    print(f"  ✔  {msg}")

def _step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit retrieval models with built-in sample data")
    parser.add_argument("--dataset", default="dataset1", choices=["dataset1", "dataset2"],
                        help="Target dataset slot (default: dataset1)")
    parser.add_argument("--no-embedding", action="store_true",
                        help="Skip embedding model fitting (faster, no model download)")
    args = parser.parse_args()

    total_steps = 3 if args.no_embedding else 4

    print(f"\n{'='*60}")
    print(f"  IR System — Quick Fit")
    print(f"  Documents : {len(SAMPLE_DOCUMENTS)}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Embedding : {'disabled' if args.no_embedding else 'enabled'}")
    print(f"{'='*60}")

    # ── 1. Preprocess ─────────────────────────────────────────────────────
    _step(1, total_steps, "Preprocessing documents")
    from preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor(do_stemming=True, remove_stopwords=True)
    tokenised_docs = [preprocessor.preprocess_document(dict(doc)) for doc in SAMPLE_DOCUMENTS]
    _ok(f"{len(tokenised_docs)} documents tokenised")

    models_dir = os.path.join(ROOT, "data", "indexes", args.dataset, "models")
    os.makedirs(models_dir, exist_ok=True)

    # ── 2. TF-IDF ─────────────────────────────────────────────────────────
    _step(2, total_steps, "Fitting TF-IDF model")
    from models.tfidf_model import TFIDFModel
    tfidf = TFIDFModel()
    tfidf.fit(tokenised_docs)
    tfidf_path = os.path.join(models_dir, "tfidf.joblib")
    tfidf.save(tfidf_path)
    stats = tfidf.get_stats()
    _ok(f"TF-IDF saved — {stats['num_documents']} docs, {stats['vocab_size']} vocab terms")

    # ── 3. BM25 ───────────────────────────────────────────────────────────
    _step(3, total_steps, "Fitting BM25 model")
    from models.bm25_model import BM25Model
    bm25 = BM25Model()
    bm25.fit(tokenised_docs)
    bm25_path = os.path.join(models_dir, "bm25.pkl")
    bm25.save(bm25_path)
    stats = bm25.get_stats()
    _ok(f"BM25 saved — {stats['num_documents']} docs, avg_dl={stats['avg_doc_length']:.1f}")

    # ── 4. Embedding (optional) ────────────────────────────────────────────
    if not args.no_embedding:
        _step(4, total_steps, "Fitting Embedding model (downloading ~90MB model, please wait)")
        from models.embedding_model import EmbeddingModel
        emb = EmbeddingModel()
        emb.load_model()
        raw_docs = [{"doc_id": d["doc_id"], "text": d["text"]} for d in SAMPLE_DOCUMENTS]
        emb.build_index(raw_docs)
        emb_path = os.path.join(models_dir, "embedding")
        emb.save(emb_path)
        _ok(f"Embedding model saved to {emb_path}/")
    else:
        print(f"\n[4/{total_steps}] Embedding model — skipped (--no-embedding)")

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Models saved to: {models_dir}")
    print(f"{'='*60}")
    print()
    print("  Next step: reload the retrieval service so it picks up the")
    print("  saved models.  Either:")
    print()
    print("    Option A — POST to reload endpoint (no restart needed):")
    print("      curl -X POST http://localhost:8003/reload")
    print()
    print("    Option B — Restart the retrieval terminal in VS Code")
    print()


if __name__ == "__main__":
    main()
