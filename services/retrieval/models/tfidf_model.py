"""
TF-IDF Vector Space Model retrieval.

Uses sklearn's TfidfVectorizer backed by scipy sparse matrices.
Cosine similarity is computed via efficient matrix operations.
"""

import logging
import time
from typing import Dict, List, Optional

import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class TFIDFModel:
    """TF-IDF retrieval model using the Vector Space Model.

    Args:
        max_features: Maximum vocabulary size passed to TfidfVectorizer.
    """

    def __init__(self, max_features: int = 50000) -> None:
        self.max_features = max_features
        self._vectorizer: Optional[TfidfVectorizer] = None
        # L2-normalised document matrix (scipy sparse, shape n_docs × vocab).
        self._matrix: Optional[spmatrix] = None
        self._doc_ids: List[str] = []
        # Reverse mapping: doc_id → row index (for O(1) single-doc lookup).
        self._doc_id_to_idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, documents: List[Dict]) -> None:
        """Build the TF-IDF matrix from pre-tokenised documents.

        Args:
            documents: List of dicts with ``doc_id`` (str) and ``tokens``
                (list[str]).  Tokens are joined to a string internally so
                that sklearn's analyser receives the exact vocabulary
                produced by the upstream preprocessing step.
        """
        if not documents:
            raise ValueError("'documents' must not be empty.")

        t0 = time.perf_counter()
        logger.info("Fitting TF-IDF model on %d documents.", len(documents))

        self._doc_ids = [d["doc_id"] for d in documents]
        self._doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}

        # Join pre-tokenised lists back to strings so sklearn tokenisation
        # is a no-op (analyzer="word", token_pattern matches any non-space run).
        corpus = [" ".join(d.get("tokens", [])) for d in documents]

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            analyzer="word",
            token_pattern=r"\S+",  # preserve already-stemmed tokens
            sublinear_tf=True,     # log(1+tf) dampening
        )
        raw_matrix = self._vectorizer.fit_transform(corpus)
        # Pre-normalise rows so cosine similarity reduces to a dot product.
        self._matrix = normalize(raw_matrix, norm="l2", copy=False)

        elapsed = time.perf_counter() - t0
        logger.info(
            "TF-IDF fit complete: %d docs, vocab=%d, elapsed=%.2fs.",
            len(documents),
            len(self._vectorizer.vocabulary_),
            elapsed,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def transform_query(self, query_tokens: List[str]) -> np.ndarray:
        """Transform pre-tokenised query tokens into a TF-IDF vector.

        Args:
            query_tokens: Tokens already processed by the same pipeline
                used to fit the model.

        Returns:
            Dense 1-D numpy array of shape ``(vocab_size,)``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        query_text = " ".join(query_tokens)
        vec = self._vectorizer.transform([query_text])  # type: ignore[union-attr]
        vec = normalize(vec, norm="l2", copy=False)
        return vec.toarray().flatten()

    def search(
        self, query_tokens: List[str], top_k: int = 10
    ) -> List[Dict]:
        """Return the top-k documents by cosine similarity.

        Cosine similarity is computed as a single sparse matrix–vector
        multiply (O(nnz)) rather than looping over documents.

        Args:
            query_tokens: Pre-processed query tokens.
            top_k: Maximum number of results to return.

        Returns:
            List of ``{doc_id, score, rank}`` dicts sorted by descending score.
            Documents with zero similarity are excluded.
        """
        self._require_fitted()

        t0 = time.perf_counter()
        query_text = " ".join(query_tokens)
        query_vec = self._vectorizer.transform([query_text])  # type: ignore[union-attr]
        query_vec = normalize(query_vec, norm="l2", copy=False)

        # Shape: (1, n_docs) — matrix multiply query row against doc matrix.
        scores_sparse = query_vec.dot(self._matrix.T)  # type: ignore[union-attr]
        scores: np.ndarray = scores_sparse.toarray().flatten()

        # Efficient partial sort: only materialise top_k indices.
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [
            {
                "doc_id": self._doc_ids[int(idx)],
                "score": float(scores[idx]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(top_indices)
            if scores[idx] > 0
        ]

        logger.debug(
            "TF-IDF search: %d results in %.4fs.", len(results), time.perf_counter() - t0
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* using joblib.

        Args:
            path: Destination file path (e.g. ``models/tfidf.joblib``).

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        payload = {
            "vectorizer": self._vectorizer,
            "matrix": self._matrix,
            "doc_ids": self._doc_ids,
            "doc_id_to_idx": self._doc_id_to_idx,
            "max_features": self.max_features,
        }
        joblib.dump(payload, path, compress=3)
        logger.info("TF-IDF model saved to '%s'.", path)

    def load(self, path: str) -> None:
        """Deserialise a previously saved model from *path*.

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: '{path}'")
        payload = joblib.load(path)
        self._vectorizer = payload["vectorizer"]
        self._matrix = payload["matrix"]
        self._doc_ids = payload["doc_ids"]
        self._doc_id_to_idx = payload["doc_id_to_idx"]
        self.max_features = payload.get("max_features", self.max_features)
        logger.info(
            "TF-IDF model loaded from '%s': %d docs.", path, len(self._doc_ids)
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return summary statistics.

        Returns:
            Dict with keys ``vocab_size``, ``num_documents``, ``model_type``.
        """
        vocab_size = (
            len(self._vectorizer.vocabulary_)
            if self._vectorizer is not None
            else 0
        )
        return {
            "vocab_size": vocab_size,
            "num_documents": len(self._doc_ids),
            "model_type": "tfidf",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() before using this method."
            )