"""
BM25 retrieval model using rank_bm25.BM25Okapi.

Supports per-query k1/b overrides and term-level score explanation
for UI parameter visualisation.
"""

import logging
import math
import pickle
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Model:
    """BM25Okapi retrieval model with explainability support.

    Args:
        k1: Term-frequency saturation parameter (default 1.5).
        b:  Document-length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        # Keep tokenised corpus for rebuild when params change.
        self._tokenised_corpus: List[List[str]] = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, documents: List[Dict]) -> None:
        """Build the BM25 index from pre-tokenised documents.

        Args:
            documents: List of dicts with ``doc_id`` (str) and ``tokens``
                (list[str]).
        """
        if not documents:
            raise ValueError("'documents' must not be empty.")

        logger.info("Fitting BM25 model on %d documents.", len(documents))
        self._doc_ids = [d["doc_id"] for d in documents]
        self._tokenised_corpus = [d.get("tokens", []) for d in documents]
        self._bm25 = BM25Okapi(self._tokenised_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 fit complete: %d docs.", len(self._doc_ids))

    def _bm25_with_params(self, k1: float, b: float) -> BM25Okapi:
        """Return a BM25Okapi instance with custom k1/b, reusing corpus."""
        if k1 == self.k1 and b == self.b:
            return self._bm25  # type: ignore[return-value]
        return BM25Okapi(self._tokenised_corpus, k1=k1, b=b)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_tokens: List[str],
        top_k: int = 10,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ) -> List[Dict]:
        """Return the top-k documents by BM25 score.

        Args:
            query_tokens: Pre-processed query tokens.
            top_k: Maximum number of results to return.
            k1: Per-query override for the k1 parameter.
            b:  Per-query override for the b parameter.

        Returns:
            List of ``{doc_id, score, rank}`` dicts, sorted by descending
            score.  Documents with zero score are excluded.
        """
        self._require_fitted()

        effective_k1 = k1 if k1 is not None else self.k1
        effective_b = b if b is not None else self.b
        bm25 = self._bm25_with_params(effective_k1, effective_b)

        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            {
                "doc_id": self._doc_ids[i],
                "score": float(scores[i]),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]

    def get_score_for_query(
        self,
        query_tokens: List[str],
        doc_id: str,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ) -> float:
        """Return the BM25 score for a specific document and query.

        Useful for hybrid model score fusion.

        Args:
            query_tokens: Pre-processed query tokens.
            doc_id: Target document identifier.
            k1: Optional per-call override.
            b:  Optional per-call override.

        Returns:
            BM25 score as a float, or 0.0 if *doc_id* is not in the index.
        """
        self._require_fitted()
        if doc_id not in self._doc_ids:
            return 0.0

        idx = self._doc_ids.index(doc_id)
        effective_k1 = k1 if k1 is not None else self.k1
        effective_b = b if b is not None else self.b
        bm25 = self._bm25_with_params(effective_k1, effective_b)
        scores = bm25.get_scores(query_tokens)
        return float(scores[idx])

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def update_params(self, k1: float, b: float) -> None:
        """Permanently update k1 and b and rebuild the BM25 index.

        Args:
            k1: New term-frequency saturation value.
            b:  New document-length normalisation value.
        """
        self.k1 = k1
        self.b = b
        if self._tokenised_corpus:
            logger.info("Rebuilding BM25 index with k1=%.2f, b=%.2f.", k1, b)
            self._bm25 = BM25Okapi(self._tokenised_corpus, k1=k1, b=b)

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain_params(
        self, query_tokens: List[str], doc_id: str
    ) -> Dict:
        """Return a per-term BM25 score breakdown for *doc_id*.

        Useful for visualising how k1 and b affect individual term
        contributions in the UI parameter explorer.

        Args:
            query_tokens: Pre-processed query tokens.
            doc_id: Target document identifier.

        Returns:
            Dict with keys:

            - ``doc_id``
            - ``k1``, ``b``: parameters in effect
            - ``terms``: list of per-term dicts::

                  {
                      "term": str,
                      "tf_in_doc": int,
                      "df_in_corpus": int,
                      "idf": float,
                      "bm25_contribution": float,
                  }

            - ``final_score``: sum of per-term contributions
        """
        self._require_fitted()

        if doc_id not in self._doc_ids:
            return {"doc_id": doc_id, "error": "Document not found in index."}

        doc_idx = self._doc_ids.index(doc_id)
        doc_tokens = self._tokenised_corpus[doc_idx]
        doc_len = len(doc_tokens)
        n = len(self._doc_ids)
        avg_dl = self._bm25.avgdl  # type: ignore[union-attr]

        term_details = []
        total_score = 0.0

        for term in set(query_tokens):
            tf = doc_tokens.count(term)
            df = sum(1 for doc in self._tokenised_corpus if term in doc)
            if df == 0:
                idf = 0.0
            else:
                # IDF formula used by rank_bm25.BM25Okapi
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / avg_dl
            )
            contribution = idf * (numerator / denominator) if denominator else 0.0
            total_score += contribution

            term_details.append(
                {
                    "term": term,
                    "tf_in_doc": tf,
                    "df_in_corpus": df,
                    "idf": round(idf, 6),
                    "bm25_contribution": round(contribution, 6),
                }
            )

        term_details.sort(key=lambda x: x["bm25_contribution"], reverse=True)

        return {
            "doc_id": doc_id,
            "k1": self.k1,
            "b": self.b,
            "terms": term_details,
            "final_score": round(total_score, 6),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise model to *path* using pickle.

        Args:
            path: Destination file path.
        """
        self._require_fitted()
        payload = {
            "k1": self.k1,
            "b": self.b,
            "bm25": self._bm25,
            "doc_ids": self._doc_ids,
            "tokenised_corpus": self._tokenised_corpus,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 model saved to '%s'.", path)

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
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        self.k1 = payload["k1"]
        self.b = payload["b"]
        self._bm25 = payload["bm25"]
        self._doc_ids = payload["doc_ids"]
        self._tokenised_corpus = payload.get("tokenised_corpus", [])
        logger.info(
            "BM25 model loaded from '%s': %d docs.", path, len(self._doc_ids)
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return summary statistics.

        Returns:
            Dict with keys ``num_documents``, ``k1``, ``b``,
            ``avg_doc_length``, ``model_type``.
        """
        return {
            "num_documents": len(self._doc_ids),
            "k1": self.k1,
            "b": self.b,
            "avg_doc_length": round(self._bm25.avgdl, 4) if self._bm25 else 0.0,
            "model_type": "bm25",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._bm25 is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() before using this method."
            )