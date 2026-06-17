"""
Inverted index for IR system.

Stores term → {doc_id: term_frequency} mappings with document length
metadata needed for BM25-style ranking. Supports pickle serialisation
and incremental document addition.
"""

import logging
import os
import pickle
from collections import Counter
from typing import Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


class InvertedIndex:
    """In-memory inverted index with pickle persistence.

    Internal data structures
    ------------------------
    index : dict[str, dict[str, int]]
        ``{term: {doc_id: term_frequency}}``
    doc_lengths : dict[str, int]
        ``{doc_id: total_token_count}``
    doc_count : int
        Total number of indexed documents.
    avg_doc_length : float
        Mean token count across all documents (updated after every
        :meth:`build_from_dataset` call; updated incrementally by
        :meth:`add_document`).
    vocabulary : set[str]
        All unique terms in the index.

    Args:
        index_name: Logical name used as the default filename stem.
        storage_path: Directory where index files are saved / loaded.
    """

    def __init__(self, index_name: str, storage_path: str) -> None:
        self.index_name = index_name
        self.storage_path = storage_path

        self.index: Dict[str, Dict[str, int]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.doc_count: int = 0
        self.avg_doc_length: float = 0.0
        self.vocabulary: set = set()

        os.makedirs(storage_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, tokens: List[str]) -> None:
        """Add a single document's tokens to the index.

        Term frequencies are computed from *tokens*. If *doc_id* already
        exists in the index its previous contribution is silently
        overwritten.

        Args:
            doc_id: Unique document identifier.
            tokens: Pre-processed token list for this document.
        """
        if not tokens:
            logger.debug("Skipping empty document doc_id='%s'.", doc_id)
            return

        tf = Counter(tokens)
        doc_len = len(tokens)

        # Update postings lists.
        for term, freq in tf.items():
            self.index.setdefault(term, {})[doc_id] = freq

        self.vocabulary.update(tf.keys())
        self.doc_lengths[doc_id] = doc_len
        self.doc_count += 1

        # Keep a running average without recomputing from scratch.
        prev_total = self.avg_doc_length * (self.doc_count - 1)
        self.avg_doc_length = (prev_total + doc_len) / self.doc_count

    def build_from_dataset(self, documents: List[Dict]) -> None:
        """Build the index from scratch from a list of document dicts.

        Resets any existing index state before building.

        Args:
            documents: Each dict must have keys ``doc_id`` (str) and
                ``tokens`` (list[str]).
        """
        logger.info(
            "Building index '%s' from %d documents.", self.index_name, len(documents)
        )
        # Reset state.
        self.index = {}
        self.doc_lengths = {}
        self.doc_count = 0
        self.avg_doc_length = 0.0
        self.vocabulary = set()

        total_tokens = 0
        for doc in tqdm(documents, desc=f"Indexing '{self.index_name}'", unit="doc"):
            doc_id: str = doc["doc_id"]
            tokens: List[str] = doc.get("tokens", [])

            if not tokens:
                logger.debug("Skipping doc_id='%s' — no tokens.", doc_id)
                continue

            tf = Counter(tokens)
            doc_len = len(tokens)

            for term, freq in tf.items():
                self.index.setdefault(term, {})[doc_id] = freq

            self.vocabulary.update(tf.keys())
            self.doc_lengths[doc_id] = doc_len
            self.doc_count += 1
            total_tokens += doc_len

        self.avg_doc_length = total_tokens / self.doc_count if self.doc_count else 0.0
        logger.info(
            "Index built: %d docs, %d terms, avg_doc_length=%.2f",
            self.doc_count,
            len(self.vocabulary),
            self.avg_doc_length,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_postings(self, term: str) -> Dict[str, int]:
        """Return the postings list for a term.

        Args:
            term: Query term (should match the tokenisation used at
                index-build time, i.e. already lowercased / stemmed).

        Returns:
            ``{doc_id: term_frequency}`` or an empty dict if the term is
            not in the vocabulary.
        """
        return dict(self.index.get(term, {}))

    def get_document_frequency(self, term: str) -> int:
        """Return the number of documents that contain *term*.

        Args:
            term: Query term.

        Returns:
            Document frequency, or 0 if the term is absent.
        """
        return len(self.index.get(term, {}))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _default_path(self, filename: Optional[str]) -> str:
        name = filename or f"{self.index_name}.pkl"
        if not os.path.isabs(name):
            name = os.path.join(self.storage_path, name)
        return name

    def save(self, filename: Optional[str] = None) -> str:
        """Serialise the index to disk using pickle.

        Args:
            filename: Target file path. Defaults to
                ``<storage_path>/<index_name>.pkl``.

        Returns:
            The absolute path of the saved file.

        Raises:
            OSError: If the file cannot be written.
        """
        path = self._default_path(filename)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        payload = {
            "index_name": self.index_name,
            "index": self.index,
            "doc_lengths": self.doc_lengths,
            "doc_count": self.doc_count,
            "avg_doc_length": self.avg_doc_length,
            "vocabulary": self.vocabulary,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Index saved to '%s' (%d bytes).", path, os.path.getsize(path))
        return path

    def load(self, filename: Optional[str] = None) -> None:
        """Deserialise the index from disk.

        Args:
            filename: Source file path. Defaults to
                ``<storage_path>/<index_name>.pkl``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be deserialised.
        """
        path = self._default_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: '{path}'")

        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            raise ValueError(f"Failed to deserialise index from '{path}': {exc}") from exc

        self.index_name = payload["index_name"]
        self.index = payload["index"]
        self.doc_lengths = payload["doc_lengths"]
        self.doc_count = payload["doc_count"]
        self.avg_doc_length = payload["avg_doc_length"]
        self.vocabulary = payload["vocabulary"]

        logger.info(
            "Index loaded from '%s': %d docs, %d terms.",
            path,
            self.doc_count,
            len(self.vocabulary),
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, object]:
        """Return summary statistics for the index.

        Returns:
            Dict with keys:
            - ``vocab_size``: number of unique terms
            - ``doc_count``: number of indexed documents
            - ``avg_doc_length``: mean tokens per document
            - ``total_terms``: sum of all term frequencies across all documents
        """
        total_terms = sum(self.doc_lengths.values())
        return {
            "vocab_size": len(self.vocabulary),
            "doc_count": self.doc_count,
            "avg_doc_length": round(self.avg_doc_length, 4),
            "total_terms": total_terms,
        }
